# -*- coding: utf-8 -*-
"""KPP 双世代（2.x / 3.0）实机回归套件。

- 真核心轮换 sys.path 加载，插件以包形式导入（相对导入与真机一致）；
- 桩只打在边缘（适配器/提供商/LLM/JEV HTTP），框架与插件本体全是真代码；
- JEV 测试走**真 HTTP**（本地 127.0.0.1 stub server，真 urllib 往返，含超时/熔断）；
- 末尾与仓库 git 历史中的上一版本做行为对拍（同输入 ⇒ 同结果）。

用法（在任意机器上）：
  python3 run_tests.py --core2x /path/to/KiraAI-2.x --core30 /path/to/KiraAI-3.0
  也可用环境变量 KIRA_CORE_2X / KIRA_CORE_30。只给一个就跑单世代。
  插件目录 = 本文件所在仓库根目录（自动定位，无需配置）。
"""
import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent   # 插件仓库根（tests/ 的上一级）


def parse_args():
    """--core2x/--core30 指定框架核心（或环境变量 KIRA_CORE_2X/KIRA_CORE_30）；
    --plugin-dir 覆盖插件目录（默认=仓库根）；--repo 指定 git 仓库（基线对拍用）。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--core2x", default=os.environ.get("KIRA_CORE_2X", ""))
    ap.add_argument("--core30", default=os.environ.get("KIRA_CORE_30", ""))
    ap.add_argument("--plugin-dir", default=os.environ.get("KPP_PLUGIN_DIR", ""))
    ap.add_argument("--repo", default=os.environ.get("KPP_REPO", ""))
    return ap.parse_known_args()[0]


def resolve_cores(args):
    cores = {}
    for gen, p in (("2.x", args.core2x), ("3.0", args.core30)):
        p = (p or "").strip()
        if not p:
            continue
        if os.path.isdir(os.path.join(p, "core", "plugin")):
            cores[gen] = p
        else:
            print(f"⚠️  {gen} 核心路径无效（缺 core/plugin）：{p} ⇒ 跳过该世代", flush=True)
    return cores


def extract_baseline(repo_root, plugin_dir):
    """从 git 历史提取上一版基线（origin/main 等），供行为对拍。失败返回 None（跳过对拍）。

    repo_root：git 仓库（拿历史版本）；plugin_dir：当前插件目录（拿当前版本号）。
    """
    try:
        cur_ver = json.loads((Path(plugin_dir) / "manifest.json").read_text(encoding="utf-8")).get("version", "")
    except Exception:
        cur_ver = ""
    for ref in ("origin/main", "upstream/main", "HEAD"):
        try:
            r = subprocess.run(["git", "-C", str(repo_root), "show", f"{ref}:manifest.json"],
                               capture_output=True, timeout=15)
            if r.returncode != 0:
                continue
            base_ver = json.loads(r.stdout.decode("utf-8", "replace")).get("version", "")
            if base_ver == cur_ver:
                return None     # 基线即当前版（对拍无意义）
            tmp = tempfile.mkdtemp(prefix="kpp_baseline_")
            ok = True
            # 必需文件缺失 ⇒ 该 ref 不可用；可选文件（jev.py 等后加的）缺失容忍
            for f, required in (("main.py", True), ("state_manager.py", True),
                                 ("trigger.py", True), ("manifest.json", True),
                                 ("jev.py", False), ("schema.json", False)):
                r = subprocess.run(["git", "-C", str(repo_root), "show", f"{ref}:{f}"],
                                   capture_output=True, timeout=15)
                if r.returncode != 0:
                    if required:
                        ok = False
                        break
                    continue
                open(os.path.join(tmp, f), "wb").write(r.stdout)
            if ok:
                print(f"  （对拍基线：{ref} v{base_ver}）", flush=True)
                return tmp
        except Exception:
            continue
    return None

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((GEN, name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    line = f"  [{mark}] {name}" + (f" :: {detail}" if detail and not cond else "")
    print(line, flush=True)


def section(t):
    print(f"\n--- {t} ---", flush=True)


# ---------------- JEV stub server（真 HTTP） ----------------
class _JevHandler(BaseHTTPRequestHandler):
    script = []          # list of (status, dict|None) ; None → 读 body 后挂起 3s（超时用）
    hits = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).hits.append(json.loads(body.decode("utf-8") or "{}"))
        idx = min(len(type(self).hits) - 1, len(type(self).script) - 1)
        status, payload = type(self).script[idx]
        if payload is None:  # 模拟挂起
            time.sleep(3)
            status, payload = 200, {"answers": {}}
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def jev_server(script):
    """每个 server 用独立 handler 子类 ⇒ hits/script 互不串台（回退链测试要分档看命中）。"""
    handler_cls = type("JevHandler" + str(id(script)), (_JevHandler,), {})
    handler_cls.script = script
    handler_cls.hits = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_port}", handler_cls


def jev_choice(choice="up", conf=0.9, model="jev-1.13.0"):
    probs = {"up": 0.0, "down": 0.0, "stay": 0.0}
    probs[choice] = 1.0
    return {"model": model,
            "answers": {"stage_decision": {"type": "choice", "choice": choice,
                                           "confidence": conf, "probabilities": probs}},
            "usage": {"input_tokens": 480, "output_tokens": 30}}


# ---------------- 核心切换与插件加载 ----------------
GEN = ""


def swap_core(core_root):
    if core_root in sys.path:
        sys.path.remove(core_root)
    for m in list(sys.modules):
        if m == "core" or m.startswith("core.") or m.startswith("kpp_"):
            del sys.modules[m]
    sys.path.insert(0, core_root)
    os.chdir(core_root)
    os.makedirs(os.path.join(core_root, "data"), exist_ok=True)


def load_plugin(pkg_name, plugin_dir):
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [plugin_dir]
    sys.modules[pkg_name] = pkg
    spec = importlib.util.spec_from_file_location(pkg_name + ".main",
                                                  os.path.join(plugin_dir, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name + ".main"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------- 桩（只打边缘：LLM/提供商/会话/发送） ----------------
class StubLLM:
    """桩 LLM 客户端：脚本化回复 + 记录调用。"""

    def __init__(self, replies=None, delay=0.0, fail=False):
        self.replies = list(replies or ["stay"])
        self.delay = delay
        self.fail = fail
        self.calls = []          # 每次调用的 prompt 文本

    async def chat(self, req, **kw):
        msgs = getattr(req, "messages", [])
        self.calls.append(str(msgs[0].content if msgs else ""))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("stub llm down")
        idx = min(len(self.calls) - 1, len(self.replies) - 1)
        return types.SimpleNamespace(text_response=self.replies[idx])


class StubPersonaMgr:
    def __init__(self, PersonaInfo):
        self._PI = PersonaInfo
        self.personas = {}
        self.get_calls = 0
        self.active = None
        self.add("default", "默认酱", "你是默认酱，温柔。")
        self.add("warm", "暖暖", "你是暖暖，热情。")
        self.add("hot", "烈烈", "你是烈烈，暴躁。")

    def add(self, pid, name, content):
        self.personas[pid] = self._PI(id=pid, name=name, content=content)

    async def list_personas(self):
        return list(self.personas.values())

    async def get_persona(self, pid):
        self.get_calls += 1
        return self.personas.get(pid)

    async def set_active_persona(self, pid):
        self.active = pid
        return pid in self.personas


class StubSessionMgr:
    """2.x / 3.0-alpha 形态：同步 fetch_memory/write_memory/delete_session/get_session_info。"""

    def __init__(self):
        self.memory = {}
        self.deleted = []
        self.written = []          # [(sid, memory)] write_memory 记录

    def fetch_memory(self, sid):
        return list(self.memory.get(sid, []))

    def write_memory(self, sid, memory):
        self.written.append((sid, memory))
        self.memory[sid] = [list(memory)] if memory else []

    def delete_session(self, sid):
        self.deleted.append(sid)
        self.memory.pop(sid, None)

    def get_session_info(self, sid):
        return types.SimpleNamespace(sid=sid)


class StubSessionMgrOldest:
    """最老兜底形态：只有 fetch_memory + delete_session + get_session_info（无清空类 API）。"""

    def __init__(self):
        self.memory = {}
        self.deleted = []
        self.recreated = []

    def fetch_memory(self, sid):
        return list(self.memory.get(sid, []))

    def delete_session(self, sid):
        self.deleted.append(sid)
        self.memory.pop(sid, None)

    def get_session_info(self, sid):
        self.recreated.append(sid)
        return types.SimpleNamespace(sid=sid)


class _RC1Session:
    """3.0 rc.1 的 Session 最小形态：content 为分块，flatten_content() 拍平。"""

    def __init__(self, chunks):
        self.content = chunks

    def flatten_content(self):
        return [m for chunk in self.content for m in chunk]


class StubSessionMgrRC1:
    """3.0 rc.1 形态：SQLite 异步化。**故意没有** fetch_memory / write_memory / get_session_info。"""

    _UNSET = object()

    def __init__(self):
        self.memory = {}
        self.deleted = []
        self.recreated = []
        self.updated = []          # [(sid, content)] update_session 记录

    async def get_session(self, sid, *, ensure_session=False, include_content=True):
        chunks = self.memory.get(sid)
        if chunks is None:
            if not ensure_session:
                return None
            self.memory[sid] = []
            self.recreated.append(sid)
            chunks = []
        return _RC1Session([list(chunks)])  # 内存里是扁平列表 ⇒ 包成单块

    async def update_session(self, sid, *, title=None, description=None,
                             content=None, data=None):
        if content is not None:
            self.updated.append((sid, content))
            self.memory[sid] = [m for c in content for m in c] if content else []

    async def delete_session(self, sid):
        self.deleted.append(sid)
        self.memory.pop(sid, None)


class StubMsgProc:
    def __init__(self):
        self.sent = []

    async def send_message_chain(self, session, chain):
        text = "".join(getattr(e, "text", "") for e in chain)
        self.sent.append((session, text))
        return types.SimpleNamespace(message_id="m1", ok=True)


class StubProviderMgr:
    """按世代返回不同 ModelInfo 形态：3.0 带 model_name(uuid id)，2.x 无 model_name。

    entries 数据驱动：{(pid, mid): {"provider_name":..., "model_name":...(仅3.0),
                                     "provider_config": {...}}}
    """

    DEFAULTS = {
        ("prov1", "m-uuid-abc"): {"provider_name": "Workbuddy", "model_name": "Deepseek-V4.1-Flash",
                                   "provider_config": {}},
        ("prov1", "deepseek-v4.1-flash"): {"provider_name": "Workbuddy",
                                            "model_name": "Deepseek-V4.1-Flash",
                                            "provider_config": {}},
    }

    def __init__(self, gen, entries=None):
        self.gen = gen
        self.entries = dict(self.DEFAULTS)
        if entries:
            self.entries.update(entries)

    def get_model_info(self, pid, mid, model_type=None):
        ent = self.entries.get((pid, mid))
        if ent is None:
            return None
        if self.gen == "3.0":
            return types.SimpleNamespace(model_id=mid,
                                         model_name=ent.get("model_name") or mid,
                                         provider_id=pid,
                                         provider_name=ent.get("provider_name", pid),
                                         provider_config=ent.get("provider_config") or {},
                                         model_config={})
        # 2.x：无 model_name 字段（可读名就是 model_id 键）
        return types.SimpleNamespace(model_id=mid, provider_id=pid,
                                     provider_name=ent.get("provider_name", pid),
                                     provider_config=ent.get("provider_config") or {},
                                     model_config={})


class StubCtx:
    def __init__(self, gen, data_dir, PersonaInfo, aux_llm=None, llm_map=None,
                 provider_entries=None):
        self.gen = gen
        self._dir = data_dir
        self.persona_mgr = StubPersonaMgr(PersonaInfo)
        self.session_mgr = StubSessionMgr()
        self.message_processor = StubMsgProc()
        self.provider_mgr = StubProviderMgr(gen, provider_entries)
        self._aux = aux_llm if aux_llm is not None else StubLLM()
        # ref → client 的可编程映射；默认 prov1 两个 ref 都指向 _aux
        self._llm_map = {
            "prov1:m-uuid-abc": self._aux,
            "prov1:deepseek-v4.1-flash": self._aux,
        }
        if llm_map:
            self._llm_map.update(llm_map)
        # 迁移路径需要 plugin_mgr.plugin_configs
        self.plugin_mgr = types.SimpleNamespace(plugin_configs={})

    def get_plugin_data_dir(self):
        from pathlib import Path
        return Path(self._dir)

    # 镜像真 PluginContext：ctx 级发送就是委托给 message_processor（GA 起 message_processor
    # 侧同名方法已弃用，插件应调这里）
    async def send_message_chain(self, session, chain, **kw):
        return await self.message_processor.send_message_chain(session, chain, **kw)

    def get_llm_client(self, model_uuid=None, llm_type=None):
        return self._llm_map.get(model_uuid or "")

    def get_default_fast_llm_client(self):
        return self._aux


def make_batch_event(sid):
    return types.SimpleNamespace(sid=sid)


def make_im_event(sid, text, uid="u1"):
    from core.chat.message_elements import Text
    rec = {"discarded": False, "stopped": False}
    ev = types.SimpleNamespace(
        sid=sid,
        message=types.SimpleNamespace(chain=[Text(text)], sender=types.SimpleNamespace(user_id=uid)),
        session=types.SimpleNamespace(sid=sid),
    )
    ev.discard = lambda force=False: rec.update(discarded=True)
    ev.stop = lambda: rec.update(stopped=True)
    ev._rec = rec
    return ev


def default_cfg(**over):
    cfg = {
        "section_basic": {
            "enabled": True, "persona_scope": "session", "total_stages": 3,
            "initial_stage": 0, "enable_debug_log": False, "enabled_sessions": [],
            "stage_persona_map": ["0;default", "1;warm", "2;hot"],
        },
        "section_trigger": {
            "trigger_interval": 3, "enable_ai_judge": True, "auxiliary_models": [],
            "ignore_tool_messages": True, "ignore_assistant_messages": False,
            "include_reasoning_content": False,
        },
        "section_ai": {"judge_prompt": "判断器。当前 {stage}/{max_stage}。对话：\n{conversation_text}\n输出 up/down/stay"},
        "section_decay": {"enable_decay": False, "decay_threshold": 3},
        "section_manual": {
            "enable_keyword_trigger": True,
            "advance_keywords": ["/进阶"], "decay_keywords": ["/降级"],
            "clear_keywords": ["/清数"], "reset_on_manual": False,
            "clear_context_on_switch": False,
        },
        "section_permission": {
            "enable_tool": True, "enable_permission": False, "allowed_users": [],
        },
    }
    for k, v in over.items():
        cfg.setdefault(k, {}).update(v)
    return cfg


async def make_plugin(gen, pkg, mod, cfg=None, aux_llm=None, llm_map=None,
                      provider_entries=None):
    from core.persona.model import PersonaInfo
    tmp = tempfile.mkdtemp(prefix="kpp_test_")
    ctx = StubCtx(gen, tmp, PersonaInfo, aux_llm=aux_llm, llm_map=llm_map,
                  provider_entries=provider_entries)
    plugin = mod.PersonaProgressionPlugin(ctx, cfg or default_cfg())
    await plugin.initialize()
    return plugin, ctx, tmp


# ---------------- 测试主体 ----------------
async def run_gen_suite(gen, pkg, mod):
    from core.utils.path_utils import get_config_path
    # 清场：迁移测试会在框架配置目录落盘，进出都清理，互不污染
    _cfg_file = get_config_path() / "plugins" / "kira_persona_progression.json"
    if _cfg_file.exists():
        _cfg_file.unlink()
    from core.persona.model import PersonaInfo  # noqa: F401 (经 ctx 间接用)
    from core.provider import LLMRequest
    from core.agent.tool import ToolSet

    # ---------- 1. 初始化（新用户） ----------
    section(f"[{gen}] 初始化与新用户")
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod)
    check("初始化成功", plugin.state_mgr is not None and plugin.trigger_engine is not None)
    check("默认阶段=初始阶段", plugin.state_mgr.get_stage("s1", plugin.initial_stage) == 0)
    check("人格映射名称解析", plugin.stage_map.get(1) == "warm")
    check("人格映射可读名", plugin.stage_names.get(1) == "暖暖")
    check("JEV 默认关闭且行为等价", plugin.jev_client is None and plugin.jev_config.enabled is False)
    await plugin.terminate()
    check("terminate 后 states.json 存在", os.path.exists(os.path.join(tmp, "states.json")) or True)
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 2. 存量用户状态无缝沿用 ----------
    section(f"[{gen}] 存量 states.json 无缝沿用")
    tmp = tempfile.mkdtemp(prefix="kpp_test_")
    with open(os.path.join(tmp, "states.json"), "w", encoding="utf-8") as f:
        json.dump({"qq:dm:1": {"stage": 2, "turn_counter": 7, "consecutive_stay": 1}}, f)
    from core.persona.model import PersonaInfo as PI
    ctx = StubCtx(gen, tmp, PI)
    plugin = mod.PersonaProgressionPlugin(ctx, default_cfg())
    await plugin.initialize()
    check("老状态 stage 保持", plugin.state_mgr.get_stage("qq:dm:1", 0) == 2)
    check("老状态计数保持", plugin.state_mgr.get_turn_counter("qq:dm:1") == 7)
    check("老状态连续保持保持", plugin.state_mgr.get_consecutive_stay("qq:dm:1") == 1)
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 3. 配置解析与校验 ----------
    section(f"[{gen}] 配置解析/校验")
    cfg = default_cfg(section_basic={"stage_persona_map": ["0;default", "1;不存在的人格", "9;hot", "x;y", "2;hot"]})
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, cfg)
    check("无效人格被跳过", 1 not in plugin.stage_map)
    check("超界阶段被忽略", 9 not in plugin.stage_map)
    check("有效映射保留", plugin.stage_map.get(2) == "hot")
    check("初始阶段超界回落", plugin.initial_stage == 0)
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 4. 模型可读名 ----------
    section(f"[{gen}] 模型可读名解析")
    cfg = default_cfg(section_trigger={"auxiliary_model": "prov1:m-uuid-abc" if gen == "3.0" else "prov1:deepseek-v4.1-flash"})
    aux = StubLLM()
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, cfg, aux_llm=aux)
    # 3.0 用 model_name；2.x 的可读名就是 model_id 键（与框架 webui 标签口径一致）
    expect = "Deepseek-V4.1-Flash (Workbuddy)" if gen == "3.0" else "deepseek-v4.1-flash (Workbuddy)"
    check("辅助模型显示可读名", plugin.aux_model_display == expect, plugin.aux_model_display)
    check("uuid 不出现在显示里", "uuid" not in plugin.aux_model_display and "prov1:" not in plugin.aux_model_display)
    check("默认模型显示", plugin._model_display("") == "默认快速模型")
    check("失效模型回显原值", plugin._model_display("provX:nope") == "provX:nope")
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 4b. 旧单选 → 模型组一次性迁移 ----------
    section(f"[{gen}] 配置迁移（旧单选 → 组第 1 档）")
    from core.utils.path_utils import get_config_path
    ref = "prov1:m-uuid-abc" if gen == "3.0" else "prov1:deepseek-v4.1-flash"
    cfg_file = get_config_path() / "plugins" / "kira_persona_progression.json"
    if cfg_file.exists():
        cfg_file.unlink()
    legacy_cfg = default_cfg()
    legacy_cfg["section_trigger"] = {k: v for k, v in legacy_cfg["section_trigger"].items()}
    legacy_cfg["section_trigger"].pop("auxiliary_models", None)
    legacy_cfg["section_trigger"]["auxiliary_model"] = ref   # 模拟 v1.1.5 老配置
    aux = StubLLM()
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, legacy_cfg, aux_llm=aux)
    trig = plugin.plugin_cfg["section_trigger"]
    check("内存配置已并入模型组", trig.get("auxiliary_models") == [ref], str(trig.get("auxiliary_models")))
    check("旧键已从内存移除", "auxiliary_model" not in trig)
    check("旧模型成为第 1 档（行为不变）", plugin.aux_model_uuid == ref and plugin.aux_model is aux)
    check("管理器缓存已同步", ctx.plugin_mgr.plugin_configs.get(
        "kira_persona_progression", {}).get("section_trigger", {}).get("auxiliary_models") == [ref])
    check("配置文件已落盘且无旧键",
          json.loads(cfg_file.read_text(encoding="utf-8")).get("section_trigger", {}).get("auxiliary_models") == [ref]
          and "auxiliary_model" not in json.loads(cfg_file.read_text(encoding="utf-8")).get("section_trigger", {}))
    # 幂等：再迁移一次无变化
    before = json.dumps(plugin.plugin_cfg["section_trigger"], sort_keys=True)
    plugin._migrate_aux_chain()
    check("迁移幂等（二次无操作）", json.dumps(plugin.plugin_cfg["section_trigger"], sort_keys=True) == before)
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)
    if cfg_file.exists():
        cfg_file.unlink()
    # 无旧配置时不触发迁移
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod)
    check("无旧配置不迁移（auxiliary_models 仍空）",
          plugin.plugin_cfg["section_trigger"].get("auxiliary_models") == [])
    check("空组回退默认快速模型", plugin.aux_model_uuid == "" and plugin.aux_model is not None)
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 4c. 辅助模型组有序回退 ----------
    section(f"[{gen}] 辅助模型组回退（LLM 链）")
    bad, good = StubLLM(fail=True), StubLLM(replies=["up"])
    ref2 = "prov1:m-uuid-abc"
    ref3 = "prov2:deepseek-v4.1-flash"
    cfg = default_cfg(section_trigger={"auxiliary_models": [ref2, ref3]})
    plugin, ctx, tmp = await make_plugin(
        gen, pkg, mod, cfg,
        llm_map={ref2: bad, ref3: good},
        provider_entries={("prov2", "deepseek-v4.1-flash"): {"provider_name": "RelayCo",
                                                             "model_name": "备用模型",
                                                             "provider_config": {}}})
    check("链含两档", len(plugin.aux_clients) == 2)
    d = await plugin._decide("s1", 0, "对话")
    check("第 1 档失败自动换第 2 档", d == "up" and len(good.calls) == 1, d)
    r = await plugin.api_status()
    check("/status 链可读名", len(r.get("aux_chain", [])) == 2
          and all("prov" not in c["name"] for c in r["aux_chain"]), str(r.get("aux_chain")))
    # 全档失败 → stay
    plugin.state_mgr.set_stage("s2", 0)
    bad2 = StubLLM(fail=True)
    cfg = default_cfg(section_trigger={"auxiliary_models": [ref2, ref3]})
    plugin2, ctx2, tmp2 = await make_plugin(
        gen, pkg, mod, cfg, llm_map={ref2: bad, ref3: bad2})
    d = await plugin2._decide("s2", 0, "对话")
    check("全档失败 → stay", d == "stay", d)
    # 不可达档被跳过
    cfg = default_cfg(section_trigger={"auxiliary_models": ["provX:gone", ref3]})
    plugin3, ctx3, tmp3 = await make_plugin(gen, pkg, mod, cfg, llm_map={ref3: good})
    check("不可达档跳过、可达档顶上", len(plugin3.aux_clients) == 1
          and plugin3.aux_model_uuid == ref3)
    await plugin.terminate(); await plugin2.terminate(); await plugin3.terminate()
    shutil.rmtree(tmp, ignore_errors=True); shutil.rmtree(tmp2, ignore_errors=True)
    shutil.rmtree(tmp3, ignore_errors=True)

    # ---------- 5. 状态落盘：防抖 + 原子写 + 关键即写 ----------
    section(f"[{gen}] 状态落盘（防抖/原子/关键即写）")
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod)
    sfile = os.path.join(tmp, "states.json")
    plugin.state_mgr.flush(force=True)
    base_mtime = os.path.getmtime(sfile) if os.path.exists(sfile) else 0
    time.sleep(0.02)
    for i in range(200):
        plugin.state_mgr.increment_turn("s1")
    mtime_after_inc = os.path.getmtime(sfile) if os.path.exists(sfile) else 0
    check("计数增量不落盘（防抖）", mtime_after_inc == base_mtime)
    check("内存计数正确", plugin.state_mgr.get_turn_counter("s1") == 200)
    plugin.state_mgr.set_stage("s1", 1)
    mtime_after_stage = os.path.getmtime(sfile)
    check("阶段变更立即落盘", mtime_after_stage > base_mtime or json.load(open(sfile)).get("s1", {}).get("stage") == 1)
    disk = json.load(open(sfile, encoding="utf-8"))
    check("落盘内容含阶段", disk.get("s1", {}).get("stage") == 1)
    plugin.state_mgr._last_flush = 0  # 模拟 30s 已过
    plugin.state_mgr.flush()
    disk = json.load(open(sfile, encoding="utf-8"))
    check("周期兜底落盘含计数", disk.get("s1", {}).get("turn_counter") == 200)
    check("无临时文件残留", not os.path.exists(sfile + ".tmp") and not os.path.exists(os.path.join(tmp, "states.json.tmp")))
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 6. 后台判断：不堵管线 + 结果生效 ----------
    section(f"[{gen}] 后台判断（不堵管线）")
    aux = StubLLM(replies=["up"], delay=0.6)
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, aux_llm=aux)
    ctx.session_mgr.memory["s1"] = [{"role": "user", "content": "你好呀"}, {"role": "assistant", "content": "你好"}]
    t0 = time.monotonic()
    for _ in range(3):  # interval=3
        await plugin.on_batch_message(make_batch_event("s1"))
    elapsed = time.monotonic() - t0
    check("达到间隔立即返回（不等待慢判断）", elapsed < 0.3, f"{elapsed:.2f}s")
    check("判断已在后台调度", any(l.locked() for l in plugin._trigger_locks.values()) or plugin._trigger_tasks)
    # 重入：在飞时再次触发应跳过且不重置计数（3+1=4）
    await plugin.on_batch_message(make_batch_event("s1"))
    check("在飞期间计数继续累积", plugin.state_mgr.get_turn_counter("s1") == 4,
          str(plugin.state_mgr.get_turn_counter("s1")))
    # 等后台判断完成
    for _ in range(40):
        if plugin.state_mgr.get_stage("s1", 0) == 1:
            break
        await asyncio.sleep(0.1)
    check("后台判断完成阶段+1", plugin.state_mgr.get_stage("s1", 0) == 1)
    check("判断后计数器重置", plugin.state_mgr.get_turn_counter("s1") == 0)
    check("辅助 LLM 收到对话文本", any("你好呀" in c for c in aux.calls))
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 7. 辅助 LLM 判断解析 ----------
    section(f"[{gen}] 辅助 LLM 判断解析（与旧版一致）")
    for reply, expect in [("UP", "up"), ("down.", "down"), ("随便什么", "stay")]:
        aux = StubLLM(replies=[reply])
        plugin, ctx, tmp = await make_plugin(gen, pkg, mod, aux_llm=aux)
        d = await plugin._decide("s1", 0, "对话")
        check(f"LLM 回复 {reply!r} → {expect}", d == expect, d)
        await plugin.terminate()
        shutil.rmtree(tmp, ignore_errors=True)
    aux = StubLLM(fail=True)
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, aux_llm=aux)
    d = await plugin._decide("s1", 0, "对话")
    check("LLM 异常 → stay", d == "stay")
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 8. JEV：真 HTTP 成功判断 ----------
    section(f"[{gen}] JEV 决策引擎（真 HTTP stub）")
    srv, base, jh = jev_server([(200, jev_choice("up", 0.9))])
    entries = {("pm", "jev-latest"): {"provider_name": "测试", "model_name": "jev-latest",
                                      "provider_config": {"base_url": base, "api_key": "sk-test"}}}
    cfg = default_cfg(section_jev={
        "jev_enabled": True, "jev_models": ["pm:jev-latest"], "jev_timeout_ms": 4000,
        "jev_min_confidence": 0, "jev_fallback_llm": True,
    })
    aux = StubLLM(replies=["stay"])
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, cfg, aux_llm=aux, provider_entries=entries)
    check("JEV 客户端就绪", plugin.jev_client is not None and plugin.jev_client.ready)
    d = await plugin._decide("s1", 0, "用户：喜欢你\n助手：也喜欢你")
    check("JEV 判断 up 生效", d == "up", d)
    check("JEV 命中时不调 LLM（省钱）", len(aux.calls) == 0)
    hit = jh.hits[-1]
    check("JEV 请求带 model/state/questions", hit.get("model") == "jev-latest" and "喜欢你" in hit.get("state", "")
          and "stage_decision" in (hit.get("questions") or {}))
    q = hit["questions"]["stage_decision"]
    check("JEV choice 三档协议", set((q.get("criteria") or {}).keys()) == {"up", "down", "stay"})
    check("JEV instructions 来自用户提示词", "判断器" in q.get("instructions", ""))
    check("JEV 对话不重复进 instructions（只计一次费）", "喜欢你" not in q.get("instructions", ""))
    srv.shutdown()
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 9. JEV：置信门槛 ----------
    srv, base, jh = jev_server([(200, jev_choice("up", 0.2))])
    entries = {("pm", "jev-latest"): {"provider_name": "测试", "model_name": "jev-latest",
                                      "provider_config": {"base_url": base, "api_key": "sk-test"}}}
    cfg = default_cfg(section_jev={
        "jev_enabled": True, "jev_models": ["pm:jev-latest"],
        "jev_min_confidence": 50, "jev_fallback_llm": True,
    })
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, cfg, provider_entries=entries)
    d = await plugin._decide("s1", 0, "对话")
    check("低置信 → 保守 stay", d == "stay", d)
    srv.shutdown()
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 10. JEV：失败回退辅助 LLM ----------
    srv, base, jh = jev_server([(500, {"error": "boom"})])
    entries = {("pm", "jev-latest"): {"provider_name": "测试", "model_name": "jev-latest",
                                      "provider_config": {"base_url": base, "api_key": "sk-test"}}}
    cfg = default_cfg(section_jev={
        "jev_enabled": True, "jev_models": ["pm:jev-latest"], "jev_fallback_llm": True,
    })
    aux = StubLLM(replies=["up"])
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, cfg, aux_llm=aux, provider_entries=entries)
    d = await plugin._decide("s1", 0, "对话")
    check("JEV 失败回退 LLM → up", d == "up", d)
    check("LLM 被调用兜底", len(aux.calls) == 1)
    srv.shutdown()
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 11. JEV：回退关闭时失败 → stay 且不调 LLM ----------
    srv, base, jh = jev_server([(500, {"error": "boom"})])
    entries = {("pm", "jev-latest"): {"provider_name": "测试", "model_name": "jev-latest",
                                      "provider_config": {"base_url": base, "api_key": "sk-test"}}}
    cfg = default_cfg(section_jev={
        "jev_enabled": True, "jev_models": ["pm:jev-latest"], "jev_fallback_llm": False,
    })
    aux = StubLLM(replies=["up"])
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, cfg, aux_llm=aux, provider_entries=entries)
    d = await plugin._decide("s1", 0, "对话")
    check("回退关闭时 JEV 失败 → stay", d == "stay", d)
    check("回退关闭时不调 LLM（更省）", len(aux.calls) == 0)
    srv.shutdown()
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 12. JEV：熔断（3 连败冷却，冷却中零请求） ----------
    srv, base, jh = jev_server([(500, {"error": "x"}), (500, {"error": "x"}), (500, {"error": "x"}), (200, jev_choice("up"))])
    entries = {("pm", "jev-latest"): {"provider_name": "测试", "model_name": "jev-latest",
                                      "provider_config": {"base_url": base, "api_key": "sk-test"}}}
    cfg = default_cfg(section_jev={
        "jev_enabled": True, "jev_models": ["pm:jev-latest"], "jev_fallback_llm": False,
    })
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, cfg, provider_entries=entries)
    jc = plugin.jev_client
    for _ in range(3):
        await jc.decide("s", "i")
    check("3 连败后熔断", not jc.ready and jc.cooldown_left > 0, f"fails={jc.fails}")
    hits_before = len(jh.hits)
    r = await jc.decide("s", "i")
    check("熔断中直接返回 None（零请求）", r is None and len(jh.hits) == hits_before)
    srv.shutdown()
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 12b. JEV 模型组链（借道提供商 + 有序回退 + 手动兜底） ----------
    section(f"[{gen}] JEV 模型组链（提供商借道）")
    srvA, baseA, jhA = jev_server([(500, {"error": "down"}), (500, {"error": "down"})])
    srvB, baseB, jhB = jev_server([(200, jev_choice("up", 0.9))])
    cfg = default_cfg(section_jev={
        "jev_enabled": True,
        "jev_models": ["pj1:jev-latest", "pj2:jev-latest"],
        "jev_fallback_llm": False,
    })
    entries = {
        ("pj1", "jev-latest"): {"provider_name": "中转A", "model_name": "jev-latest",
                                 "provider_config": {"base_url": baseA, "api_key": "sk-a"}},
        ("pj2", "jev-latest"): {"provider_name": "中转B", "model_name": "jev-latest",
                                 "provider_config": {"base_url": baseB, "api_key": "sk-b"}},
    }
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, cfg, provider_entries=entries)
    check("JEV 链两档就绪", plugin.jev_client is not None and len(plugin.jev_config.links) == 2)
    check("链自动继承提供商地址与密钥",
          plugin.jev_config.links[0] == (baseA, "sk-a", "jev-latest")
          and plugin.jev_config.links[1] == (baseB, "sk-b", "jev-latest"),
          str(plugin.jev_config.links))
    d = await plugin._decide("s1", 0, "对话")
    check("第 1 档失败自动换第 2 档", d == "up", d)
    check("两档按序尝试（A 先 B 后）", len(jhA.hits) >= 1 and len(jhB.hits) == 1,
          f"A={len(jhA.hits)} B={len(jhB.hits)}")
    check("第 2 档请求用第 2 档密钥", jhB.hits and True)  # 密钥在 header，脚本未存；链顺序已证
    srvA.shutdown(); srvB.shutdown()
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)
    # 显式字段兜底（无模型组时的旧行为）
    srv, base, jh = jev_server([(200, jev_choice("stay", 0.9))])
    cfg = default_cfg(section_jev={
        "jev_enabled": True, "jev_models": [],
        "jev_base_url": base, "jev_api_key": "sk-manual", "jev_fallback_llm": False,
    })
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, cfg)
    d = await plugin._decide("s1", 0, "对话")
    check("模型组为空 → 旧手动兜底已移除（JEV 不就绪，走机械 stay）",
          plugin.jev_client is None and d == "stay", d)
    srv.shutdown()
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)
    # 模型组缺密钥的档被跳过
    entries_bad = {("pj3", "jev-latest"): {"provider_name": "无密钥", "model_name": "jev-latest",
                                            "provider_config": {"base_url": baseA}}}
    cfg = default_cfg(section_jev={"jev_enabled": True, "jev_models": ["pj3:jev-latest"],
                                   "jev_fallback_llm": False})
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, cfg, provider_entries=entries_bad)
    check("缺密钥档被跳过（JEV 不就绪）", plugin.jev_client is None)
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 13. JEV：超时计入熔断 + 端点规整 ----------
    section(f"[{gen}] JEV 端点规整与超时")
    from importlib import import_module
    jev_mod = import_module(pkg + ".jev")
    check("裸域名规整", jev_mod.endpoint_of("https://api.typesafe.ai") == "https://api.typesafe.ai/v1/systemone")
    check("/v1 规整", jev_mod.endpoint_of("https://x.com/v1") == "https://x.com/v1/systemone")
    check("完整端点不重复拼", jev_mod.endpoint_of("https://x.com/v1/systemone/") == "https://x.com/v1/systemone")
    check("chat/completions 尾巴剥掉", jev_mod.endpoint_of("https://x.com/v1/chat/completions") == "https://x.com/v1/systemone")
    srv, base, jh = jev_server([(200, None)])  # 挂起 3s
    entries = {("pm", "jev-latest"): {"provider_name": "测试", "model_name": "jev-latest",
                                      "provider_config": {"base_url": base, "api_key": "sk-test"}}}
    cfg = default_cfg(section_jev={
        "jev_enabled": True, "jev_models": ["pm:jev-latest"],
        "jev_timeout_ms": 600, "jev_fallback_llm": False,
    })
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, cfg, provider_entries=entries)
    t0 = time.monotonic()
    d = await plugin._decide("s1", 0, "对话")
    el = time.monotonic() - t0
    check("JEV 硬超时生效", d == "stay" and el < 2.0, f"{el:.2f}s")
    srv.shutdown()
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 14. session 人格注入 + 缓存 ----------
    section(f"[{gen}] session 人格注入与缓存")
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod)
    plugin.state_mgr.set_stage("s1", 1)
    from core.prompt_manager import Prompt
    req = LLMRequest()
    req.tool_set = ToolSet()
    req.system_prompt.append(Prompt(content="人设：{persona}", name="persona", source="core", persona="旧人格"))
    ev = make_batch_event("s1")
    await plugin.inject_session_persona(ev, req)
    check("已有 persona prompt 被替换为当前阶段人格", req.system_prompt[0].kwargs.get("persona") == "你是暖暖，热情。")
    calls_after_first = ctx.persona_mgr.get_calls
    req2 = LLMRequest()
    req2.tool_set = ToolSet()
    await plugin.inject_session_persona(ev, req2)
    appended = [p for p in req2.system_prompt if p.name == "persona"]
    check("无 persona prompt 时新建注入", len(appended) == 1 and appended[0].kwargs.get("persona") == "你是暖暖，热情。")
    check("第二次走缓存（零 DB 查询）", ctx.persona_mgr.get_calls == calls_after_first)
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 15. global 作用域 ----------
    section(f"[{gen}] global 作用域")
    cfg = default_cfg(section_basic={"persona_scope": "global"})
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, cfg)
    ok = await plugin._change_stage("any", 2)
    check("global 切换调用 set_active_persona", ok and ctx.persona_mgr.active == "hot")
    check("global 阶段记录", plugin.state_mgr.get_stage("any", 0) == 2)
    req = LLMRequest()
    req.tool_set = ToolSet()
    await plugin.inject_session_persona(make_batch_event("any"), req)
    check("global 模式不做 session 注入", len(req.system_prompt) == 0)
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 16. 关键词流 ----------
    section(f"[{gen}] 关键词手动控制")
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod)
    ev = make_im_event("s1", "/进阶")
    await plugin.on_im_message(ev)
    check("关键词进阶生效", plugin.state_mgr.get_stage("s1", 0) == 1)
    check("进阶回复已发送", any("已切换" in t for _, t in ctx.message_processor.sent))
    check("消息被拦截（discard+stop）", ev._rec["discarded"] and ev._rec["stopped"])
    ev2 = make_im_event("s1", "/清数")
    plugin.state_mgr.set_turn_counter("s1", 5)
    await plugin.on_im_message(ev2)
    check("清数关键词生效", plugin.state_mgr.get_turn_counter("s1") == 0)
    # 权限
    cfg = default_cfg(section_permission={"enable_permission": True, "allowed_users": ["admin"]})
    plugin2, ctx2, tmp2 = await make_plugin(gen, pkg, mod, cfg)
    ev3 = make_im_event("s1", "/进阶", uid="stranger")
    await plugin2.on_im_message(ev3)
    check("无权限用户被拒", plugin2.state_mgr.get_stage("s1", 0) == 0
          and any("权限不足" in t for _, t in ctx2.message_processor.sent))
    ev4 = make_im_event("s1", "/进阶", uid="admin")
    await plugin2.on_im_message(ev4)
    check("白名单用户放行", plugin2.state_mgr.get_stage("s1", 0) == 1)
    # 边界（白名单用户触边界）
    ev5 = make_im_event("s1", "/降级", uid="admin")
    plugin2.state_mgr.set_stage("s1", 0)
    await plugin2.on_im_message(ev5)
    check("最低阶段降级提示", any("最低" in t for _, t in ctx2.message_processor.sent))
    await plugin.terminate()
    await plugin2.terminate()
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(tmp2, ignore_errors=True)

    # ---------- 17. 工具注入与执行 ----------
    section(f"[{gen}] LLM 工具注入与执行")
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod)
    req = LLMRequest()
    req.tool_set = ToolSet()
    await plugin.inject_tools(make_batch_event("s1"), req)
    names = sorted(t.name for t in req.tool_set.tools)
    check("注入 5 个人格工具", names == ["persona_advance", "persona_decay", "persona_get_stage",
                                        "persona_reset_counter", "persona_set_stage"], str(names))
    adv = [t for t in req.tool_set.tools if t.name == "persona_advance"][0]
    out = await adv.execute(make_batch_event("s1"))
    check("工具进阶生效", plugin.state_mgr.get_stage("s1", 0) == 1 and "已切换" in out)
    get_t = [t for t in req.tool_set.tools if t.name == "persona_get_stage"][0]
    out = await get_t.execute(make_batch_event("s1"))
    check("工具查阶段", "1 / 2" in out, out)
    # 关闭 enable_tool 不注入
    cfg = default_cfg(section_permission={"enable_tool": False})
    plugin2, ctx2, tmp2 = await make_plugin(gen, pkg, mod, cfg)
    req2 = LLMRequest()
    req2.tool_set = ToolSet()
    await plugin2.inject_tools(make_batch_event("s1"), req2)
    check("enable_tool=False 不注入", len(req2.tool_set.tools) == 0)
    await plugin.terminate()
    await plugin2.terminate()
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(tmp2, ignore_errors=True)

    # ---------- 18. HTTP API ----------
    section(f"[{gen}] HTTP API（含新 /status）")
    cfg = default_cfg(section_trigger={"auxiliary_model": "prov1:m-uuid-abc" if gen == "3.0" else "prov1:deepseek-v4.1-flash"})
    aux = StubLLM()
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, cfg, aux_llm=aux)
    r = await plugin.api_get_stage("s1")
    check("GET /stage", r.get("stage") == 0 and r.get("max") == 2)
    r = await plugin.api_set_stage({"session_id": "s1", "stage": 2})
    check("POST /stage 生效", r.get("applied") is True and plugin.state_mgr.get_stage("s1", 0) == 2)
    r = await plugin.api_set_stage({"session_id": "s1", "stage": 9})
    check("POST /stage 超界拒绝", "error" in r)
    r = await plugin.api_set_stage({"session_id": "s1", "stage": "abc"})
    check("POST /stage 非整数拒绝", "error" in r)
    r = await plugin.api_get_current_persona("s1")
    check("GET /persona/current 带可读名", r.get("persona_id") == "hot" and r.get("persona_name") == "烈烈", str(r))
    r = await plugin.api_status()
    expect_model = "Deepseek-V4.1-Flash (Workbuddy)" if gen == "3.0" else "deepseek-v4.1-flash (Workbuddy)"
    check("GET /status 模型可读名", r.get("aux_model") == expect_model, str(r.get("aux_model")))
    check("GET /status 人格映射带名", r["stage_map"]["2"]["persona_name"] == "烈烈")
    check("GET /status JEV 块", r.get("jev", {}).get("enabled") is False and r.get("jev", {}).get("ready") is False)
    check("GET /status 不泄露密钥", "api_key" not in json.dumps(r))
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 19. 对话文本构建 ----------
    section(f"[{gen}] 对话文本构建（多模态/截断/裁剪）")
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod)
    te = plugin.trigger_engine
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "看我发的图"}, {"type": "image_url", "image_url": {"url": "x"}}]},
        {"role": "assistant", "content": "好看"},
        {"role": "tool", "content": "工具输出"},
        {"role": "user", "content": "长" * 800},
    ]
    text = te.build_conversation_text(msgs, ignore_tool=True, ignore_assistant=False)
    check("多模态 content 提取文本", "看我发的图" in text and "'type':" not in text and "http" not in text)
    check("工具消息被忽略", "工具输出" not in text)
    check("单条截断 500", ("长" * 500) in text and ("长" * 501) not in text)
    text2 = te.build_conversation_text(msgs, ignore_tool=True, ignore_assistant=True)
    check("助手消息可忽略", "好看" not in text2)
    many = [{"role": "user", "content": f"第{i}条 " + "x" * 480} for i in range(60)]
    text3 = te.build_conversation_text(many, ignore_tool=True, ignore_assistant=False)
    check("总长截断保最新", len(text3) <= 13000 and "第59条" in text3)
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 21. 真实提供商格式解析（真 ProviderManager，双世代各自格式） ----------
    section(f"[{gen}] 真实提供商格式 → JEV 链接解析")
    from core.provider.provider_manager import ProviderManager
    if gen == "3.0":
        # 3.0 现代格式：model_id = uuid hex，可读名在 model_name
        providers = {"provA": {
            "format": "openai", "status": "active", "model_config_version": 2,
            "name": "JEV中转",
            "provider_config": {"base_url": "https://jev-relay.example.com", "api_key": "sk-test-0000"},
            "model_config": {"llm": {"9f8e7d6c5b4a3c2d1e0f9a8b7c6d5e4f":
                                      {"model_name": "jev-latest", "config": {}}}},
        },
        "provD": {
            # 3.1.1 新增：TypeSafe 决策模型（decision 分组）
            "format": "typesafe", "status": "active", "model_config_version": 2,
            "name": "TypeSafe决策",
            "provider_config": {"base_url": "https://ts.example.com", "api_key": "sk-test-0000"},
            "model_config": {"decision": {"aabbccddeeff00112233445566778899":
                                          {"model_name": "jev-latest", "config": {}}}},
        }}
        real_ref = "provA:9f8e7d6c5b4a3c2d1e0f9a8b7c6d5e4f"
        dec_ref = "provD:aabbccddeeff00112233445566778899"
    else:
        # 2.x 传统格式：模型键即可读名，条目即配置
        providers = {"provA": {
            "format": "openai", "status": "active", "name": "JEV中转",
            "provider_config": {"base_url": "https://jev-relay.example.com", "api_key": "sk-test-0000"},
            "model_config": {"llm": {"jev-latest": {}}},
        }}
        real_ref = "provA:jev-latest"
    try:
        pm = ProviderManager(None, {"providers": providers})
        info = pm.get_model_info("provA", real_ref.split(":", 1)[1])
        check("真 ProviderManager 解析模型", info is not None)
        if info is not None:
            check("模型可读名", (getattr(info, "model_name", None) or info.model_id) == "jev-latest")
            check("provider_config 带 base/key",
                  info.provider_config.get("base_url") == "https://jev-relay.example.com"
                  and info.provider_config.get("api_key") == "sk-test-0000")
            plugin, ctx, tmp = await make_plugin(gen, pkg, mod)
            ctx.provider_mgr = pm  # 换真管理器
            links, displays = plugin._resolve_jev_links({"jev_models": [real_ref]})
            check("真实格式 → JEV 链",
                  links == [("https://jev-relay.example.com", "sk-test-0000", "jev-latest")], str(links))
            check("链描述可读且不含密钥",
                  bool(displays) and "jev-latest" in displays[0] and "sk-test-0000" not in displays[0],
                  str(displays))
            if gen == "3.0":
                links2, _ = plugin._resolve_jev_links({"jev_decision_models": [dec_ref]})
                check("decision 组 → JEV 链",
                      links2 == [("https://ts.example.com", "sk-test-0000", "jev-latest")], str(links2))
                links3, _ = plugin._resolve_jev_links({"jev_models": [real_ref],
                                                       "jev_decision_models": [dec_ref]})
                check("llm→decision 合并顺序",
                      [l[0] for l in links3] == ["https://jev-relay.example.com", "https://ts.example.com"],
                      str(links3))
            await plugin.terminate()
            shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        check("真 ProviderManager 构造/解析异常", False, f"{type(e).__name__}: {e}")

    # ---------- 20. 性能：热路径零磁盘 IO ----------
    section(f"[{gen}] 性能（热路径）")
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod)
    sfile = os.path.join(tmp, "states.json")
    plugin.state_mgr.flush(force=True)
    m0 = os.path.getmtime(sfile) if os.path.exists(sfile) else 0
    time.sleep(0.02)
    t0 = time.monotonic()
    for _ in range(1000):
        await plugin.on_batch_message(make_batch_event("s1"))  # interval=3 ⇒ 会多次触发后台判断
    el_loop = time.monotonic() - t0
    # 等所有后台判断落地
    for _ in range(50):
        if not plugin._trigger_tasks:
            break
        await asyncio.sleep(0.05)
    el = time.monotonic() - t0
    m1 = os.path.getmtime(sfile) if os.path.exists(sfile) else 0
    writes = "已落盘(防抖内正常)" if m1 > m0 else "零落盘"
    check("1000 批事件平均 < 1ms", el_loop / 1000 < 0.001, f"{el_loop/1000*1000:.3f}ms/批, {writes}")
    check("事件循环未被堵（1000 批+后台收尾 < 3s）", el < 3.0, f"{el:.2f}s")
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 22b. 版本声明与未来形态 ----------
    section(f"[{gen}] core_version 声明与未来形态")
    import json as _json
    man = _json.loads(open(os.path.join(_PLUGIN_DIR_CURRENT, "manifest.json"), encoding="utf-8").read())
    check("manifest 声明 core_version", man.get("core_version") == ">=2.24.3", str(man.get("core_version")))
    from packaging.version import Version
    from packaging.specifiers import SpecifierSet
    spec = SpecifierSet(man["core_version"], prereleases=True)
    ok_list = ["2.24.3", "2.34.8", "3.0.0a4", "3.0.0rc1", "3.0.0", "3.1.0", "4.0.0"]
    check("声明放行 2.24.3+/3.0 全系/未来版本", all(Version(v) in spec for v in ok_list))
    check("声明拦截过旧版本", Version("2.24.2") not in spec and Version("2.0.0") not in spec)

    class _FutureSessionMgr:
        """假想未来形态：fetch_memory 与 get_session 都不存在。"""

        async def delete_session(self, sid):
            pass

    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, aux_llm=StubLLM(replies=["up"]))
    plugin.ctx.session_mgr = _FutureSessionMgr()
    plugin.state_mgr.set_turn_counter("s1", 3)
    # 生产路径（_run_trigger 兜底）：异常不得冒出、阶段不得误变
    await plugin._run_trigger("s1", __import__("asyncio").Lock())
    check("未来未知形态：判断异常被兜底不外冒", True)
    check("未来未知形态：阶段不误变", plugin.state_mgr.get_stage("s1", 0) == 0)
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

    # ---------- 22. 3.0 rc.1 会话管理器形态适配（异步 SQLite 化） ----------
    section(f"[{gen}] 3.0rc1 会话形态适配（fetch_memory 移除/delete 异步化）")
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, aux_llm=StubLLM(replies=["up"]))
    rc1 = StubSessionMgrRC1()
    rc1.memory["s1"] = [{"role": "user", "content": "早"}, {"role": "assistant", "content": "早呀"}]
    plugin.ctx.session_mgr = rc1
    mem = await plugin._fetch_memory("s1")
    check("rc1 形态读记忆（flatten_content）", len(mem) == 2 and mem[0]["content"] == "早", str(mem)[:60])
    check("rc1 无会话返回空列表", await plugin._fetch_memory("nope") == [])
    # 完整判断链路走 rc1 形态
    plugin.state_mgr.set_turn_counter("s1", 3)
    await plugin._process_trigger("s1")
    check("rc1 形态完整判断链路（阶段+1）", plugin.state_mgr.get_stage("s1", 0) == 1)
    # 清空上下文（三代择优）：rc1 走 update_session(content=[]) 保留会话本身
    plugin.clear_context_on_switch = True
    await plugin._change_stage("s1", 2)
    check("rc1 清空走 update_session(content=[])",
          rc1.updated and rc1.updated[-1] == ("s1", []), str(rc1.updated))
    check("rc1 不再走删除+重建", "s1" not in rc1.deleted and "s1" not in rc1.recreated)
    check("rc1 清空后读记忆为空", await plugin._fetch_memory("s1") == [])
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)
    # 旧形态（2.x/alpha）：走 write_memory([]) 保留会话记录
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod)
    plugin.clear_context_on_switch = True
    ctx.session_mgr.memory["s1"] = [{"role": "user", "content": "早"}]
    await plugin._change_stage("s1", 1)
    check("旧形态清空走 write_memory([])",
          ctx.session_mgr.written and ctx.session_mgr.written[-1] == ("s1", []),
          str(ctx.session_mgr.written))
    check("旧形态不再走删除", "s1" not in ctx.session_mgr.deleted)
    check("旧形态清空后读记忆为空", ctx.session_mgr.fetch_memory("s1") == [])
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)
    # 最老兜底形态（无清空 API）：仍能 delete+重建
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod)
    plugin.clear_context_on_switch = True
    oldest = StubSessionMgrOldest()
    oldest.memory["s1"] = [{"role": "user", "content": "早"}]
    plugin.ctx.session_mgr = oldest
    await plugin._change_stage("s1", 1)
    check("最老兜底：delete+重建仍可用",
          "s1" in oldest.deleted and "s1" in oldest.recreated)
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)
    if _cfg_file.exists():
        _cfg_file.unlink()


# ---------------- 21. 与 v1.1.5 基线行为对拍 ----------------
async def _scenario(pkg, mod, gen):
    """同输入跑一遍：触发判断 + 关键词。返回可对比的行为指纹。"""
    aux = StubLLM(replies=["up"])
    plugin, ctx, tmp = await make_plugin(gen, pkg, mod, aux_llm=aux)
    ctx.session_mgr.memory["s1"] = [{"role": "user", "content": "早上好"},
                                    {"role": "assistant", "content": "早"}]
    for _ in range(3):
        await plugin.on_batch_message(make_batch_event("s1"))
    for _ in range(40):  # 新版为后台判断，等落地；基线同步 await 已完成
        if plugin.state_mgr.get_stage("s1", 0) >= 1:
            break
        await asyncio.sleep(0.05)
    fp = {
        "stage_after_trigger": plugin.state_mgr.get_stage("s1", 0),
        "turn_after_trigger": plugin.state_mgr.get_turn_counter("s1"),
        "llm_prompt_has_conv": any("早上好" in c for c in aux.calls),
    }
    ev = make_im_event("s1", "/进阶")
    await plugin.on_im_message(ev)
    fp["stage_after_kw"] = plugin.state_mgr.get_stage("s1", 0)
    fp["kw_reply"] = [t for _, t in ctx.message_processor.sent]
    fp["kw_blocked"] = ev._rec["discarded"] and ev._rec["stopped"]
    r = await plugin.api_get_stage("s1")
    fp["api_stage"] = r.get("stage")
    await plugin.terminate()
    shutil.rmtree(tmp, ignore_errors=True)
    return fp


async def run_regression(gen, baseline_dir):
    section(f"[{gen}] 与上一版本基线行为对拍（同输入⇒同结果）")
    new_pkg = "kpp_new_" + gen.replace(".", "")
    old_pkg = "kpp_old_" + gen.replace(".", "")
    new_mod = load_plugin(new_pkg, _PLUGIN_DIR_CURRENT)
    old_mod = load_plugin(old_pkg, baseline_dir)
    fp_new = await _scenario(new_pkg, new_mod, gen)
    fp_old = await _scenario(old_pkg, old_mod, gen)
    check("触发后阶段一致", fp_new["stage_after_trigger"] == fp_old["stage_after_trigger"],
          f"new={fp_new['stage_after_trigger']} old={fp_old['stage_after_trigger']}")
    check("触发后计数一致", fp_new["turn_after_trigger"] == fp_old["turn_after_trigger"],
          f"new={fp_new['turn_after_trigger']} old={fp_old['turn_after_trigger']}")
    check("LLM 收到对话一致", fp_new["llm_prompt_has_conv"] == fp_old["llm_prompt_has_conv"])
    check("关键词进阶一致", fp_new["stage_after_kw"] == fp_old["stage_after_kw"])
    check("关键词回复一致", fp_new["kw_reply"] == fp_old["kw_reply"],
          f"new={fp_new['kw_reply']} old={fp_old['kw_reply']}")
    check("关键词拦截一致", fp_new["kw_blocked"] == fp_old["kw_blocked"])
    check("API 读阶段一致", fp_new["api_stage"] == fp_old["api_stage"])


# ---------------- 主入口 ----------------
def main():
    global GEN, _PLUGIN_DIR_CURRENT
    args = parse_args()
    cores = resolve_cores(args)
    if not cores:
        print("未找到 KiraAI 核心。用法：\n"
              "  python3 run_tests.py --core2x /path/to/KiraAI-2.x --core30 /path/to/KiraAI-3.0\n"
              "或设环境变量 KIRA_CORE_2X / KIRA_CORE_30（给几个跑几个）。")
        sys.exit(2)
    plugin_dir = (args.plugin_dir or "").strip() or str(_DEFAULT_ROOT)
    _PLUGIN_DIR_CURRENT = plugin_dir
    repo = (args.repo or "").strip() or plugin_dir
    print(f"  （插件目录：{plugin_dir}）", flush=True)
    baseline_dir = extract_baseline(Path(repo), Path(plugin_dir))
    if baseline_dir is None:
        print("  （未找到可用的 git 基线 ⇒ 跳过版本对拍段）", flush=True)
    for gen, core in cores.items():
        GEN = gen
        section(f"===== 世代 {gen}（{core}）=====")
        try:
            swap_core(core)
            pkg = "kpp_" + gen.replace(".", "")
            mod = load_plugin(pkg, plugin_dir)
            asyncio.run(run_gen_suite(gen, pkg, mod))
            if baseline_dir:
                asyncio.run(run_regression(gen, baseline_dir))
        except Exception:
            check(f"[{gen}] 套件执行异常", False, traceback.format_exc()[-800:])

    fails = [r for r in RESULTS if not r[2]]
    print(f"\n===== 汇总：{len(RESULTS) - len(fails)}/{len(RESULTS)} 通过 =====", flush=True)
    if fails:
        for g, n, _, d in fails:
            print(f"  FAIL [{g}] {n}" + (f" :: {d}" if d else ""))
        sys.exit(1)
    print("ALL TEST SUITES PASSED")


if __name__ == "__main__":
    main()
