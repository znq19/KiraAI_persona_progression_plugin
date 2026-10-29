from __future__ import annotations
import asyncio
import inspect
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from core.plugin import BasePlugin, logger, on, Priority, register
from core.provider import LLMRequest, LLMModelClient
from core.chat.message_utils import KiraMessageBatchEvent, KiraMessageEvent
from core.chat import MessageChain
from core.chat.message_elements import Text
from core.persona import PersonaManager
from core.prompt_manager import Prompt
from core.utils.tool_utils import BaseTool

from .state_manager import StateManager
from .trigger import TriggerEngine
from .jev import (JevConfig, endpoint_of, DEFAULT_BASE_URL, DEFAULT_MODEL,
                  _pick, _env, _BASE_KEYS, _KEY_KEYS, split_ref)

#: 人格内容缓存 TTL（秒）：人格编辑最多这么久生效；阶段切换时主动失效
PERSONA_CACHE_TTL = 60.0
#: 缺失人格告警去重窗口（秒）：同一人格 id 最多这么久 warn 一次（防刷屏）
PERSONA_WARN_INTERVAL = 600.0


class PersonaProgressionPlugin(BasePlugin):
    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self.data_dir: Optional[Path] = None
        self.state_mgr: Optional[StateManager] = None
        self.trigger_engine: Optional[TriggerEngine] = None
        self.persona_mgr: Optional[PersonaManager] = None

        # 基础配置
        self.enabled = True
        self.persona_scope = "session"
        self.total_stages = 5
        self.stage_map: Dict[int, str] = {}
        self.stage_names: Dict[int, str] = {}   # 阶段 → 人格可读名（展示用，映射仍以 stage_map 为准）
        self.initial_stage = 0
        self.enable_debug_log = False
        self.enabled_sessions: List[str] = []

        # 触发配置
        self.trigger_interval = 10
        self.enable_ai = True
        self.aux_model = None
        self.aux_model_uuid = ""
        self.aux_model_display = "默认快速模型"
        self.aux_clients: List[tuple] = []       # [(ref, client)] 有序回退链
        self.jev_link_displays: List[str] = []   # JEV 链可读描述（不含密钥）
        self.judge_prompt = ""
        self.ignore_tool = True
        self.ignore_assistant = False
        self.include_reasoning_content = False

        # JEV 决策引擎（可选，默认关闭 ⇒ 行为与 v1.1.5 完全一致）
        self.jev_config = JevConfig(enabled=False)
        self.jev_client = None

        # 衰退配置
        self.enable_decay = False
        self.decay_threshold = 3

        # 手动控制配置
        self.enable_keyword = True
        self.advance_keywords: List[str] = []
        self.decay_keywords: List[str] = []
        self.clear_keywords: List[str] = []
        self.enable_tool = True
        self.reset_on_manual = False
        self.clear_context_on_switch = False

        # 权限与反馈配置
        self.enable_permission = False
        self.allowed_users: List[str] = []
        self.permission_denied_message = "❌ 权限不足：您没有权限切换人格阶段"
        self.success_message = "✅ 已切换人格阶段：{action} 至阶段 {stage}"
        self.error_message = "❌ 切换失败：{error}"
        self.max_stage_message = "⚠️ 已达最大阶段（{max_stage}），无法继续进阶"
        self.min_stage_message = "⚠️ 已达最低阶段（{min_stage}），无法继续降级"

        # 运行时：后台判断（不堵消息管线）
        self._trigger_locks: Dict[str, asyncio.Lock] = {}
        self._trigger_tasks: set = set()
        # 运行时：人格内容缓存 / 缺失告警去重
        self._persona_cache: Dict[str, tuple] = {}   # persona_id -> (PersonaInfo, ts)
        self._persona_warned: Dict[str, float] = {}  # persona_id -> last warn ts

    async def initialize(self):
        self.data_dir = self.ctx.get_plugin_data_dir()
        if self.data_dir is None:
            logger.error("无法获取插件数据目录")
            return

        # 一次性配置迁移：旧单选 auxiliary_model → auxiliary_models 第 1 档
        # （就地合并自己负责的键，不经框架 update_plugin_config —— 它会重启插件，
        #   初始化期调用会递归；幂等：旧键清空后不再触发）
        self._migrate_aux_chain()

        cfg = self.plugin_cfg

        # 基础设置
        basic = cfg.get("section_basic", {})
        self.enabled = basic.get("enabled", True)
        self.persona_scope = basic.get("persona_scope", "session")
        self.total_stages = basic.get("total_stages", 5)
        self.initial_stage = basic.get("initial_stage", 0)
        self.enable_debug_log = basic.get("enable_debug_log", False)
        self.enabled_sessions = [s.strip() for s in basic.get("enabled_sessions", []) if s.strip()]

        if self.initial_stage < 0 or self.initial_stage >= max(1, self.total_stages):
            logger.warning("初始阶段 %d 超出范围（0 ~ %d），将按 0 处理",
                           self.initial_stage, self.total_stages - 1)
            self.initial_stage = 0

        self.persona_mgr = self.ctx.persona_mgr
        all_personas = await self.persona_mgr.list_personas()
        persona_lookup: Dict[str, str] = {}
        persona_names: Dict[str, str] = {}
        for p in all_personas:
            persona_lookup[p.id] = p.id
            if p.name:
                persona_lookup[p.name] = p.id
                persona_names[p.id] = p.name
        if self.enable_debug_log:
            logger.debug(f"人格查找表: {persona_lookup}")

        raw_map = basic.get("stage_persona_map", [])
        self.stage_map = {}
        self.stage_names = {}
        if isinstance(raw_map, list):
            for entry in raw_map:
                if not entry or not entry.strip():
                    continue
                parts = entry.strip().split(";", 1)
                if len(parts) != 2:
                    logger.warning(f"无效的映射条目: {entry}，格式应为 '阶段;人格标识'")
                    continue
                stage_str, identifier = parts[0].strip(), parts[1].strip()
                try:
                    stage = int(stage_str)
                except ValueError:
                    logger.warning(f"阶段编号不是有效整数: {stage_str}")
                    continue
                if not identifier:
                    logger.warning(f"人格标识为空: {entry}")
                    continue
                if stage < 0 or stage >= self.total_stages:
                    logger.warning(f"阶段 {stage} 超出范围（0 ~ {self.total_stages - 1}），已忽略: {entry}")
                    continue
                real_id = persona_lookup.get(identifier)
                if not real_id:
                    logger.warning(f"找不到人格标识 '{identifier}'，请检查名称或ID是否正确")
                    continue
                if stage in self.stage_map:
                    logger.warning(f"阶段 {stage} 被重复定义，新值 {real_id} 将覆盖旧值 {self.stage_map[stage]}")
                self.stage_map[stage] = real_id
                self.stage_names[stage] = persona_names.get(real_id, real_id)
                if self.enable_debug_log:
                    logger.debug(f"映射阶段 {stage} -> 人格ID {real_id} (来自标识 '{identifier}')")
        else:
            logger.warning("stage_persona_map 配置不是列表，请检查配置格式")
            self.stage_map = {}

        if self.stage_map:
            missing = [str(i) for i in range(self.total_stages) if i not in self.stage_map]
            if missing:
                logger.warning(f"以下阶段未配置人格映射: {', '.join(missing)}")
            mapping_desc = ", ".join(
                f"{s}→「{self.stage_names.get(s, self.stage_map[s])}」" for s in sorted(self.stage_map)
            )
            logger.info("阶段人格映射: %s", mapping_desc)

        # 触发设置
        trigger = cfg.get("section_trigger", {})
        self.trigger_interval = trigger.get("trigger_interval", 10)
        self.enable_ai = trigger.get("enable_ai_judge", True)
        self.ignore_tool = trigger.get("ignore_tool_messages", True)
        self.ignore_assistant = trigger.get("ignore_assistant_messages", False)
        self.include_reasoning_content = trigger.get("include_reasoning_content", False)
        # 辅助模型组（有序回退；迁移后旧单选已在组内第 1 档）
        aux_refs = [str(m).strip() for m in (trigger.get("auxiliary_models") or [])
                    if str(m).strip()][:10]
        self.aux_clients = []
        for ref in aux_refs:
            client = self.ctx.get_llm_client(model_uuid=ref)
            if client:
                self.aux_clients.append((ref, client))
            else:
                logger.warning("辅助模型组：%s 不可达（提供商/模型已删除？），已跳过",
                               self._model_display(ref))
        if not aux_refs:
            default_client = self.ctx.get_default_fast_llm_client()
            if default_client:
                self.aux_clients.append(("", default_client))
        self.aux_model = self.aux_clients[0][1] if self.aux_clients else None
        self.aux_model_uuid = self.aux_clients[0][0] if self.aux_clients else ""
        self.aux_model_display = self._model_display(self.aux_model_uuid) if self.aux_model_uuid else "默认快速模型"
        if not self.aux_model:
            logger.warning("辅助 LLM 不可用，AI 判断将失效")
            self.enable_ai = False
        elif len(self.aux_clients) == 1:
            logger.info("辅助判断模型: %s", self.aux_model_display)
        else:
            logger.info("辅助模型组（%d 档有序回退）: %s", len(self.aux_clients),
                        " → ".join(self._model_display(ref) if ref else "默认快速模型"
                                   for ref, _ in self.aux_clients))

        # AI 判断提示词
        ai_section = cfg.get("section_ai", {})
        self.judge_prompt = ai_section.get("judge_prompt", "")

        # JEV 决策引擎（可选）
        jev_sec = cfg.get("section_jev", {}) or {}
        try:
            timeout_ms = float(jev_sec.get("jev_timeout_ms", 4000) or 4000)
        except (TypeError, ValueError):
            timeout_ms = 4000
        try:
            min_conf = float(jev_sec.get("jev_min_confidence", 0) or 0) / 100.0
        except (TypeError, ValueError):
            min_conf = 0.0
        jev_links, self.jev_link_displays = self._resolve_jev_links(jev_sec)
        self.jev_config = JevConfig(
            enabled=jev_sec.get("jev_enabled", False),
            links=jev_links,
            timeout=timeout_ms / 1000.0,
            min_confidence=min_conf,
            fallback_llm=jev_sec.get("jev_fallback_llm", True),
        )
        self.jev_client = self.jev_config.client(logger) if self.jev_config.ready else None
        if self.jev_config.enabled:
            if self.jev_client:
                logger.info("JEV 决策引擎已启用（%d 档有序回退，超时 %.1fs，置信门槛 %.2f，失败回退辅助LLM: %s）: %s",
                            len(self.jev_config.links), self.jev_config.timeout,
                            self.jev_config.min_confidence,
                            "开" if self.jev_config.fallback_llm else "关",
                            " → ".join(self.jev_link_displays))
            else:
                logger.warning("JEV 已启用但没有可用连接——请在「JEV 模型组」挑选已注册的模型，"
                               "或手动填写接口地址与密钥")

        # 衰退机制
        decay = cfg.get("section_decay", {})
        self.enable_decay = decay.get("enable_decay", False)
        self.decay_threshold = decay.get("decay_threshold", 3)

        # 手动控制
        manual = cfg.get("section_manual", {})
        self.enable_keyword = manual.get("enable_keyword_trigger", True)
        self.advance_keywords = [kw.strip().lower() for kw in manual.get("advance_keywords", ["/进阶", "/升级"]) if kw.strip()]
        self.decay_keywords = [kw.strip().lower() for kw in manual.get("decay_keywords", ["/降级", "/退阶"]) if kw.strip()]
        self.clear_keywords = [kw.strip().lower() for kw in manual.get("clear_keywords", ["/消级", "/清数"]) if kw.strip()]
        self.reset_on_manual = manual.get("reset_on_manual", False)
        self.clear_context_on_switch = manual.get("clear_context_on_switch", False)

        # 权限与反馈
        perm = cfg.get("section_permission", {})
        self.enable_tool = perm.get("enable_tool", True)
        self.enable_permission = perm.get("enable_permission", False)
        self.allowed_users = [str(uid).strip() for uid in perm.get("allowed_users", []) if str(uid).strip()]
        self.permission_denied_message = perm.get("permission_denied_message", "❌ 权限不足：您没有权限切换人格阶段")
        self.success_message = perm.get("success_message", "✅ 已切换人格阶段：{action} 至阶段 {stage}")
        self.error_message = perm.get("error_message", "❌ 切换失败：{error}")
        self.max_stage_message = perm.get("max_stage_message", "⚠️ 已达最大阶段（{max_stage}），无法继续进阶")
        self.min_stage_message = perm.get("min_stage_message", "⚠️ 已达最低阶段（{min_stage}），无法继续降级")

        # 初始化状态管理器
        self.state_mgr = StateManager(self.data_dir, self.persona_scope)
        self.trigger_engine = TriggerEngine(
            self.persona_mgr,
            self.aux_model if self.enable_ai else None,
            llm_clients=[c for _, c in self.aux_clients] if self.enable_ai else None,
        )

        if self.persona_scope == "global":
            g_state = self.state_mgr.get_state("global")
            if not g_state:
                self.state_mgr.set_stage("global", self.initial_stage)
                self.state_mgr.reset_turn_counter("global")

        logger.info("人设进阶系统初始化完成，总阶段数: %d，作用域: %s，已配置 %d 个阶段映射，调试日志: %s，权限控制: %s，清除上下文: %s",
                    self.total_stages, self.persona_scope, len(self.stage_map),
                    "启用" if self.enable_debug_log else "禁用",
                    "启用" if self.enable_permission else "禁用",
                    "启用" if self.clear_context_on_switch else "禁用")

    async def terminate(self):
        if self.state_mgr:
            self.state_mgr.flush(force=True)
        logger.info("人设进阶系统终止")

    # ---------- 配置迁移 ----------
    def _migrate_aux_chain(self):
        """一次性迁移：旧单选 auxiliary_model → auxiliary_models 第 1 档。

        · 幂等：旧键清空后不再触发；
        · 只动 section_trigger 里自己负责的两个键（读当前段→改→写回整段）；
        · 不经框架 update_plugin_config（它会重启插件 ⇒ 初始化期调用会递归），
          改为「同步管理器内存缓存 + 直接落盘」（路径与框架一致：
          data/config/plugins/<plugin_id>.json）；
        · 写盘失败不影响本次运行（读侧已合并），下轮启动自动重试。
        """
        try:
            trig = self.plugin_cfg.get("section_trigger") or {}
            if not isinstance(trig, dict):
                return
            legacy = str(trig.get("auxiliary_model") or "").strip()
            if not legacy:
                return
            chain = [str(m).strip() for m in (trig.get("auxiliary_models") or [])
                     if str(m).strip()]
            if not chain:
                chain = [legacy]
            trig_new = dict(trig)
            trig_new["auxiliary_models"] = chain
            trig_new.pop("auxiliary_model", None)   # 移除旧键（防配置膨胀）
            # 内存配置即时生效（本次启动）
            self.plugin_cfg["section_trigger"] = trig_new
            pid = "kira_persona_progression"
            # 同步管理器缓存（保持 webui 读写一致）
            mgr = getattr(self.ctx, "plugin_mgr", None)
            if mgr is not None and isinstance(getattr(mgr, "plugin_configs", None), dict):
                full = dict(mgr.plugin_configs.get(pid) or self.plugin_cfg)
                full["section_trigger"] = trig_new
                mgr.plugin_configs[pid] = full
            # 落盘（与框架同路径同格式）
            from core.utils.path_utils import get_config_path
            cfg_dir = get_config_path() / "plugins"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            cfg_file = cfg_dir / f"{pid}.json"
            disk = {}
            if cfg_file.exists():
                try:
                    disk = json.loads(cfg_file.read_text(encoding="utf-8"))
                except Exception:
                    disk = {}
            if not isinstance(disk, dict):
                disk = {}
            disk["section_trigger"] = trig_new
            cfg_file.write_text(json.dumps(disk, ensure_ascii=False, indent=4), encoding="utf-8")
            logger.info("配置迁移：辅助模型旧单选已并入模型组（第 1 档：%s），旧键已移除",
                        self._model_display(legacy))
        except Exception as e:
            logger.warning("辅助模型配置迁移写盘失败（不影响本次运行，下次启动重试）: %s", e)

    # ---------- JEV 连接链解析 ----------
    def _resolve_jev_links(self, jev_sec: dict):
        """jev_models / jev_decision_models → [(base, key, model)] 有序回退链。

        返回 (links, displays)。displays 只含可读名与端点主机，**绝不含密钥**。

        3.1.1 起框架有专门的 TypeSafe 决策模型提供商（model_config.decision 分组），
        而配置页一个模型字段只能绑定一个 model_type ⇒ 用两个多选分别收集，
        按「llm 组 → 决策组」合并成同一条有序回退链（都为空 = 旧行为）。
        """
        links = []
        displays = []
        refs = jev_sec.get("jev_models") or []
        if not isinstance(refs, list):
            refs = []
        drefs = jev_sec.get("jev_decision_models") or []
        if not isinstance(drefs, list):
            drefs = []
        # 两组各自 ≤10 档（合计 ≤20）：不能因为填满一组就把另一组静默丢掉
        refs = refs[:10] + drefs[:10]
        for ref in refs:
            ref = str(ref).strip()
            if not ref:
                continue
            pid, mid = split_ref(ref)
            if not pid or not mid:
                continue
            try:
                info = self.ctx.provider_mgr.get_model_info(pid, mid)
            except Exception:
                info = None
            if not info:
                logger.warning("JEV 模型组：%s 无法解析（提供商/模型已删除？），跳过", ref)
                continue
            conf = getattr(info, "provider_config", None) or {}
            base = _pick(conf, _BASE_KEYS)
            key = _env(_pick(conf, _KEY_KEYS))
            # 上游模型名：3.0 用 model_name；2.x 用 model_id 键
            model = getattr(info, "model_name", None) or getattr(info, "model_id", "") or mid
            if not base or not key:
                logger.warning("JEV 模型组：%s 缺少接口地址或密钥，跳过",
                               self._model_display(ref))
                continue
            if (base, key, model) in links:
                continue
            links.append((base, key, model))
            host = base.split("://", 1)[-1].split("/", 1)[0]
            displays.append(f"{self._model_display(ref)} ⇒ {model}@{host}")
        # 旧版手动连接字段已移除（3.1.1 起改用提供商注册）：旧配置一律**忽略**，绝不隐形生效
        if str(jev_sec.get("jev_api_key") or "").strip():
            logger.warning(
                "检测到旧版手动 JEV 配置（jev_base_url / jev_api_key / jev_model_name）："
                "该方式已移除，配置已忽略、不再生效；请在「设置 → 提供商」注册 TypeSafe "
                "或 OpenAI 兼容提供商，再在 JEV 模型组里选择。"
            )
        return links, displays

    # ---------- 展示辅助 ----------
    def _model_display(self, model_uuid: str) -> str:
        """`provider:model` → '可读名 (提供商名)'；解析失败返回原值（绝不空白）。

        3.0 的 model_id 是 uuid 串、可读名在 model_name（含 legacy_ids 解析）；
        2.x 的可读名就是 model_id 键。
        """
        if not model_uuid:
            return "默认快速模型"
        try:
            pid, _, mid = model_uuid.partition(":")
            if not pid or not mid:
                return model_uuid
            info = self.ctx.provider_mgr.get_model_info(pid, mid)
            if info:
                name = getattr(info, "model_name", None) or getattr(info, "model_id", "") or mid
                provider = getattr(info, "provider_name", "") or pid
                return f"{name} ({provider})"
        except Exception:
            pass
        return model_uuid

    def _persona_display(self, persona_id: str) -> str:
        """人格 id → '名称 (id)' 或原 id。"""
        if not persona_id:
            return "(未配置)"
        try:
            cached = self._persona_cache.get(persona_id)
            if cached and getattr(cached[0], "name", None):
                return f"{cached[0].name} ({persona_id})"
        except Exception:
            pass
        for stage, pid in self.stage_map.items():
            if pid == persona_id:
                name = self.stage_names.get(stage)
                if name and name != persona_id:
                    return f"{name} ({persona_id})"
                break
        return persona_id

    def _check_permission(self, event) -> bool:
        """检查触发事件的用户是否在允许列表中"""
        if not self.enable_permission:
            return True
        try:
            if hasattr(event, "message") and hasattr(event.message, "sender"):
                user_id = str(event.message.sender.user_id)
            elif hasattr(event, "messages") and event.messages and hasattr(event.messages[0], "sender"):
                user_id = str(event.messages[0].sender.user_id)
            else:
                return True
            if not self.allowed_users:
                return True
            return user_id in self.allowed_users
        except Exception:
            return True

    def _get_sid(self, event) -> str:
        if hasattr(event, "sid"):
            return event.sid
        if hasattr(event, "session") and hasattr(event.session, "sid"):
            return event.session.sid
        return "default"

    def _is_session_enabled(self, session_id: str) -> bool:
        """检查会话是否在白名单中，白名单为空则全部启用"""
        if not self.enabled_sessions:
            return True
        return session_id in self.enabled_sessions

    # ---------- 核心逻辑 ----------
    async def _change_stage(self, session_id: str, new_stage: int) -> bool:
        if new_stage < 0 or new_stage >= self.total_stages:
            return False
        current = self.state_mgr.get_stage(session_id, self.initial_stage)
        if new_stage == current:
            return True

        persona_id = self.stage_map.get(new_stage)
        if not persona_id:
            logger.warning("阶段 %d 未配置人格 ID", new_stage)
            return False

        if self.persona_scope == "global":
            ok = await self.trigger_engine.apply_stage_global(new_stage, self.stage_map)
            if ok:
                self.state_mgr.set_stage(session_id, new_stage)
                if self.enable_debug_log:
                    logger.debug("全局人格切换至阶段 %d，人格: %s", new_stage, self._persona_display(persona_id))
                if self.clear_context_on_switch:
                    await self._clear_session_context(session_id)
            return ok
        else:
            self.state_mgr.set_stage(session_id, new_stage)
            if self.enable_debug_log:
                logger.debug("会话 %s 阶段 %d → %d (session 人格待注入，人格: %s)",
                             session_id, current, new_stage, self._persona_display(persona_id))
            if self.clear_context_on_switch:
                await self._clear_session_context(session_id)
            return True

    # ---------- 会话管理器双世代适配（2.x/3.0a4 同步 JSON ⇄ 3.0rc1+ 异步 SQLite） ----------
    async def _fetch_memory(self, session_id: str) -> list:
        """读会话记忆（扁平消息列表）。3.0 rc.1 起 fetch_memory 移除，
        改为 get_session(...).flatten_content()（SQLite 持久化，异步）。"""
        sm = self.ctx.session_mgr
        if sm is None:
            return []
        if hasattr(sm, "fetch_memory"):
            return sm.fetch_memory(session_id) or []
        # 3.0 rc.1+：异步 SQLite 形态
        sess = await sm.get_session(session_id, ensure_session=False)
        if sess is None:
            return []
        return sess.flatten_content()

    async def _delete_and_recreate_session(self, session_id: str) -> None:
        """删除并重建会话（清除上下文用）。delete_session 在 3.0 rc.1 起变为异步。"""
        sm = self.ctx.session_mgr
        result = sm.delete_session(session_id)
        if inspect.isawaitable(result):
            await result
        # 重新初始化会话（重建空记忆）
        if hasattr(sm, "get_session_info"):
            sm.get_session_info(session_id)
        else:
            # 3.0 rc.1+：ensure_session=True 不存在则建空会话
            await sm.get_session(session_id, ensure_session=True)

    async def _clear_session_context(self, session_id: str) -> None:
        """清空会话上下文（**只清对话内容，会话本身的标题/能力设置等元数据保留**）。

        三代适配（按能力择优，特征识别而非版本号）：
          · 3.0 rc.1+：`update_session(sid, content=[])`（SQLite 异步，发
            session_memory_written 事件，记录保留）；
          · 2.x / 3.0-alpha：`write_memory(sid, [])`（同步，会话记录保留）；
          · 更老兜底：delete + 重建（v1.1.5 原行为）。
        """
        try:
            sm = self.ctx.session_mgr
            if sm is None:
                logger.warning("SessionManager 不可用，无法清除上下文")
                return
            if hasattr(sm, "update_session"):
                # 3.0 rc.1+：SQLite 形态，清空 content 保留会话元数据/能力
                await sm.update_session(session_id, content=[])
            elif hasattr(sm, "write_memory"):
                # 2.x / 3.0-alpha：清空 memory 块，会话记录保留
                result = sm.write_memory(session_id, [])
                if inspect.isawaitable(result):
                    await result
            else:
                # 最老形态兜底：删除+重建
                await self._delete_and_recreate_session(session_id)
            if self.enable_debug_log:
                logger.debug("已清空会话上下文（会话本身保留）: %s", session_id)
        except Exception as e:
            logger.warning("清除会话上下文失败 (%s): %s", session_id, e)

    # ---------- 后台判断（不堵消息管线） ----------
    def _schedule_trigger(self, session_id: str):
        """达到间隔 → 后台执行判断；同一会话已有判断在飞则跳过（计数保留，不重置）。

        v1.1.5 在消息事件里同步 await 辅助 LLM（一次 1~30s），会拖住该会话的消息管线；
        改为 create_task 后台跑 + 每会话锁防重入。
        """
        lock = self._trigger_locks.setdefault(session_id, asyncio.Lock())
        if lock.locked():
            if self.enable_debug_log:
                logger.debug("会话 %s 上一次判断仍在进行，跳过本次调度", session_id)
            return
        task = asyncio.create_task(self._run_trigger(session_id, lock))
        self._trigger_tasks.add(task)
        task.add_done_callback(self._trigger_tasks.discard)

    async def _run_trigger(self, session_id: str, lock: asyncio.Lock):
        async with lock:
            try:
                await self._process_trigger(session_id)
            except Exception as e:
                logger.warning("后台判断异常 (%s): %s", session_id, e)

    async def _decide(self, session_id: str, current_stage: int, conversation_text: str) -> str:
        """统一判断入口：JEV 优先（便宜快速）→ 失败按配置回退辅助 LLM → stay。

        JEV 未启用时与 v1.1.5 逐字等价（只走辅助 LLM）。
        """
        max_stage = self.total_stages - 1
        # 1) JEV 决策引擎
        if self.jev_client is not None:
            instructions = self.trigger_engine.render_jev_instructions(
                self.judge_prompt, current_stage, max_stage)
            result = await self.jev_client.decide(conversation_text, instructions)
            if result is not None:
                if self.enable_debug_log:
                    logger.debug("JEV 判断: %s (置信 %.2f, 概率 %s, 模型 %s)",
                                 result["choice"], result["confidence"],
                                 result.get("probabilities"), result.get("model"))
                if result["confidence"] < self.jev_config.min_confidence:
                    if self.enable_debug_log:
                        logger.debug("JEV 置信 %.2f 低于门槛 %.2f，保守取 stay",
                                     result["confidence"], self.jev_config.min_confidence)
                    return "stay"
                return result["choice"]
            # JEV 失败/熔断：按配置回退
            if not self.jev_config.fallback_llm:
                return "stay"
        # 2) 辅助 LLM（JEV 未启用时的原路径）
        if self.enable_ai and self.aux_model:
            decision = await self.trigger_engine.judge(
                self.judge_prompt, current_stage, max_stage, conversation_text)
            if self.enable_debug_log:
                logger.debug("AI 判断结果: %s", decision)
            return decision
        return "stay"

    async def _process_trigger(self, session_id: str):
        if not self.trigger_engine:
            return
        current_stage = self.state_mgr.get_stage(session_id, self.initial_stage)
        if self.enable_debug_log:
            logger.debug("触发检测: session=%s, 当前阶段=%d", session_id, current_stage)

        history = await self._fetch_memory(session_id)
        max_history = 50
        if len(history) > max_history:
            history = history[-max_history:]
            if self.enable_debug_log:
                logger.debug("历史消息截断至最近 %d 条", max_history)

        conversation_text = self.trigger_engine.build_conversation_text(
            history,
            ignore_tool=self.ignore_tool,
            ignore_assistant=self.ignore_assistant,
            include_reasoning=self.include_reasoning_content
        )
        if self.enable_debug_log:
            logger.debug("发送给判断引擎的对话文本长度: %d 字符", len(conversation_text))

        decision = await self._decide(session_id, current_stage, conversation_text)

        new_stage = current_stage
        if decision == "up" and current_stage < self.total_stages - 1:
            new_stage = current_stage + 1
            self.state_mgr.set_consecutive_stay(session_id, 0)
            if self.enable_debug_log:
                logger.debug("判断决定升级，新阶段 %d", new_stage)
        elif decision == "down" and current_stage > 0:
            new_stage = current_stage - 1
            self.state_mgr.set_consecutive_stay(session_id, 0)
            if self.enable_debug_log:
                logger.debug("判断决定降级，新阶段 %d", new_stage)
        else:
            if self.enable_decay:
                stay_count = self.state_mgr.get_consecutive_stay(session_id) + 1
                self.state_mgr.set_consecutive_stay(session_id, stay_count)
                if self.enable_debug_log:
                    logger.debug("连续保持计数: %d", stay_count)
                if stay_count >= self.decay_threshold and current_stage > 0:
                    new_stage = current_stage - 1
                    self.state_mgr.set_consecutive_stay(session_id, 0)
                    logger.info("连续保持 %d 次，触发衰退降级至阶段 %d", stay_count, new_stage)
            else:
                if self.enable_debug_log:
                    logger.debug("判断决定保持，且衰退未启用")

        if new_stage != current_stage:
            await self._change_stage(session_id, new_stage)
        else:
            if self.enable_debug_log:
                logger.debug("阶段未变化")

        self.state_mgr.reset_turn_counter(session_id)
        if self.enable_debug_log:
            logger.debug("轮数计数器已重置")

    # ---------- 事件钩子 ----------
    @on.im_batch_message(priority=Priority.MEDIUM)
    async def on_batch_message(self, event: KiraMessageBatchEvent):
        if not self.enabled:
            return
        session_id = event.sid
        if not self._is_session_enabled(session_id):
            return
        self.state_mgr.increment_turn(session_id)
        self.state_mgr.flush()  # 防抖兜底：最多 30s 落一次盘，热路径零阻塞
        turn = self.state_mgr.get_turn_counter(session_id)
        if self.enable_debug_log:
            logger.debug("会话 %s 当前轮数: %d / %d", session_id, turn, self.trigger_interval)

        if turn >= self.trigger_interval:
            if self.enable_debug_log:
                logger.debug("达到触发间隔，调度后台判断")
            self._schedule_trigger(session_id)

    # ---------- 手动控制：关键词拦截 ----------
    @on.im_message(priority=Priority.HIGH)
    async def on_im_message(self, event: KiraMessageEvent):
        if not self.enabled or not self.enable_keyword:
            return
        text = "".join(elem.text for elem in event.message.chain if isinstance(elem, Text))
        if not text:
            return
        text_lower = text.strip().lower()
        session_id = event.session.sid

        # 白名单过滤：未在白名单中的会话不处理关键词
        if not self._is_session_enabled(session_id):
            return

        # 清空计数（无需权限）
        if text_lower in self.clear_keywords:
            self.state_mgr.reset_turn_counter(session_id)
            if self.enable_debug_log:
                logger.debug("会话 %s 清空计数关键词触发", session_id)
            await self.ctx.send_message_chain(
                session=session_id,
                chain=MessageChain([Text("✅ 已清空轮数计数器")])
            )
            event.discard(force=True)
            event.stop()
            return

        # 处理进阶/降级
        action = None
        target_stage = -1
        current = self.state_mgr.get_stage(session_id, self.initial_stage)

        if text_lower in self.advance_keywords:
            action = "进阶"
            target_stage = current + 1 if current < self.total_stages - 1 else current
        elif text_lower in self.decay_keywords:
            action = "降级"
            target_stage = current - 1 if current > 0 else current

        if action:
            # 权限检查
            if not self._check_permission(event):
                await self.ctx.send_message_chain(
                    session=session_id,
                    chain=MessageChain([Text(self.permission_denied_message)])
                )
                event.discard(force=True)
                event.stop()
                return

            # 执行切换或边界回复
            if target_stage != current:
                ok = await self._change_stage(session_id, target_stage)
                if ok:
                    if self.reset_on_manual:
                        self.state_mgr.reset_turn_counter(session_id)
                    reply = self.success_message.format(action=action, stage=target_stage)
                    if self.enable_debug_log:
                        logger.debug("关键词%s: 阶段 %d -> %d", action, current, target_stage)
                else:
                    reply = self.error_message.format(error="切换失败，请检查人格映射或配置")
                    logger.warning("关键词%s失败: 阶段 %d -> %d", action, current, target_stage)
            else:
                # 边界情况
                if action == "进阶" and current >= self.total_stages - 1:
                    reply = self.max_stage_message.format(
                        max_stage=self.total_stages - 1,
                        current_stage=current
                    )
                elif action == "降级" and current <= 0:
                    reply = self.min_stage_message.format(
                        min_stage=0,
                        current_stage=current
                    )
                else:
                    reply = "⚠️ 阶段未变化"

            await self.ctx.send_message_chain(
                session=session_id,
                chain=MessageChain([Text(reply)])
            )
            event.discard(force=True)
            event.stop()
            return

    # ---------- LLM 工具动态注入 ----------
    @on.llm_request(priority=Priority.MEDIUM)
    async def inject_tools(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        if not self.enabled or not self.enable_tool:
            return
        session_id = event.sid
        if not self._is_session_enabled(session_id):
            return
        req.tool_set.add(_PersonaGetStageTool(self.ctx, self))
        req.tool_set.add(_PersonaSetStageTool(self.ctx, self))
        req.tool_set.add(_PersonaAdvanceTool(self.ctx, self))
        req.tool_set.add(_PersonaDecayTool(self.ctx, self))
        req.tool_set.add(_PersonaResetCounterTool(self.ctx, self))

    # ---------- HTTP API ----------
    @register.api(method="GET", path="/stage", auth=True)
    async def api_get_stage(self, session_id: str) -> dict:
        if not self.state_mgr:
            return {"error": "not initialized"}
        if not self._is_session_enabled(session_id):
            return {"error": "session not enabled"}
        stage = self.state_mgr.get_stage(session_id, self.initial_stage)
        return {"stage": stage, "max": self.total_stages - 1}

    @register.api(method="POST", path="/stage", auth=True)
    async def api_set_stage(self, body: dict) -> dict:
        if not self.state_mgr:
            return {"error": "not initialized"}
        session_id = body.get("session_id")
        stage = body.get("stage")
        if not session_id or stage is None:
            return {"error": "missing session_id or stage"}
        if not self._is_session_enabled(session_id):
            return {"error": "session not enabled"}
        try:
            stage = int(stage)
        except (TypeError, ValueError):
            return {"error": "stage must be integer"}
        if stage < 0 or stage >= self.total_stages:
            return {"error": f"stage must be 0~{self.total_stages-1}"}
        old = self.state_mgr.get_stage(session_id, self.initial_stage)
        ok = await self._change_stage(session_id, stage)
        if ok and self.reset_on_manual:
            self.state_mgr.reset_turn_counter(session_id)
        if self.enable_debug_log:
            logger.debug("API set_stage: %s 阶段 %d -> %d", session_id, old, stage)
        return {"old": old, "new": stage, "applied": ok}

    @register.api(method="GET", path="/persona/current", auth=True)
    async def api_get_current_persona(self, session_id: str) -> dict:
        if not self.state_mgr:
            return {"error": "not initialized"}
        if not self._is_session_enabled(session_id):
            return {"error": "session not enabled"}
        stage = self.state_mgr.get_stage(session_id, self.initial_stage)
        persona_id = self.stage_map.get(stage)
        return {
            "stage": stage,
            "persona_id": persona_id,
            "persona_name": self.stage_names.get(stage, persona_id),
        }

    @register.api(method="GET", path="/status", auth=True)
    async def api_status(self) -> dict:
        """插件状态总览：模型与人格全部给可读名（uuid 仅存不用作显示）。"""
        if not self.state_mgr:
            return {"error": "not initialized"}
        jev_info = {
            "enabled": self.jev_config.enabled,
            "ready": bool(self.jev_client and self.jev_client.ready),
            "links": [
                {"model": m, "endpoint": endpoint_of(b)} for b, _, m in self.jev_config.links
            ],
            "links_display": list(self.jev_link_displays),
            "fallback_llm": self.jev_config.fallback_llm,
            "min_confidence": self.jev_config.min_confidence,
        }
        if self.jev_client:
            jev_info.update({
                "calls": self.jev_client.calls,
                "input_tokens": self.jev_client.tokens,
                "cooldown_left": self.jev_client.cooldown_left,
                "last_error": self.jev_client.last_error,
            })
        return {
            "enabled": self.enabled,
            "scope": self.persona_scope,
            "total_stages": self.total_stages,
            "initial_stage": self.initial_stage,
            "trigger_interval": self.trigger_interval,
            "stage_map": {
                str(s): {"persona_id": pid, "persona_name": self.stage_names.get(s, pid)}
                for s, pid in sorted(self.stage_map.items())
            },
            "aux_model": self.aux_model_display,
            "aux_model_uuid": self.aux_model_uuid or "",
            "aux_chain": [
                {"ref": ref, "name": self._model_display(ref) if ref else "默认快速模型"}
                for ref, _ in self.aux_clients
            ],
            "ai_judge_enabled": self.enable_ai,
            "jev": jev_info,
            "judging_sessions": [sid for sid, lock in self._trigger_locks.items() if lock.locked()],
        }

    # ---------- session 模式人格注入 ----------
    async def _get_persona_cached(self, persona_id: str):
        """人格查询带 60s 缓存（人格编辑最多 60s 生效；阶段切换不经过此缓存键变更）。"""
        now = time.monotonic()
        cached = self._persona_cache.get(persona_id)
        if cached and (now - cached[1]) < PERSONA_CACHE_TTL:
            return cached[0]
        info = await self.persona_mgr.get_persona(persona_id)
        if info:
            self._persona_cache[persona_id] = (info, now)
        else:
            # 缺人格不缓存（可能正在创建），但告警去重防刷屏
            last = self._persona_warned.get(persona_id, 0.0)
            if now - last > PERSONA_WARN_INTERVAL:
                self._persona_warned[persona_id] = now
                logger.warning("人格 ID '%s' 不存在，无法注入", persona_id)
        return info

    @on.llm_request(priority=Priority.SYS_HIGH)
    async def inject_session_persona(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        if not self.enabled or self.persona_scope != "session":
            return
        session_id = event.sid
        if not self._is_session_enabled(session_id):
            return
        stage = self.state_mgr.get_stage(session_id, self.initial_stage)
        persona_id = self.stage_map.get(stage)
        if not persona_id:
            if self.enable_debug_log:
                logger.debug("阶段 %d 未配置人格映射，跳过注入", stage)
            return
        persona_info = await self._get_persona_cached(persona_id)
        if not persona_info:
            return
        persona_text = persona_info.content

        if self.enable_debug_log:
            logger.debug("注入人格: %s (阶段 %d), 内容长度: %d",
                         self._persona_display(persona_id), stage, len(persona_text))

        for p in req.system_prompt:
            if p.name == "persona":
                p.kwargs["persona"] = persona_text
                if self.enable_debug_log:
                    logger.debug("已注入 session 人格到 kwargs: %s (阶段 %d)",
                                 self._persona_display(persona_id), stage)
                break
        else:
            req.system_prompt.append(Prompt(
                content="## 角色扮演（Persona）\n你需要进行角色扮演：\n{persona}\n",
                name="persona",
                source="kira_persona_progression",
                persona=persona_text
            ))
            if self.enable_debug_log:
                logger.debug("已创建新的 persona prompt (阶段 %d)", stage)


class _PersonaGetStageTool(BaseTool):
    name = "persona_get_stage"
    description = "获取当前会话的人格阶段"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, ctx, plugin):
        self.ctx = ctx
        self.plugin = plugin

    async def execute(self, event: KiraMessageBatchEvent, *_) -> str:
        if not self.plugin.state_mgr:
            return "插件未初始化"
        session_id = event.sid
        if not self.plugin._is_session_enabled(session_id):
            return "该会话未启用人格进阶插件"
        stage = self.plugin.state_mgr.get_stage(session_id, self.plugin.initial_stage)
        return f"当前阶段：{stage} / {self.plugin.total_stages - 1}"


class _PersonaSetStageTool(BaseTool):
    name = "persona_set_stage"
    description = "手动设置当前会话的人格阶段"
    parameters = {
        "type": "object",
        "properties": {
            "stage": {"type": "integer", "description": "目标阶段（0 ~ max）"}
        },
        "required": ["stage"]
    }

    def __init__(self, ctx, plugin):
        self.ctx = ctx
        self.plugin = plugin

    async def execute(self, event: KiraMessageBatchEvent, *_, stage: int) -> str:
        if not self.plugin.state_mgr:
            return "插件未初始化"
        if not self.plugin._check_permission(event):
            return self.plugin.permission_denied_message

        session_id = event.sid
        if not self.plugin._is_session_enabled(session_id):
            return "该会话未启用人格进阶插件"
        if stage < 0 or stage >= self.plugin.total_stages:
            return f"阶段必须在 0 ~ {self.plugin.total_stages - 1} 之间"
        old = self.plugin.state_mgr.get_stage(session_id, self.plugin.initial_stage)
        ok = await self.plugin._change_stage(session_id, stage)
        if ok:
            if self.plugin.reset_on_manual:
                self.plugin.state_mgr.reset_turn_counter(session_id)
            if self.plugin.enable_debug_log:
                logger.debug("工具 set_stage: %s 阶段 %d -> %d", session_id, old, stage)
            return self.plugin.success_message.format(action="设置", stage=stage)
        else:
            return self.plugin.error_message.format(error="设置失败，请检查人格映射")


class _PersonaAdvanceTool(BaseTool):
    name = "persona_advance"
    description = "手动将当前会话阶段提升一级"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, ctx, plugin):
        self.ctx = ctx
        self.plugin = plugin

    async def execute(self, event: KiraMessageBatchEvent, *_) -> str:
        if not self.plugin.state_mgr:
            return "插件未初始化"
        if not self.plugin._check_permission(event):
            return self.plugin.permission_denied_message

        session_id = event.sid
        if not self.plugin._is_session_enabled(session_id):
            return "该会话未启用人格进阶插件"
        current = self.plugin.state_mgr.get_stage(session_id, self.plugin.initial_stage)
        if current >= self.plugin.total_stages - 1:
            return self.plugin.max_stage_message.format(
                max_stage=self.plugin.total_stages - 1,
                current_stage=current
            )
        new = current + 1
        ok = await self.plugin._change_stage(session_id, new)
        if ok:
            if self.plugin.reset_on_manual:
                self.plugin.state_mgr.reset_turn_counter(session_id)
            if self.plugin.enable_debug_log:
                logger.debug("工具 advance: %s 阶段 %d -> %d", session_id, current, new)
            return self.plugin.success_message.format(action="进阶", stage=new)
        else:
            return self.plugin.error_message.format(error="升级失败，请检查人格映射")


class _PersonaDecayTool(BaseTool):
    name = "persona_decay"
    description = "手动将当前会话阶段降低一级"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, ctx, plugin):
        self.ctx = ctx
        self.plugin = plugin

    async def execute(self, event: KiraMessageBatchEvent, *_) -> str:
        if not self.plugin.state_mgr:
            return "插件未初始化"
        if not self.plugin._check_permission(event):
            return self.plugin.permission_denied_message

        session_id = event.sid
        if not self.plugin._is_session_enabled(session_id):
            return "该会话未启用人格进阶插件"
        current = self.plugin.state_mgr.get_stage(session_id, self.plugin.initial_stage)
        if current <= 0:
            return self.plugin.min_stage_message.format(
                min_stage=0,
                current_stage=current
            )
        new = current - 1
        ok = await self.plugin._change_stage(session_id, new)
        if ok:
            if self.plugin.reset_on_manual:
                self.plugin.state_mgr.reset_turn_counter(session_id)
            if self.plugin.enable_debug_log:
                logger.debug("工具 decay: %s 阶段 %d -> %d", session_id, current, new)
            return self.plugin.success_message.format(action="降级", stage=new)
        else:
            return self.plugin.error_message.format(error="降级失败，请检查人格映射")


class _PersonaResetCounterTool(BaseTool):
    name = "persona_reset_counter"
    description = "重置当前会话的轮数计数器（不影响阶段）"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, ctx, plugin):
        self.ctx = ctx
        self.plugin = plugin

    async def execute(self, event: KiraMessageBatchEvent, *_) -> str:
        if not self.plugin.state_mgr:
            return "插件未初始化"
        if not self.plugin._check_permission(event):
            return self.plugin.permission_denied_message

        session_id = event.sid
        if not self.plugin._is_session_enabled(session_id):
            return "该会话未启用人格进阶插件"
        self.plugin.state_mgr.reset_turn_counter(session_id)
        if self.plugin.enable_debug_log:
            logger.debug("工具 reset_counter: %s 计数器已重置", session_id)
        return "轮数计数器已重置"
