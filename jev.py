"""JEV（TypeSafe System-One）决策引擎 —— 可选的低成本判断后端。

与生成式辅助 LLM 的区别：JEV 不生成文本，输入 state + 类型化问题，
直接返回「选项 + 概率 + 置信度」。人设进阶的 up/down/stay 判断天然是
一道 choice 题：

  state     = 裁剪后的对话文本（只计一次费）
  questions = {"stage_decision": {"type": "choice", "instructions": <用户提示词>,
                                  "criteria": {"up": ..., "down": ..., "stay": ...}}}

协议实测（2026-10-12，经中转站）：
  POST {base}/v1/systemone  Authorization: Bearer <key>
  → {"model": "jev-1.13.0",
     "answers": {"stage_decision": {"type": "choice", "choice": "stay",
                 "confidence": 0.27, "probabilities": {"up": 0.49, "down": 0.0, "stay": 0.51}}},
     "usage": {"input_tokens": 501, "output_tokens": 41}}

设计要点（与 Z 插件 mdecide.py 同构、按本插件体量裁剪）：
  · 零依赖：urllib 同步实现跑在 asyncio.to_thread（可硬超时）；
  · 熔断：连续失败 3 次 → 冷却 300s（每次失败重新计时）；冷却中直接返回 None；
  · 端点规整：容忍用户粘贴 .../v1、.../v1/systemone、.../v1/chat/completions 等；
  · 任何失败都返回 None（由上层回退到辅助 LLM / stay），绝不外泄异常。
"""
from __future__ import annotations

import asyncio
import json
import time
import urllib.request
from typing import Optional

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"

# 常见"末端路径"——用户可能直接粘贴完整端点，规整时先剥掉
_ENDPOINT_SUFFIXES = ("/v1/systemone", "/systemone", "/chat/completions",
                      "/completions", "/embeddings", "/models", "/v1/messages")


def endpoint_of(base_url: str) -> str:
    """把 base_url 规整成 systemone 端点。

      https://api.typesafe.ai            → .../v1/systemone
      https://api.typesafe.ai/v1         → .../v1/systemone
      https://x/v1/chat/completions      → .../v1/systemone
      https://x/v1/systemone             → .../v1/systemone（不重复拼）
    """
    b = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    for suffix in _ENDPOINT_SUFFIXES:
        if b.endswith(suffix):
            b = b[: -len(suffix)].rstrip("/")
            break
    if b.endswith("/v1"):
        b = b[:-3].rstrip("/")
    return b + "/v1/systemone"


def _env(value: str) -> str:
    """支持 `$$ENV_NAME` 形式（与框架敏感配置一致）。"""
    v = (value or "").strip()
    if v.startswith("$$"):
        import os
        return os.environ.get(v[2:], "") or ""
    return v


# 不同提供商/第三方格式对这两个字段的叫法可能不同 ⇒ 多键名兜底
_BASE_KEYS = ("base_url", "api_base", "openai_api_base", "base", "endpoint",
              "url", "host", "server", "api_host")
_KEY_KEYS = ("api_key", "apikey", "api_token", "token", "key",
             "access_key", "secret", "auth_token")


def _pick(cfg: dict, names) -> str:
    """从 provider_config 里挑第一个非空的候选键；支持嵌套一层（如 {"openai": {...}}）。"""
    if not isinstance(cfg, dict):
        return ""
    for name in names:
        value = cfg.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in cfg.values():
        if isinstance(value, dict):
            found = _pick(value, names)
            if found:
                return found
    return ""


def split_ref(ref: str):
    """`provider_id:model_id` → 两段（model_id 允许含冒号）。"""
    pid, _, mid = (ref or "").partition(":")
    return pid.strip(), mid.strip()


class JevConfig:
    """不可变快照：从插件配置解析一次。

    links = [(base_url, api_key, model), ...] **有序回退链**：
      · 来自 webui 已注册提供商的模型多选（jev_models），自动继承接口地址/密钥/模型名；
      · 旧版手动连接字段已移除（3.1.1 起改用提供商注册）；
      · 缺地址或密钥的档在解析侧就被跳过（记日志），不会进链。
    """

    __slots__ = ("enabled", "links", "timeout", "min_confidence", "fallback_llm")

    def __init__(self, enabled=False, links=(), timeout=4.0,
                 min_confidence=0.0, fallback_llm=True):
        self.enabled = bool(enabled)
        norm = []
        for link in links or ():
            try:
                b, k, m = link
            except (TypeError, ValueError):
                continue
            b = (b or "").strip()
            k = _env(k or "")
            m = (m or DEFAULT_MODEL).strip() or DEFAULT_MODEL
            if b and k:
                norm.append((b, k, m))
        self.links = tuple(norm)
        try:
            self.timeout = max(0.5, float(timeout))
        except (TypeError, ValueError):
            self.timeout = 4.0
        try:
            self.min_confidence = min(1.0, max(0.0, float(min_confidence)))
        except (TypeError, ValueError):
            self.min_confidence = 0.0
        self.fallback_llm = bool(fallback_llm)

    @property
    def ready(self) -> bool:
        return bool(self.enabled and self.links)

    def client(self, logger) -> Optional["JevClient"]:
        if not self.ready:
            return None
        return JevClient(self.links, self.timeout, logger)


#: choice 题的固定三档（键是协议，不可改）；描述给 JEV 判读用。
DEFAULT_CRITERIA = {
    "up": "对话显示关系/剧情/亲密度明显升温，应进入更高阶段",
    "down": "对话出现冲突、冷淡、负面或倒退信号，应降低阶段",
    "stay": "状态平稳或信号不明确，维持当前阶段",
}


def build_stage_questions(instructions: str) -> dict:
    """构造 up/down/stay 判断题。instructions 来自用户提示词模板渲染结果。"""
    return {
        "stage_decision": {
            "type": "choice",
            "instructions": instructions,
            "criteria": dict(DEFAULT_CRITERIA),
        }
    }


def parse_choice(answers: dict, key: str = "stage_decision"):
    """→ (choice|None, confidence, probabilities)。"""
    item = (answers or {}).get(key) or {}
    if not isinstance(item, dict):
        return None, 0.0, {}
    choice = item.get("choice")
    try:
        conf = float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    probs = item.get("probabilities") if isinstance(item.get("probabilities"), dict) else {}
    return (choice if isinstance(choice, str) else None), conf, probs


class JevClient:
    """最小原生 JEV 客户端。任何失败 → None；连续失败自动熔断。

    有序回退链：每次调用从第 1 档开始逐档尝试（超时/连接失败/HTTP 错/坏 JSON
    ⇒ 换下一档），全档失败才计入熔断。链式结构与 Z 插件 _post_sync 同构。
    """

    COOLDOWN = 300.0

    def __init__(self, links, timeout: float, logger):
        # links: [(base_url, api_key, model), ...] 有序；第一条为主力
        self.links = tuple(links)
        self.timeout = float(timeout)
        self.logger = logger
        self.fails = 0
        self.opened_at = 0.0
        self.last_error = ""
        self.tokens = 0          # 累计输入 token（状态观测）
        self.calls = 0           # 成功调用次数

    @property
    def model(self) -> str:
        return self.links[0][2] if self.links else DEFAULT_MODEL

    @property
    def endpoint(self) -> str:
        return endpoint_of(self.links[0][0]) if self.links else ""

    # ---------- 状态 ----------
    @property
    def ready(self) -> bool:
        if not self.links:
            return False
        if self.fails >= 3 and (time.time() - self.opened_at) < self.COOLDOWN:
            return False
        return True

    @property
    def cooldown_left(self) -> int:
        if self.fails >= 3:
            left = self.COOLDOWN - (time.time() - self.opened_at)
            return int(left) if left > 0 else 0
        return 0

    def _ok(self):
        self.fails = 0

    def _bad(self, err: str = ""):
        self.fails += 1
        if err:
            self.last_error = str(err)[:200]
        if self.fails >= 3:
            # 每次失败都让冷却从当前时刻重新计时（否则冷却一次后熔断失效）
            self.opened_at = time.time()
            if self.fails == 3:
                self.logger.warning("JEV 连续失败 3 次，冷却 300 秒（期间自动回退）")

    # ---------- 同步实现 ----------
    def _post_sync(self, payload: dict, timeout: float) -> dict:
        """投递 —— 有序回退链逐档尝试，全档失败抛最后一个异常（由 call() 收敛）。"""
        last_exc = None
        for idx, (base, key, model_name) in enumerate(self.links, 1):
            body = dict(payload)
            if model_name:
                body["model"] = model_name
            req = urllib.request.Request(
                endpoint_of(base),
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": "Bearer " + (key or ""),
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8", "replace")
                return json.loads(raw)
            except Exception as exc:      # 逐档兜住，最后一档才抛
                last_exc = exc
                if len(self.links) > 1:
                    self.logger.info("JEV 第 %d 档失败（%s）⇒ 换下一档", idx, type(exc).__name__)
        raise last_exc if last_exc else RuntimeError("jev_no_endpoint")

    # ---------- 对外 ----------
    async def decide(self, state: str, instructions: str,
                     timeout: Optional[float] = None) -> Optional[dict]:
        """→ {"choice": str, "confidence": float, "probabilities": dict, "model": str}；
        任何失败/未就绪/熔断 → None。"""
        if not self.ready or not instructions:
            return None
        payload = {
            "model": self.model,
            "state": state or "",
            "questions": build_stage_questions(instructions),
        }
        limit = float(timeout or self.timeout)
        try:
            data = await asyncio.wait_for(
                asyncio.to_thread(self._post_sync, payload, limit), timeout=limit + 0.5
            )
        except Exception as exc:  # 契约：任何失败都不得外泄
            self._bad(str(exc))
            self.logger.info("JEV 判断未成功（%s），本轮走回退逻辑", type(exc).__name__)
            return None
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            self._bad("bad_response")
            return None
        choice, conf, probs = parse_choice(data["answers"])
        if choice not in ("up", "down", "stay"):
            self._bad("bad_choice")
            return None
        self._ok()
        self.calls += 1
        try:
            self.tokens += int((data.get("usage") or {}).get("input_tokens") or 0)
        except (TypeError, ValueError):
            pass
        return {
            "choice": choice,
            "confidence": conf,
            "probabilities": probs,
            "model": str(data.get("model") or self.model),
        }
