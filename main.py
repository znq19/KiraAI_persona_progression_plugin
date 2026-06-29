from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from core.plugin import BasePlugin, logger, on, Priority, register
from core.provider import LLMRequest, LLMModelClient
from core.chat.message_utils import KiraMessageBatchEvent, KiraMessageEvent
from core.chat import MessageChain
from core.chat.message_elements import Text
from core.persona import PersonaManager
from core.prompt_manager import Prompt

from .state_manager import StateManager
from .trigger import TriggerEngine


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
        self.initial_stage = 0
        self.enable_debug_log = False

        # 触发配置
        self.trigger_interval = 10
        self.enable_ai = True
        self.aux_model = None
        self.judge_prompt = ""
        self.ignore_tool = True
        self.ignore_assistant = False

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
        self.clear_context_on_switch = False  # ✅ 新增

        # 权限与反馈配置
        self.enable_permission = False
        self.allowed_users: List[str] = []
        self.permission_denied_message = "❌ 权限不足：您没有权限切换人格阶段"
        self.success_message = "✅ 已切换人格阶段：{action} 至阶段 {stage}"
        self.error_message = "❌ 切换失败：{error}"
        self.max_stage_message = "⚠️ 已达最大阶段（{max_stage}），无法继续进阶"
        self.min_stage_message = "⚠️ 已达最低阶段（{min_stage}），无法继续降级"

    async def initialize(self):
        self.data_dir = self.ctx.get_plugin_data_dir()
        if self.data_dir is None:
            logger.error("无法获取插件数据目录")
            return

        cfg = self.plugin_cfg

        # 基础设置
        basic = cfg.get("section_basic", {})
        self.enabled = basic.get("enabled", True)
        self.persona_scope = basic.get("persona_scope", "session")
        self.total_stages = basic.get("total_stages", 5)
        self.initial_stage = basic.get("initial_stage", 0)
        self.enable_debug_log = basic.get("enable_debug_log", False)

        self.persona_mgr = self.ctx.persona_mgr
        all_personas = await self.persona_mgr.list_personas()
        persona_lookup: Dict[str, str] = {}
        for p in all_personas:
            persona_lookup[p.id] = p.id
            if p.name:
                persona_lookup[p.name] = p.id
        if self.enable_debug_log:
            logger.debug(f"人格查找表: {persona_lookup}")

        raw_map = basic.get("stage_persona_map", [])
        self.stage_map = {}
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
                real_id = persona_lookup.get(identifier)
                if not real_id:
                    logger.warning(f"找不到人格标识 '{identifier}'，请检查名称或ID是否正确")
                    continue
                if stage in self.stage_map:
                    logger.warning(f"阶段 {stage} 被重复定义，新值 {real_id} 将覆盖旧值 {self.stage_map[stage]}")
                self.stage_map[stage] = real_id
                if self.enable_debug_log:
                    logger.debug(f"映射阶段 {stage} -> 人格ID {real_id} (来自标识 '{identifier}')")
        else:
            logger.warning("stage_persona_map 配置不是列表，请检查配置格式")
            self.stage_map = {}

        if self.stage_map:
            missing = [str(i) for i in range(self.total_stages) if i not in self.stage_map]
            if missing:
                logger.warning(f"以下阶段未配置人格映射: {', '.join(missing)}")

        # 触发设置
        trigger = cfg.get("section_trigger", {})
        self.trigger_interval = trigger.get("trigger_interval", 10)
        self.enable_ai = trigger.get("enable_ai_judge", True)
        self.ignore_tool = trigger.get("ignore_tool_messages", True)
        self.ignore_assistant = trigger.get("ignore_assistant_messages", False)
        aux_model_uuid = trigger.get("auxiliary_model", "")
        if aux_model_uuid:
            self.aux_model = self.ctx.get_llm_client(model_uuid=aux_model_uuid)
        else:
            self.aux_model = self.ctx.get_default_fast_llm_client()
        if not self.aux_model:
            logger.warning("辅助 LLM 不可用，AI 判断将失效")
            self.enable_ai = False

        # AI 判断提示词
        ai_section = cfg.get("section_ai", {})
        self.judge_prompt = ai_section.get("judge_prompt", "")

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
        self.enable_tool = manual.get("enable_tool", True)
        self.reset_on_manual = manual.get("reset_on_manual", False)
        self.clear_context_on_switch = manual.get("clear_context_on_switch", False)  # ✅ 新增

        # 权限与反馈
        perm = cfg.get("section_permission", {})
        self.enable_permission = perm.get("enable_permission", False)
        self.allowed_users = [str(uid).strip() for uid in perm.get("allowed_users", []) if str(uid).strip()]
        self.permission_denied_message = perm.get("permission_denied_message", "❌ 权限不足：您没有权限切换人格阶段")
        self.success_message = perm.get("success_message", "✅ 已切换人格阶段：{action} 至阶段 {stage}")
        self.error_message = perm.get("error_message", "❌ 切换失败：{error}")
        self.max_stage_message = perm.get("max_stage_message", "⚠️ 已达最大阶段（{max_stage}），无法继续进阶")
        self.min_stage_message = perm.get("min_stage_message", "⚠️ 已达最低阶段（{min_stage}），无法继续降级")

        # 初始化状态管理器
        self.state_mgr = StateManager(self.data_dir, self.persona_scope)
        self.trigger_engine = TriggerEngine(self.persona_mgr, self.aux_model if self.enable_ai else None)

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
        logger.info("人设进阶系统终止")

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
                    logger.debug("全局人格切换至阶段 %d，人格ID: %s", new_stage, persona_id)
                # ✅ 新增：切换成功后清除上下文（全局模式也支持，按会话清除）
                if self.clear_context_on_switch:
                    await self._clear_session_context(session_id)
            return ok
        else:
            self.state_mgr.set_stage(session_id, new_stage)
            if self.enable_debug_log:
                logger.debug("会话 %s 阶段 %d → %d (session 人格待注入，人格ID: %s)",
                             session_id, current, new_stage, persona_id)
            # ✅ 新增：切换成功后清除上下文
            if self.clear_context_on_switch:
                await self._clear_session_context(session_id)
            return True

    async def _clear_session_context(self, session_id: str) -> None:
        """清除会话上下文记忆（参考 reboot 插件实现）"""
        try:
            if self.ctx.session_mgr is None:
                logger.warning("SessionManager 不可用，无法清除上下文")
                return
            self.ctx.session_mgr.delete_session(session_id)
            # 重新初始化会话（重建空记忆）
            self.ctx.session_mgr.get_session_info(session_id)
            if self.enable_debug_log:
                logger.debug("已清除会话上下文: %s", session_id)
        except Exception as e:
            logger.warning("清除会话上下文失败 (%s): %s", session_id, e)

    async def _process_trigger(self, session_id: str):
        if not self.trigger_engine:
            return
        current_stage = self.state_mgr.get_stage(session_id, self.initial_stage)
        if self.enable_debug_log:
            logger.debug("触发检测: session=%s, 当前阶段=%d", session_id, current_stage)

        history = self.ctx.session_mgr.fetch_memory(session_id)
        max_history = 50
        if len(history) > max_history:
            history = history[-max_history:]
            if self.enable_debug_log:
                logger.debug("历史消息截断至最近 %d 条", max_history)

        conversation_text = self.trigger_engine.build_conversation_text(
            history,
            ignore_tool=self.ignore_tool,
            ignore_assistant=self.ignore_assistant
        )
        if self.enable_debug_log:
            logger.debug("发送给辅助LLM的对话文本长度: %d 字符", len(conversation_text))

        if self.enable_ai and self.aux_model:
            decision = await self.trigger_engine.judge(
                self.judge_prompt,
                current_stage,
                self.total_stages - 1,
                conversation_text
            )
            if self.enable_debug_log:
                logger.debug("AI 判断结果: %s", decision)
        else:
            decision = "stay"
            if self.enable_debug_log:
                logger.debug("AI 判断未启用，默认 stay")

        new_stage = current_stage
        if decision == "up" and current_stage < self.total_stages - 1:
            new_stage = current_stage + 1
            self.state_mgr.set_consecutive_stay(session_id, 0)
            if self.enable_debug_log:
                logger.debug("AI 决定升级，新阶段 %d", new_stage)
        elif decision == "down" and current_stage > 0:
            new_stage = current_stage - 1
            self.state_mgr.set_consecutive_stay(session_id, 0)
            if self.enable_debug_log:
                logger.debug("AI 决定降级，新阶段 %d", new_stage)
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
                    logger.debug("AI 决定保持，且衰退未启用")

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
        self.state_mgr.increment_turn(session_id)
        turn = self.state_mgr.get_turn_counter(session_id)
        if self.enable_debug_log:
            logger.debug("会话 %s 当前轮数: %d / %d", session_id, turn, self.trigger_interval)

        if turn >= self.trigger_interval:
            if self.enable_debug_log:
                logger.debug("达到触发间隔，开始检测")
            await self._process_trigger(session_id)

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

        # 清空计数（无需权限）
        if text_lower in self.clear_keywords:
            self.state_mgr.reset_turn_counter(session_id)
            if self.enable_debug_log:
                logger.debug("会话 %s 清空计数关键词触发", session_id)
            await self.ctx.message_processor.send_message_chain(
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
                await self.ctx.message_processor.send_message_chain(
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

            await self.ctx.message_processor.send_message_chain(
                session=session_id,
                chain=MessageChain([Text(reply)])
            )
            event.discard(force=True)
            event.stop()
            return

    # ---------- LLM 工具 ----------
    @register.tool(
        name="persona_get_stage",
        description="获取当前会话的人格阶段",
        params={"type": "object", "properties": {}}
    )
    async def tool_get_stage(self, event: KiraMessageBatchEvent, *_) -> str:
        if not self.state_mgr:
            return "插件未初始化"
        session_id = event.sid
        stage = self.state_mgr.get_stage(session_id, self.initial_stage)
        return f"当前阶段：{stage} / {self.total_stages - 1}"

    @register.tool(
        name="persona_set_stage",
        description="手动设置当前会话的人格阶段",
        params={
            "type": "object",
            "properties": {
                "stage": {"type": "integer", "description": "目标阶段（0 ~ max）"}
            },
            "required": ["stage"]
        }
    )
    async def tool_set_stage(self, event: KiraMessageBatchEvent, *_, stage: int) -> str:
        if not self.state_mgr:
            return "插件未初始化"
        if not self._check_permission(event):
            return self.permission_denied_message

        session_id = event.sid
        if stage < 0 or stage >= self.total_stages:
            return f"阶段必须在 0 ~ {self.total_stages - 1} 之间"
        old = self.state_mgr.get_stage(session_id, self.initial_stage)
        ok = await self._change_stage(session_id, stage)
        if ok:
            if self.reset_on_manual:
                self.state_mgr.reset_turn_counter(session_id)
            if self.enable_debug_log:
                logger.debug("工具 set_stage: %s 阶段 %d -> %d", session_id, old, stage)
            return self.success_message.format(action="设置", stage=stage)
        else:
            return self.error_message.format(error="设置失败，请检查人格映射")

    @register.tool(
        name="persona_advance",
        description="手动将当前会话阶段提升一级",
        params={"type": "object", "properties": {}}
    )
    async def tool_advance(self, event: KiraMessageBatchEvent, *_) -> str:
        if not self.state_mgr:
            return "插件未初始化"
        if not self._check_permission(event):
            return self.permission_denied_message

        session_id = event.sid
        current = self.state_mgr.get_stage(session_id, self.initial_stage)
        if current >= self.total_stages - 1:
            return self.max_stage_message.format(
                max_stage=self.total_stages - 1,
                current_stage=current
            )
        new = current + 1
        ok = await self._change_stage(session_id, new)
        if ok:
            if self.reset_on_manual:
                self.state_mgr.reset_turn_counter(session_id)
            if self.enable_debug_log:
                logger.debug("工具 advance: %s 阶段 %d -> %d", session_id, current, new)
            return self.success_message.format(action="进阶", stage=new)
        else:
            return self.error_message.format(error="升级失败，请检查人格映射")

    @register.tool(
        name="persona_decay",
        description="手动将当前会话阶段降低一级",
        params={"type": "object", "properties": {}}
    )
    async def tool_decay(self, event: KiraMessageBatchEvent, *_) -> str:
        if not self.state_mgr:
            return "插件未初始化"
        if not self._check_permission(event):
            return self.permission_denied_message

        session_id = event.sid
        current = self.state_mgr.get_stage(session_id, self.initial_stage)
        if current <= 0:
            return self.min_stage_message.format(
                min_stage=0,
                current_stage=current
            )
        new = current - 1
        ok = await self._change_stage(session_id, new)
        if ok:
            if self.reset_on_manual:
                self.state_mgr.reset_turn_counter(session_id)
            if self.enable_debug_log:
                logger.debug("工具 decay: %s 阶段 %d -> %d", session_id, current, new)
            return self.success_message.format(action="降级", stage=new)
        else:
            return self.error_message.format(error="降级失败，请检查人格映射")

    @register.tool(
        name="persona_reset_counter",
        description="重置当前会话的轮数计数器（不影响阶段）",
        params={"type": "object", "properties": {}}
    )
    async def tool_reset_counter(self, event: KiraMessageBatchEvent, *_) -> str:
        if not self.state_mgr:
            return "插件未初始化"
        if not self._check_permission(event):
            return self.permission_denied_message

        session_id = event.sid
        self.state_mgr.reset_turn_counter(session_id)
        if self.enable_debug_log:
            logger.debug("工具 reset_counter: %s 计数器已重置", session_id)
        return "轮数计数器已重置"

    # ---------- HTTP API ----------
    @register.api(method="GET", path="/stage", auth=True)
    async def api_get_stage(self, session_id: str) -> dict:
        if not self.state_mgr:
            return {"error": "not initialized"}
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
        try:
            stage = int(stage)
        except:
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
        stage = self.state_mgr.get_stage(session_id, self.initial_stage)
        persona_id = self.stage_map.get(stage)
        return {"stage": stage, "persona_id": persona_id}

    # ---------- session 模式人格注入 ----------
    @on.llm_request(priority=Priority.SYS_HIGH)
    async def inject_session_persona(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        if not self.enabled or self.persona_scope != "session":
            return
        session_id = event.sid
        stage = self.state_mgr.get_stage(session_id, self.initial_stage)
        persona_id = self.stage_map.get(stage)
        if not persona_id:
            if self.enable_debug_log:
                logger.debug("阶段 %d 未配置人格映射，跳过注入", stage)
            return
        persona_info = await self.persona_mgr.get_persona(persona_id)
        if not persona_info:
            logger.warning("人格 ID '%s' 不存在，无法注入", persona_id)
            return
        persona_text = persona_info.content

        if self.enable_debug_log:
            logger.debug("注入人格: %s (阶段 %d), 内容长度: %d", persona_id, stage, len(persona_text))

        for p in req.system_prompt:
            if p.name == "persona":
                p.kwargs["persona"] = persona_text
                if self.enable_debug_log:
                    logger.debug("已注入 session 人格到 kwargs: %s (阶段 %d)", persona_id, stage)
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