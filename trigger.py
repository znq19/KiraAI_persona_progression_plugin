from typing import Optional, List, Dict
from core.logging_manager import get_logger
from core.provider import LLMRequest, LLMModelClient
from core.persona import PersonaManager

logger = get_logger("persona_progression.trigger", "cyan")


class TriggerEngine:
    def __init__(self, persona_mgr: PersonaManager, llm_client: Optional[LLMModelClient]):
        self.persona_mgr = persona_mgr
        self.llm_client = llm_client

    async def judge(self, prompt_template: str, stage: int, max_stage: int, conversation_text: str) -> str:
        """返回 'up', 'down', 或 'stay'"""
        if not self.llm_client:
            logger.warning("辅助 LLM 未配置，无法进行 AI 判断，默认 stay")
            return "stay"

        prompt = prompt_template.format(
            stage=stage,
            max_stage=max_stage,
            conversation_text=conversation_text
        )
        req = LLMRequest(messages=[{"role": "user", "content": prompt}])
        try:
            resp = await self.llm_client.chat(req)
            result = resp.text_response.strip().lower()
            if "up" in result:
                return "up"
            elif "down" in result:
                return "down"
            else:
                return "stay"
        except Exception as e:
            logger.error("AI 判断失败: %s", e)
            return "stay"

    async def apply_stage_global(self, stage: int, stage_map: Dict[int, str]) -> bool:
        """全局模式：切换激活人格"""
        persona_id = stage_map.get(stage)
        if not persona_id:
            logger.warning("阶段 %d 未配置对应人格 ID", stage)
            return False
        try:
            await self.persona_mgr.set_active_persona(persona_id)
            logger.info("全局人格切换至 %s (阶段 %d)", persona_id, stage)
            return True
        except Exception as e:
            logger.exception("切换全局人格失败: %s", e)
            return False

    def build_conversation_text(self, messages: List[dict], ignore_tool: bool, ignore_assistant: bool) -> str:
        """根据裁剪配置构建发给辅助 LLM 的对话文本"""
        lines = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if ignore_tool and role in ("tool", "function"):
                continue
            if ignore_assistant and role == "assistant":
                continue
            if role == "tool":
                lines.append(f"[工具结果] {content}")
            elif role == "assistant":
                lines.append(f"助手：{content}")
            elif role == "user":
                lines.append(f"用户：{content}")
            else:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)