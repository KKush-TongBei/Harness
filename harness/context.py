from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DESCRIBE_PROMPT = """请详细描述这张图片，按以下结构输出：
1. 主体实体：图片中的主要人物、物体或装备
2. 场景环境：拍摄地点、背景、天气等
3. 动作事件：正在发生什么
4. 可见文字：图片中出现的任何文字或标识
请用客观、具体的语言描述，不要做任何真伪判断。"""

BASELINE_PROMPT = """请判断这张图片是否为伪造、篡改或虚假信息。
输出 JSON 格式：
{
  "verdict": "authentic" 或 "suspicious" 或 "fake",
  "confidence": 0.0 到 1.0 之间的数值,
  "reasoning": "简要说明判断依据"
}"""

FUSION_PROMPT_TEMPLATE = """你是一名多模态虚假信息检测专家。请综合以下信息做出最终判决。

【待检测内容描述】
{description}

【外部可信锚点】
{anchors}

【底层伪造检测分数】
{score:.4f}（范围 0~1，越高表示底层模型认为越可疑；注意：新闻图片压缩、色彩增强、合法军事展示等可能导致高分误报）

【评分规则】
- 若外部锚点表明内容为合法公开报道/官方活动，且描述与锚点一致，即使伪造分数较高也应倾向 authentic
- 若描述与锚点矛盾，或无任何锚点且伪造分数极高，倾向 suspicious 或 fake
- 必须给出可解释的证据链

请输出 JSON 格式（不要包含 markdown 代码块）：
{{
  "verdict": "authentic" 或 "suspicious" 或 "fake",
  "confidence": 0.0 到 1.0,
  "reasoning": "综合推理说明",
  "evidence_chain": ["证据1", "证据2"]
}}"""


@dataclass(frozen=True)
class ContextManager:
    """Inject system prompts and stage-specific templates."""

    describe_prompt: str = DESCRIBE_PROMPT
    baseline_prompt: str = BASELINE_PROMPT
    fusion_template: str = FUSION_PROMPT_TEMPLATE

    def build_fusion_prompt(
        self,
        description: str,
        anchors: list[dict[str, Any]],
        score: float,
    ) -> str:
        if anchors:
            anchor_lines = []
            for i, a in enumerate(anchors, 1):
                anchor_lines.append(
                    f"{i}. [{a.get('source', '未知来源')}] {a.get('title', '')} "
                    f"({a.get('date', '')}): {a.get('summary', '')}"
                )
            anchors_text = "\n".join(anchor_lines)
        else:
            anchors_text = "（未检索到相关可信锚点）"

        return self.fusion_template.format(
            description=description,
            anchors=anchors_text,
            score=score,
        )
