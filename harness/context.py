from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DESCRIBE_PROMPT = """请详细描述这张图片，按以下结构输出：
1. 主体实体：图片中的主要人物、物体或装备
2. 场景环境：拍摄地点、背景、天气等
3. 动作事件：正在发生什么
4. 可见文字：图片中出现的任何文字或标识
若画面为手机界面、社交聊天/朋友圈截图、电视或电脑屏幕上的信息、娱乐八卦配图，请在「场景环境」中明确写出。
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

【合法场景锚点】
{support_anchors}

【谣言风险警示】
{warning_anchors}

【底层伪造检测】
分数 {score:.4f}，等级：{trufor_tier}

【评分规则】（按优先级）
1. 若有合法场景锚点且与描述一致 → 优先 authentic（即使 TruFor 分数偏高）
2. 若「谣言风险警示」段落有命中且与描述高度吻合，且无对应合法场景锚点 → 倾向 suspicious 或 fake
3. TruFor 分数 alone 不足以判 fake；中等分数最高判 suspicious
4. 仅有 TruFor 高分、无任何风险警示、且无合法锚点 → suspicious（非 fake）
5. 合法场景锚点与风险警示同时出现时，以合法场景锚点为准，不要仅因 TruFor 高分判 fake
6. 必须给出可解释的证据链

请输出 JSON 格式（不要包含 markdown 代码块）：
{{
  "verdict": "authentic" 或 "suspicious" 或 "fake",
  "confidence": 0.0 到 1.0,
  "reasoning": "综合推理说明",
  "evidence_chain": ["证据1", "证据2"]
}}"""


def trufor_tier(score: float) -> str:
    if score < 0.6:
        return "低（误报概率较小）"
    if score <= 0.85:
        return "中（压缩/增强等可能导致误报）"
    return "高（需结合锚点综合判断， alone 不足以判 fake）"


def _format_anchor_lines(anchors: list[dict[str, Any]]) -> str:
    if not anchors:
        return "（无）"
    lines = []
    for i, a in enumerate(anchors, 1):
        lines.append(
            f"{i}. [{a.get('source', '未知来源')}] {a.get('title', '')} "
            f"({a.get('date', '')}): {a.get('summary', '')}"
        )
    return "\n".join(lines)


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
        support = [a for a in anchors if a.get("anchor_type", "support") != "warning"]
        warning = [a for a in anchors if a.get("anchor_type") == "warning"]

        return self.fusion_template.format(
            description=description,
            support_anchors=_format_anchor_lines(support),
            warning_anchors=_format_anchor_lines(warning),
            score=score,
            trufor_tier=trufor_tier(score),
        )
