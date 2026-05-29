from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.news_post import NewsPost


DESCRIBE_PROMPT_IMAGE_ONLY = """请详细描述这张图片，按以下结构输出：
1. 主体实体：图片中的主要人物、物体或装备
2. 场景环境：拍摄地点、背景、天气等
3. 动作事件：正在发生什么
4. 可见文字：图片中出现的任何文字或标识
若画面为手机界面、社交聊天/朋友圈截图、电视或电脑屏幕上的信息、娱乐八卦配图，请在「场景环境」中明确写出。
请用客观、具体的语言描述，不要做任何真伪判断。"""

DESCRIBE_PROMPT_WITH_NEWS = """请对以下新闻条目进行图文联合理解（仅客观描述，不做真伪判决）。

{news_text_block}

请按以下结构输出：
1. 画面内容：图片中的主体、场景、动作、可见文字
2. 新闻文字声称的内容：标题与正文在说什么事件
3. 图文关系：画面与标题/正文是否明显一致；若不一致，客观说明差异（如场景不符、时间不符、主体不符）
4. 来源说明：若提供了来源，仅复述来源名称，不做可信度判断"""

NEWS_TEXT_BLOCK = """【待检测新闻文本】
标题：{headline}
正文：{body}
来源：{source}"""

BASELINE_PROMPT_IMAGE_ONLY = """请判断这张图片是否为伪造、篡改或虚假信息。
输出 JSON 格式：
{{
  "verdict": "authentic" 或 "suspicious" 或 "fake",
  "issue_type": "matching" 或 "manipulated_image" 或 "unknown",
  "confidence": 0.0 到 1.0 之间的数值,
  "reasoning": "简要说明判断依据"
}}"""

BASELINE_PROMPT_WITH_NEWS = """请判断这条新闻（配图+标题/正文/来源）整体是否为伪造、篡改、误导或虚假信息。

{news_text_block}

常见虚假类型包括：图像篡改(manipulated_image)、图文不匹配(text_image_mismatch)、新文旧图(new_text_old_image)、谣言配文(misleading_text)、图文均假(fabricated_both)。

输出 JSON 格式（不要包含 markdown 代码块）：
{{
  "verdict": "authentic" 或 "suspicious" 或 "fake",
  "issue_type": "matching" 或 "manipulated_image" 或 "text_image_mismatch" 或 "new_text_old_image" 或 "misleading_text" 或 "fabricated_both" 或 "unknown",
  "confidence": 0.0 到 1.0,
  "reasoning": "简要说明判断依据",
  "evidence_chain": ["证据1", "证据2"]
}}"""

FUSION_PROMPT_TEMPLATE = """你是一名多模态虚假信息检测专家。请综合以下信息，对整条新闻（图+文）做出最终判决。

{news_text_block}

【视觉与图文理解】
{description}

【合法场景锚点】
{support_anchors}

【谣言风险警示】
{warning_anchors}

【底层伪造检测】
分数 {score:.4f}，等级：{trufor_tier}

【评分规则】（按优先级）
1. 若有合法场景锚点且图文与描述一致 → 优先 authentic（即使 TruFor 分数偏高）
2. 标题/正文与画面明显矛盾 → issue_type=text_image_mismatch → 倾向 suspicious 或 fake
3. 标题声称近期突发事件，画面却是无关旧场景/通用图 → issue_type=new_text_old_image → 倾向 suspicious 或 fake
4. 来源不可信且标题/正文煽动、未经核实 → issue_type=misleading_text → 倾向 suspicious 或 fake
5. TruFor 分数高且图文无矛盾 → issue_type=manipulated_image → 倾向 suspicious（中等分数）或 fake（极高分数且视觉异常）
6. TruFor 分数 alone 不足以判 fake；合法锚点与图文一致时不应仅因 TruFor 判 fake
7. 合法场景锚点与风险警示同时出现且图文一致时，以合法锚点为准
8. 必须给出 issue_type 与可解释的证据链

请输出 JSON 格式（不要包含 markdown 代码块）：
{{
  "verdict": "authentic" 或 "suspicious" 或 "fake",
  "issue_type": "matching" 或 "manipulated_image" 或 "text_image_mismatch" 或 "new_text_old_image" 或 "misleading_text" 或 "fabricated_both" 或 "unknown",
  "confidence": 0.0 到 1.0,
  "reasoning": "综合推理说明",
  "evidence_chain": ["证据1", "证据2"]
}}"""


def trufor_tier(score: float) -> str:
    if score < 0.6:
        return "低（误报概率较小）"
    if score <= 0.85:
        return "中（压缩/增强等可能导致误报）"
    return "高（需结合锚点与图文一致性综合判断， alone 不足以判 fake）"


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


def _format_news_text_block(news: NewsPost | None) -> str:
    if news is None or not news.has_text():
        return ""
    return NEWS_TEXT_BLOCK.format(
        headline=news.headline or "（无）",
        body=news.body or "（无）",
        source=news.source or "（无）",
    )


@dataclass(frozen=True)
class ContextManager:
    """Inject system prompts and stage-specific templates."""

    describe_prompt_image_only: str = DESCRIBE_PROMPT_IMAGE_ONLY
    describe_prompt_with_news: str = DESCRIBE_PROMPT_WITH_NEWS
    baseline_prompt_image_only: str = BASELINE_PROMPT_IMAGE_ONLY
    baseline_prompt_with_news: str = BASELINE_PROMPT_WITH_NEWS
    fusion_template: str = FUSION_PROMPT_TEMPLATE

    def build_describe_prompt(self, news: NewsPost | None = None) -> str:
        if news is None or not news.has_text():
            return self.describe_prompt_image_only
        return self.describe_prompt_with_news.format(
            news_text_block=_format_news_text_block(news)
        )

    def build_baseline_prompt(self, news: NewsPost | None = None) -> str:
        if news is None or not news.has_text():
            return self.baseline_prompt_image_only
        return self.baseline_prompt_with_news.format(
            news_text_block=_format_news_text_block(news)
        )

    def build_fusion_prompt(
        self,
        description: str,
        anchors: list[dict[str, Any]],
        score: float,
        news: NewsPost | None = None,
    ) -> str:
        support = [a for a in anchors if a.get("anchor_type", "support") != "warning"]
        warning = [a for a in anchors if a.get("anchor_type") == "warning"]
        news_block = _format_news_text_block(news)
        if not news_block:
            news_block = "【待检测新闻文本】\n（未提供标题/正文，仅基于图像与传感器信息判断）"

        return self.fusion_template.format(
            news_text_block=news_block,
            description=description,
            support_anchors=_format_anchor_lines(support),
            warning_anchors=_format_anchor_lines(warning),
            score=score,
            trufor_tier=trufor_tier(score),
        )
