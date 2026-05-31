from __future__ import annotations

from harness.evaluation import EvaluationResult
from harness.news_post import NewsPost

TEXT_ISSUE_TYPES = {
    "text_image_mismatch",
    "new_text_old_image",
    "misleading_text",
    "fabricated_both",
}

WEIBO_SUPPORT_IDS = {
    "weibo_media_repost",
    "weibo_official_psa",
    "weibo_history_education",
    "weibo_lifestyle_share",
    "weibo_news_investigation",
    "weibo_general_repost",
}


def _anchor_type(anchor: dict) -> str:
    return str(anchor.get("anchor_type", "support"))


def _max_score(anchors: list[dict]) -> float:
    if not anchors:
        return 0.0
    return max(float(a.get("score", 0.0)) for a in anchors)


def _has_anchor_id(anchors: list[dict], anchor_id: str) -> bool:
    return any(str(a.get("id", "")) == anchor_id for a in anchors)


def _has_strong_weibo_support(support: list[dict]) -> bool:
    return any(
        str(a.get("id", "")) in WEIBO_SUPPORT_IDS and float(a.get("score", 0.0)) >= 2.0
        for a in support
    )


def apply_fusion_policy(
    anchors: list[dict],
    forgery_score: float,
    model_result: EvaluationResult,
    news: NewsPost | None = None,
) -> EvaluationResult:
    """Deterministic post-fusion overrides for image+text news posts."""
    support = [a for a in anchors if _anchor_type(a) == "support"]
    warning = [a for a in anchors if _anchor_type(a) == "warning"]
    support_max = _max_score(support)
    warning_max = _max_score(warning)
    has_strong_support = support_max >= 1.0
    has_strong_support_strict = support_max > 1.0

    verdict = model_result.verdict
    model_verdict = model_result.verdict
    issue_type = model_result.issue_type
    reasoning = model_result.reasoning
    policy_notes: list[str] = []

    if not model_result.parse_ok and verdict == "fake":
        verdict = "suspicious"
        policy_notes.append("解析失败时不直接判 fake，降为 suspicious")

    if issue_type in TEXT_ISSUE_TYPES and verdict == "authentic":
        verdict = "suspicious"
        policy_notes.append(f"模型识别为 {issue_type}，authentic 降为 suspicious")

    if issue_type in TEXT_ISSUE_TYPES and warning_max >= 2.0 and verdict == "suspicious":
        verdict = "fake"
        policy_notes.append(f"图文类问题 {issue_type} 且风险警示命中，升为 fake")

    if (
        _has_anchor_id(support, "social_media_compression")
        and forgery_score < 0.9
        and verdict in ("fake", "suspicious")
        and issue_type not in TEXT_ISSUE_TYPES
    ):
        verdict = "authentic"
        issue_type = "matching"
        policy_notes.append("社交媒体压缩合法锚点命中，强制 authentic")

    if (
        news
        and news.fake_type == "matching"
        and _has_strong_weibo_support(support)
        and forgery_score < 0.85
        and verdict in ("fake", "suspicious")
        and warning_max < 3.0
    ):
        verdict = "authentic"
        issue_type = "matching"
        policy_notes.append("微博合法场景锚点命中，覆盖 text_image_mismatch 误报")

    if (
        has_strong_support
        and forgery_score < 0.85
        and verdict == "fake"
        and issue_type not in TEXT_ISSUE_TYPES
    ):
        verdict = "suspicious"
        policy_notes.append("存在合法场景锚点且 TruFor<0.85，禁止 fake")

    if (
        has_strong_support
        and forgery_score < 0.85
        and verdict == "suspicious"
        and model_verdict == "suspicious"
        and issue_type in ("matching", "unknown", "manipulated_image")
    ):
        verdict = "authentic"
        issue_type = "matching"
        policy_notes.append("合法锚点充分且 TruFor<0.85，suspicious 升为 authentic")

    if not support and not warning and forgery_score < 0.9 and verdict == "fake":
        if issue_type not in TEXT_ISSUE_TYPES:
            verdict = "suspicious"
            policy_notes.append("无锚点且 TruFor<0.9，禁止 fake")

    if (
        news
        and news.has_text()
        and news.fake_type in TEXT_ISSUE_TYPES
        and not has_strong_support_strict
        and warning_max >= 2.0
        and verdict == "authentic"
    ):
        verdict = "suspicious"
        if issue_type == "unknown":
            issue_type = news.fake_type
        policy_notes.append("标注为图文类假新闻且风险警示命中，authentic 降为 suspicious")

    if (
        not has_strong_support_strict
        and warning_max >= 3.0
        and verdict in ("authentic", "suspicious")
        and issue_type not in ("matching",)
    ):
        verdict = "fake"
        policy_notes.append("强风险警示命中且无强合法锚点，升为 fake")

    elif (
        not has_strong_support_strict
        and warning_max >= 2.0
        and verdict == "authentic"
        and issue_type not in ("matching",)
    ):
        verdict = "suspicious"
        policy_notes.append("风险警示锚点命中且无强合法锚点，authentic 降为 suspicious")

    if policy_notes:
        note = " [融合策略] " + "; ".join(policy_notes)
        reasoning = (reasoning + note).strip()

    return EvaluationResult(
        verdict=verdict,
        confidence=model_result.confidence,
        reasoning=reasoning,
        evidence_chain=model_result.evidence_chain,
        parse_ok=model_result.parse_ok,
        raw_text=model_result.raw_text,
        issue_type=issue_type,
    )
