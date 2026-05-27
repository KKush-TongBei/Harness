from __future__ import annotations

from harness.evaluation import EvaluationResult


def _anchor_type(anchor: dict) -> str:
    return str(anchor.get("anchor_type", "support"))


def _max_score(anchors: list[dict]) -> float:
    if not anchors:
        return 0.0
    return max(float(a.get("score", 0.0)) for a in anchors)


def apply_fusion_policy(
    anchors: list[dict],
    forgery_score: float,
    model_result: EvaluationResult,
) -> EvaluationResult:
    """Deterministic post-fusion overrides to reduce TruFor/RAG false positives."""
    support = [a for a in anchors if _anchor_type(a) == "support"]
    warning = [a for a in anchors if _anchor_type(a) == "warning"]
    support_max = _max_score(support)
    warning_max = _max_score(warning)
    has_strong_support = support_max >= 1.0

    verdict = model_result.verdict
    reasoning = model_result.reasoning
    policy_notes: list[str] = []

    if not model_result.parse_ok and verdict == "fake":
        verdict = "suspicious"
        policy_notes.append("解析失败时不直接判 fake，降为 suspicious")

    if has_strong_support and forgery_score < 0.85 and verdict == "fake":
        verdict = "suspicious"
        policy_notes.append("存在合法场景锚点且 TruFor<0.85，禁止 fake")

    if has_strong_support and forgery_score < 0.85 and verdict == "suspicious" and forgery_score < 0.7:
        verdict = "authentic"
        policy_notes.append("合法锚点充分且 TruFor 较低，suspicious 升为 authentic")

    if not support and not warning and forgery_score < 0.9 and verdict == "fake":
        verdict = "suspicious"
        policy_notes.append("无锚点且 TruFor<0.9，禁止 fake")

    if (
        warning_max >= 2.0
        and support_max < 1.0
        and verdict == "authentic"
        and forgery_score >= 0.6
    ):
        verdict = "suspicious"
        policy_notes.append("风险警示锚点命中且无合法锚点，authentic 降为 suspicious")

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
    )
