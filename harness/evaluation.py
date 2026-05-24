from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


VALID_VERDICTS = {"authentic", "suspicious", "fake"}


@dataclass
class EvaluationResult:
    verdict: str
    confidence: float
    reasoning: str
    evidence_chain: list[str]
    parse_ok: bool
    raw_text: str


def _extract_json_block(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    # Direct JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Markdown code fence
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # First {...} block
    brace = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if brace:
        try:
            obj = json.loads(brace.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def _normalize_verdict(raw: str | None) -> str:
    if not raw:
        return "suspicious"
    v = raw.strip().lower()
    if v in VALID_VERDICTS:
        return v
    mapping = {
        "真实": "authentic",
        "genuine": "authentic",
        "real": "authentic",
        "正常": "authentic",
        "可疑": "suspicious",
        "不确定": "suspicious",
        "伪造": "fake",
        "虚假": "fake",
        "tampered": "fake",
        "manipulated": "fake",
    }
    for key, val in mapping.items():
        if key in v:
            return val
    return "suspicious"


def _clamp_confidence(value: Any) -> float:
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, c))


def parse_model_output(text: str) -> EvaluationResult:
    """Parse Qwen JSON verdict output with graceful fallback."""
    obj = _extract_json_block(text)

    if obj is not None:
        verdict = _normalize_verdict(str(obj.get("verdict", "")))
        confidence = _clamp_confidence(obj.get("confidence", 0.5))
        reasoning = str(obj.get("reasoning", "")).strip()
        chain_raw = obj.get("evidence_chain", [])
        if isinstance(chain_raw, list):
            evidence_chain = [str(x) for x in chain_raw]
        else:
            evidence_chain = [str(chain_raw)] if chain_raw else []
        return EvaluationResult(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning or text[:500],
            evidence_chain=evidence_chain,
            parse_ok=True,
            raw_text=text,
        )

    # Fallback: keyword heuristics on free text
    lower = text.lower()
    if any(w in lower for w in ("伪造", "fake", "tampered", " manipulated")):
        verdict = "fake"
    elif any(w in lower for w in ("真实", "authentic", " genuine", "合法", "official")):
        verdict = "authentic"
    else:
        verdict = "suspicious"

    return EvaluationResult(
        verdict=verdict,
        confidence=0.3,
        reasoning=text.strip()[:1000],
        evidence_chain=[],
        parse_ok=False,
        raw_text=text,
    )


def verdict_matches_expected(verdict: str | None, expected: str) -> bool:
    if verdict is None:
        return False
    v = verdict.lower()
    e = expected.lower()
    if e == "authentic":
        return v == "authentic"
    if e in ("fake", "suspicious"):
        return v in ("fake", "suspicious")
    return v == e
