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


def _try_parse_json_dict(raw: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_json_block(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    obj = _try_parse_json_dict(text)
    if obj is not None:
        return obj

    # Closed markdown code fence
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        obj = _try_parse_json_dict(fence.group(1))
        if obj is not None:
            return obj

    # Unclosed fence (truncated model output)
    fence_open = re.search(r"```(?:json)?\s*(\{.*)", text, re.DOTALL)
    if fence_open:
        candidate = re.sub(r"\s*```\s*$", "", fence_open.group(1).strip())
        obj = _try_parse_json_dict(candidate)
        if obj is not None:
            return obj

    # First {...} block
    brace = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if brace:
        obj = _try_parse_json_dict(brace.group(0))
        if obj is not None:
            return obj

    return None


def _extract_partial_json_fields(text: str) -> dict[str, Any] | None:
    """Recover verdict/confidence/reasoning from truncated JSON text."""
    verdict_m = re.search(r'"verdict"\s*:\s*"(\w+)"', text, re.IGNORECASE)
    if not verdict_m:
        return None

    obj: dict[str, Any] = {"verdict": verdict_m.group(1)}

    conf_m = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
    if conf_m:
        obj["confidence"] = conf_m.group(1)

    reason_m = re.search(r'"reasoning"\s*:\s*"(.*)', text, re.DOTALL)
    if reason_m:
        raw = reason_m.group(1)
        end = re.search(r'(?<!\\)"\s*,\s*"evidence_chain"', raw)
        obj["reasoning"] = raw[: end.start()] if end else raw.rstrip('", \n\r\t')

    chain_m = re.search(r'"evidence_chain"\s*:\s*\[(.*?)(?:\]|$)', text, re.DOTALL)
    if chain_m:
        items = re.findall(r'"((?:[^"\\]|\\.)*)"', chain_m.group(1))
        if items:
            obj["evidence_chain"] = items

    return obj


def _result_from_obj(obj: dict[str, Any], text: str, *, parse_ok: bool) -> EvaluationResult:
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
        parse_ok=parse_ok,
        raw_text=text,
    )


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


def _fallback_verdict(text: str) -> str:
    """Keyword heuristics when structured JSON fields are unavailable."""
    lower = text.lower()
    fake_markers = (
        '"verdict": "fake"',
        '"verdict":"fake"',
        "verdict: fake",
        "tampered",
        "manipulated",
    )
    if any(m in lower for m in fake_markers):
        return "fake"
    auth_markers = (
        '"verdict": "authentic"',
        '"verdict":"authentic"',
        "verdict: authentic",
        "authentic",
        " genuine",
        "合法",
        "official",
        "真实",
    )
    if any(m in lower for m in auth_markers):
        return "authentic"
    if re.search(r"(?<![\w分数])伪造(?![分数检测])", text):
        return "fake"
    return "suspicious"


def parse_model_output(text: str) -> EvaluationResult:
    """Parse Qwen JSON verdict output with graceful fallback."""
    obj = _extract_json_block(text)
    if obj is not None:
        return _result_from_obj(obj, text, parse_ok=True)

    partial = _extract_partial_json_fields(text)
    if partial is not None:
        return _result_from_obj(partial, text, parse_ok=True)

    verdict = _fallback_verdict(text)
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
