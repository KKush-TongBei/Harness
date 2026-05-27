#!/usr/bin/env python3
"""Unit tests for fusion policy and typed RAG retrieval."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.evaluation import EvaluationResult
from harness.fusion_policy import apply_fusion_policy
import mocks.rag_server as rs
from mocks.rag_server import _load_kb, _score_all, select_anchors

rs._kb_cache = None


def test_support_blocks_fake_with_moderate_trufor() -> None:
    anchors = [
        {"anchor_type": "support", "score": 3.0, "title": "军演"},
    ]
    model = EvaluationResult(
        verdict="fake",
        confidence=0.9,
        reasoning="TruFor high",
        evidence_chain=[],
        parse_ok=True,
        raw_text="",
    )
    out = apply_fusion_policy(anchors, forgery_score=0.75, model_result=model)
    assert out.verdict == "suspicious", out.verdict
    print("  Policy: support + moderate TruFor blocks fake")


def test_warning_allows_fake_without_support() -> None:
    anchors = [
        {"anchor_type": "warning", "score": 3.0, "title": "谣言载体"},
    ]
    model = EvaluationResult(
        verdict="fake",
        confidence=0.85,
        reasoning="warning match",
        evidence_chain=[],
        parse_ok=True,
        raw_text="",
    )
    out = apply_fusion_policy(anchors, forgery_score=0.5, model_result=model)
    assert out.verdict == "fake", out.verdict
    print("  Policy: warning without support allows fake")


def test_no_anchors_blocks_fake_below_trufor_threshold() -> None:
    model = EvaluationResult(
        verdict="fake",
        confidence=0.8,
        reasoning="model guess",
        evidence_chain=[],
        parse_ok=True,
        raw_text="",
    )
    out = apply_fusion_policy([], forgery_score=0.85, model_result=model)
    assert out.verdict == "suspicious", out.verdict
    print("  Policy: no anchors + TruFor<0.9 blocks fake")


def test_rag_support_priority_no_warning_on_military() -> None:
    kb = _load_kb()
    q = "主体实体：主战坦克和士兵。场景环境：户外军事展示场地。"
    scored = _score_all(q, kb)
    top = select_anchors(scored, top_k=3)
    ids = [a["id"] for a in top]
    assert "military_exercise_2024" in ids
    assert not any(a.get("anchor_type") == "warning" for a in top)
    print("  RAG: military query returns support only")


def test_rag_sports_no_misleading_warning() -> None:
    kb = _load_kb()
    q = "主体实体：运动员。场景环境：体育场馆。可见文字：计分板信息。"
    scored = _score_all(q, kb)
    top = select_anchors(scored, top_k=3)
    ids = [a["id"] for a in top]
    assert "sports_event_live" in ids
    assert "misleading_context_warning" not in ids
    print("  RAG: sports query avoids misleading warning")


def test_rag_warning_only_when_support_weak() -> None:
    kb = _load_kb()
    q = "手机界面聊天截图朋友圈未核实来源谣言传播"
    scored = _score_all(q, kb)
    top = select_anchors(scored, top_k=3)
    types = [a.get("anchor_type") for a in top]
    assert "warning" in types
    print("  RAG: strong warning query includes warning anchor")


def main() -> int:
    print("=== Fusion Policy & RAG Tests ===")
    test_support_blocks_fake_with_moderate_trufor()
    test_warning_allows_fake_without_support()
    test_no_anchors_blocks_fake_below_trufor_threshold()
    test_rag_support_priority_no_warning_on_military()
    test_rag_sports_no_misleading_warning()
    test_rag_warning_only_when_support_weak()
    print("All fusion/RAG tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
