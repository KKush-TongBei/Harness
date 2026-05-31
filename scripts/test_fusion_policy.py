#!/usr/bin/env python3
"""Unit tests for fusion policy and typed RAG retrieval."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.evaluation import EvaluationResult
from harness.fusion_policy import apply_fusion_policy
from harness.news_post import NewsPost
from harness.rag_query import (
    build_misinformation_probe,
    build_official_support_probe,
)
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
        issue_type="unknown",
    )
    out = apply_fusion_policy(anchors, forgery_score=0.75, model_result=model)
    assert out.verdict == "suspicious", out.verdict
    print("  Policy: support + moderate TruFor blocks fake")


def test_suspicious_to_authentic_with_support() -> None:
    anchors = [
        {"anchor_type": "support", "score": 2.0, "id": "social_media_compression"},
    ]
    model = EvaluationResult(
        verdict="suspicious",
        confidence=0.7,
        reasoning="compression",
        evidence_chain=[],
        parse_ok=True,
        raw_text="",
        issue_type="unknown",
    )
    out = apply_fusion_policy(anchors, forgery_score=0.75, model_result=model)
    assert out.verdict == "authentic", out.verdict
    print("  Policy: support + TruFor 0.75 suspicious -> authentic")


def test_warning_escalates_authentic_to_fake() -> None:
    anchors = [
        {"anchor_type": "warning", "score": 3.5, "title": "谣言载体"},
        {"anchor_type": "support", "score": 1.0, "title": "弱匹配"},
    ]
    model = EvaluationResult(
        verdict="authentic",
        confidence=0.8,
        reasoning="looks real",
        evidence_chain=[],
        parse_ok=True,
        raw_text="",
        issue_type="unknown",
    )
    out = apply_fusion_policy(anchors, forgery_score=0.5, model_result=model)
    assert out.verdict == "fake", out.verdict
    print("  Policy: strong warning without strong support -> fake")


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
        issue_type="unknown",
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
        issue_type="unknown",
    )
    out = apply_fusion_policy([], forgery_score=0.85, model_result=model)
    assert out.verdict == "suspicious", out.verdict
    print("  Policy: no anchors + TruFor<0.9 blocks fake")


def test_text_issue_type_downgrades_authentic() -> None:
    model = EvaluationResult(
        verdict="authentic",
        confidence=0.8,
        reasoning="text mismatch ignored",
        evidence_chain=[],
        parse_ok=True,
        raw_text="",
        issue_type="text_image_mismatch",
    )
    out = apply_fusion_policy([], forgery_score=0.4, model_result=model)
    assert out.verdict == "suspicious", out.verdict
    print("  Policy: text_image_mismatch authentic -> suspicious")


def test_misinformation_probe() -> None:
    probe = build_misinformation_probe("场景环境：手机界面聊天截图，朋友圈转发")
    assert probe is not None
    assert "手机" in probe
    print("  RAG query: misinformation probe built from description")


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


def test_rag_warning_beats_weak_support() -> None:
    kb = _load_kb()
    q = "手机界面聊天截图 phone mobile share 谣言传播"
    scored = _score_all(q, kb)
    top = select_anchors(scored, top_k=3)
    assert top[0].get("anchor_type") == "warning"
    print("  RAG: warning ranks above weak support")


def test_rag_warning_only_when_support_weak() -> None:
    kb = _load_kb()
    q = "手机界面聊天截图朋友圈未核实来源谣言传播"
    scored = _score_all(q, kb)
    top = select_anchors(scored, top_k=3)
    types = [a.get("anchor_type") for a in top]
    assert "warning" in types
    print("  RAG: strong warning query includes warning anchor")


def test_weibo_psa_probe_skips_warning() -> None:
    text = "【急转！公安部刑侦局提醒：这8种情形肯定是诈骗】扩散！报警！"
    assert build_misinformation_probe(text) is None
    official = build_official_support_probe(text)
    assert official is not None
    assert "公安部" in official
    print("  RAG query: official PSA skips warning probe")


def test_weibo_support_retrieval() -> None:
    kb = _load_kb()
    q = "据中新网 印度猴子偷开巴士酿车祸 几乎车毁猴亡 微博转载"
    scored = _score_all(q, kb)
    top = select_anchors(scored, top_k=3)
    ids = [a["id"] for a in top]
    assert "weibo_media_repost" in ids
    print("  RAG: weibo media repost query hits support anchor")


def test_weibo_rumor_warning_retrieval() -> None:
    kb = _load_kb()
    q = "著名演员姜文去世 享年 表演艺术家 辞世"
    scored = _score_all(q, kb)
    top = select_anchors(scored, top_k=3)
    ids = [a["id"] for a in top]
    assert "celebrity_death_hoax" in ids
    print("  RAG: celebrity death hoax query hits warning anchor")


def test_weibo_support_overrides_text_mismatch() -> None:
    anchors = [
        {
            "anchor_type": "support",
            "score": 2.5,
            "id": "weibo_official_psa",
            "title": "公安部反诈",
        },
    ]
    news = NewsPost(
        image_path="weibo.jpg",
        headline="公安部刑侦局提醒",
        body="遇诈骗电话直接挂断",
        source="微博",
        fake_type="matching",
    )
    model = EvaluationResult(
        verdict="suspicious",
        confidence=0.7,
        reasoning="text mismatch",
        evidence_chain=[],
        parse_ok=True,
        raw_text="",
        issue_type="text_image_mismatch",
    )
    out = apply_fusion_policy(
        anchors, forgery_score=0.75, model_result=model, news=news
    )
    assert out.verdict == "authentic", out.verdict
    assert out.issue_type == "matching", out.issue_type
    print("  Policy: weibo support overrides text_image_mismatch")


def main() -> int:
    print("=== Fusion Policy & RAG Tests ===")
    test_support_blocks_fake_with_moderate_trufor()
    test_suspicious_to_authentic_with_support()
    test_warning_escalates_authentic_to_fake()
    test_warning_allows_fake_without_support()
    test_no_anchors_blocks_fake_below_trufor_threshold()
    test_text_issue_type_downgrades_authentic()
    test_misinformation_probe()
    test_rag_support_priority_no_warning_on_military()
    test_rag_sports_no_misleading_warning()
    test_rag_warning_beats_weak_support()
    test_rag_warning_only_when_support_weak()
    test_weibo_psa_probe_skips_warning()
    test_weibo_support_retrieval()
    test_weibo_rumor_warning_retrieval()
    test_weibo_support_overrides_text_mismatch()
    print("All fusion/RAG tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
