#!/usr/bin/env python3
"""Offline validation tests that do not require Qwen/TruFor services."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.context import ContextManager, trufor_tier
from harness.evaluation import parse_model_output, verdict_matches_expected
from harness.news_post import NewsPost
from harness.tools.rag_client import RagClient


async def test_rag_client() -> None:
    from fastapi.testclient import TestClient
    import mocks.rag_server as rs

    rs._kb_cache = None
    from mocks.rag_server import app

    with TestClient(app) as tc:
        health = tc.get("/health")
        assert health.status_code == 200
        assert health.json().get("status") == "ok"

        resp = tc.post("/search", json={"query": "军事装备坦克士兵军演展示", "top_k": 3})
        assert resp.status_code == 200
        anchors = resp.json().get("anchors", [])
        assert len(anchors) >= 1, "Expected at least one anchor for military query"
        assert all(a.get("anchor_type") != "warning" for a in anchors)

        sports = tc.post(
            "/search",
            json={"query": "运动员在体育场跑道比赛，计分板信息", "top_k": 3},
        )
        sports_ids = [a["id"] for a in sports.json().get("anchors", [])]
        assert "misleading_context_warning" not in sports_ids

    try:
        client = RagClient("http://127.0.0.1:8002")
        await client.health()
        anchors = await client.search("军事装备坦克士兵军演展示", top_k=2)
        print(f"  RAG live client: {len(anchors)} anchors")
    except Exception:
        print(f"  RAG in-process: {len(anchors)} anchors (live server not running; restart RAG after pull)")


def test_context_fusion_prompt() -> None:
    ctx = ContextManager()
    news = NewsPost(
        image_path="data/test_cases/military_demo_01.jpg",
        headline="某军区举行公开日",
        body="主战坦克向民众展示",
        source="新华社",
    )
    prompt = ctx.build_fusion_prompt(
        description="几名士兵在展示坦克装备",
        anchors=[
            {
                "title": "军演",
                "source": "新华社",
                "date": "2024",
                "summary": "合法展示",
                "anchor_type": "support",
            }
        ],
        score=0.75,
        news=news,
    )
    assert trufor_tier(0.75) in prompt
    assert "合法场景锚点" in prompt
    assert "待检测新闻文本" in prompt
    assert "某军区举行公开日" in prompt
    print("  Context fusion prompt with news: OK")


def test_evaluation_parser() -> None:
    raw = (
        '{"verdict": "fake", "issue_type": "text_image_mismatch", '
        '"confidence": 0.9, "reasoning": "ok", "evidence_chain": ["a"]}'
    )
    r = parse_model_output(raw)
    assert r.parse_ok and r.verdict == "fake"
    assert r.issue_type == "text_image_mismatch"
    assert verdict_matches_expected("authentic", "authentic")
    print("  Evaluation parser with issue_type: OK")


def test_news_post_suite() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_news_post", ROOT / "scripts" / "test_news_post.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0
    print("  NewsPost suite: OK")


def test_fusion_policy_suite() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_fusion_policy", ROOT / "scripts" / "test_fusion_policy.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0
    print("  Fusion policy suite: OK")


async def main() -> int:
    print("=== Harness Offline Validation ===")
    test_context_fusion_prompt()
    test_evaluation_parser()
    test_news_post_suite()
    test_fusion_policy_suite()
    try:
        await test_rag_client()
    except Exception as e:
        print(f"  RAG client FAILED: {e}", file=sys.stderr)
        return 1
    print("All offline tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
