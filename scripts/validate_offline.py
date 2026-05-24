#!/usr/bin/env python3
"""Offline validation tests that do not require Qwen/TruFor services."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.context import ContextManager
from harness.evaluation import parse_model_output, verdict_matches_expected
from harness.tools.rag_client import RagClient


async def test_rag_client() -> None:
    from fastapi.testclient import TestClient
    from mocks.rag_server import app

    # In-process test (no running server required)
    with TestClient(app) as tc:
        health = tc.get("/health")
        assert health.status_code == 200
        assert health.json().get("status") == "ok"

        resp = tc.post("/search", json={"query": "军事装备坦克士兵军演展示", "top_k": 2})
        assert resp.status_code == 200
        anchors = resp.json().get("anchors", [])
        assert len(anchors) >= 1, "Expected at least one anchor for military query"

    # Optional live HTTP test if server is running
    try:
        client = RagClient("http://127.0.0.1:8002")
        await client.health()
        anchors = await client.search("军事装备坦克士兵军演展示", top_k=2)
        print(f"  RAG live client: {len(anchors)} anchors")
    except Exception:
        print(f"  RAG in-process: {len(anchors)} anchors (live server not running)")


def test_context_fusion_prompt() -> None:
    ctx = ContextManager()
    prompt = ctx.build_fusion_prompt(
        description="几名士兵在展示坦克装备",
        anchors=[{"title": "军演", "source": "新华社", "date": "2024", "summary": "合法展示"}],
        score=0.85,
    )
    assert "0.8500" in prompt
    assert "军演" in prompt
    print("  Context fusion prompt: OK")


def test_evaluation_parser() -> None:
    raw = '{"verdict": "authentic", "confidence": 0.9, "reasoning": "ok", "evidence_chain": ["a"]}'
    r = parse_model_output(raw)
    assert r.parse_ok and r.verdict == "authentic"
    assert verdict_matches_expected("authentic", "authentic")
    print("  Evaluation parser: OK")


async def main() -> int:
    print("=== Harness Offline Validation ===")
    test_context_fusion_prompt()
    test_evaluation_parser()
    try:
        await test_rag_client()
    except Exception as e:
        print(f"  RAG client FAILED (is mock server running on :8002?): {e}", file=sys.stderr)
        return 1
    print("All offline tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
