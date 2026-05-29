#!/usr/bin/env python3
"""Tests for NewsPost model and metadata loading."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.news_post import NewsPost

METADATA = ROOT / "data" / "test_cases" / "metadata.json"


def test_news_post_has_text() -> None:
    post = NewsPost(
        image_path="/tmp/a.jpg",
        headline="标题",
        body="",
        source="",
    )
    assert post.has_text()
    print("  NewsPost.has_text: OK")


def test_rag_query_suffix() -> None:
    post = NewsPost(
        image_path="/tmp/a.jpg",
        headline="暴雨内涝",
        body="观众被困",
        source="微信朋友圈",
    )
    suffix = post.rag_query_suffix()
    assert "暴雨内涝" in suffix
    assert "微信朋友圈" in suffix
    print("  NewsPost.rag_query_suffix: OK")


def test_from_metadata() -> None:
    post = NewsPost.from_metadata(METADATA, "fake_social_share_01")
    assert post.fake_type == "text_image_mismatch"
    assert "暴雨" in post.headline
    print("  NewsPost.from_metadata: OK")


def test_metadata_all_cases_have_text_fields() -> None:
    data = json.loads(METADATA.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    assert len(cases) == 18
    for case in cases:
        assert case.get("headline"), f"missing headline: {case['id']}"
        assert case.get("body"), f"missing body: {case['id']}"
        assert case.get("source"), f"missing source: {case['id']}"
        assert case.get("fake_type"), f"missing fake_type: {case['id']}"
    print("  metadata.json: all 18 cases have text fields")


def main() -> int:
    print("=== NewsPost Tests ===")
    test_news_post_has_text()
    test_rag_query_suffix()
    test_from_metadata()
    test_metadata_all_cases_have_text_fields()
    print("All NewsPost tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
