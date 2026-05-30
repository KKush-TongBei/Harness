#!/usr/bin/env python3
"""Offline tests for Weibo dataset loader and subset metadata."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.news_post import NewsPost
from harness.weibo_loader import (
    DEFAULT_DATASET_DIR,
    DEFAULT_SUBSET_PATH,
    load_weibo_cases,
    resolve_case_image,
)


def test_load_weibo_cases() -> None:
    cases = load_weibo_cases(DEFAULT_DATASET_DIR, split="test", limit_per_class=2, seed=42)
    assert len(cases) == 4
    assert sum(1 for c in cases if c["expected_verdict"] == "fake") == 2
    assert sum(1 for c in cases if c["expected_verdict"] == "authentic") == 2
    print("  load_weibo_cases: OK")


def test_subset_metadata_file() -> None:
    assert DEFAULT_SUBSET_PATH.is_file(), "Run scripts/build_weibo_subset.py first"
    data = json.loads(DEFAULT_SUBSET_PATH.read_text(encoding="utf-8"))
    cases = data["cases"]
    assert len(cases) == 20
    for case in cases:
        image = resolve_case_image(case, DEFAULT_SUBSET_PATH)
        assert image.is_file(), f"missing image for {case['id']}"
    print("  weibo_test_subset.json: OK")


def test_news_post_from_weibo_case() -> None:
    data = json.loads(DEFAULT_SUBSET_PATH.read_text(encoding="utf-8"))
    case = data["cases"][0]
    post = NewsPost.from_case_dict(case, DEFAULT_SUBSET_PATH.parent, metadata_path=DEFAULT_SUBSET_PATH)
    assert post.has_text()
    assert Path(post.image_path).is_file()
    print("  NewsPost.from_case_dict(weibo): OK")


def main() -> None:
    if not DEFAULT_DATASET_DIR.is_dir():
        print("Skip: weibo_dataset/ not present locally")
        return
    test_load_weibo_cases()
    test_subset_metadata_file()
    test_news_post_from_weibo_case()
    print("All weibo loader tests passed.")


if __name__ == "__main__":
    main()
