#!/usr/bin/env python3
"""Print suggested RAG keywords from weibo_test_subset.json (does not modify KB)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.weibo_loader import DEFAULT_SUBSET_PATH

DEFAULT_OUTPUT = ROOT / "data" / "weibo_kb_hints.json"

STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "你",
    "我",
    "他",
    "她",
    "它",
    "我们",
    "他们",
    "一个",
    "这个",
    "那个",
    "没有",
    "不是",
    "可以",
    "已经",
    "还是",
    "就是",
    "如果",
    "因为",
    "所以",
    "但是",
    "而且",
    "或者",
    "今天",
    "昨天",
    "近日",
    "微博",
    "http",
    "https",
    "网页链接",
    "阅读全文",
    "下载",
    "客户端",
}


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", text):
        word = chunk.strip()
        if word and word not in STOPWORDS:
            tokens.append(word)
    return tokens


def _top_keywords(cases: list[dict], *, limit: int = 12) -> list[str]:
    counter: Counter[str] = Counter()
    for case in cases:
        text = f"{case.get('headline', '')} {case.get('body', '')}"
        counter.update(_tokenize(text))
    return [word for word, _ in counter.most_common(limit)]


def build_hints(subset_path: Path) -> dict:
    data = json.loads(subset_path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    rumor = [c for c in cases if c.get("expected_verdict") == "fake"]
    nonrumor = [c for c in cases if c.get("expected_verdict") == "authentic"]
    return {
        "source": str(subset_path),
        "rumor_cases": len(rumor),
        "nonrumor_cases": len(nonrumor),
        "suggested_support_keywords": _top_keywords(nonrumor),
        "suggested_warning_keywords": _top_keywords(rumor),
        "case_ids": {
            "rumor": [c.get("id") for c in rumor],
            "nonrumor": [c.get("id") for c in nonrumor],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build weibo KB keyword hints")
    parser.add_argument(
        "--subset",
        default=str(DEFAULT_SUBSET_PATH),
        help="Path to weibo_test_subset.json",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Write hints JSON here (use - for stdout only)",
    )
    args = parser.parse_args()

    subset_path = Path(args.subset)
    if not subset_path.is_file():
        print(f"Error: subset not found: {subset_path}", file=sys.stderr)
        return 1

    hints = build_hints(subset_path)
    payload = json.dumps(hints, ensure_ascii=False, indent=2)

    if args.output != "-":
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
