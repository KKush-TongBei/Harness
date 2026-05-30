#!/usr/bin/env python3
"""Run full Harness pipeline on a single news post (image + optional text)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.loop import ExecutionLoop
from harness.news_post import NewsPost
from harness.tools.registry import ToolRegistry
from harness.weibo_loader import DEFAULT_SUBSET_PATH

DEFAULT_METADATA = ROOT / "data" / "test_cases" / "metadata.json"


async def main() -> int:
    parser = argparse.ArgumentParser(description="Harness MVP news post pipeline")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--case-id", help="Load image+text from metadata.json by case id")
    parser.add_argument("--cases", default=str(DEFAULT_METADATA), help="metadata.json path")
    parser.add_argument(
        "--weibo",
        action="store_true",
        help="Use data/weibo_test_subset.json (run scripts/build_weibo_subset.py first)",
    )
    parser.add_argument("--headline", default="", help="News headline")
    parser.add_argument("--body", default="", help="News body text")
    parser.add_argument("--source", default="", help="News source")
    parser.add_argument("--qwen-url", default="http://127.0.0.1:8000")
    parser.add_argument("--trufor-url", default="http://127.0.0.1:8001")
    parser.add_argument("--rag-url", default="http://127.0.0.1:8002")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--baseline", action="store_true", help="Run baseline mode only")
    args = parser.parse_args()

    if args.case_id:
        cases_path = Path(DEFAULT_SUBSET_PATH if args.weibo else args.cases)
        if not cases_path.is_file():
            print(f"Error: cases file not found: {cases_path}", file=sys.stderr)
            if args.weibo:
                print("Run: python scripts/build_weibo_subset.py", file=sys.stderr)
            return 1
        news = NewsPost.from_metadata(cases_path, args.case_id)
        image = Path(news.image_path)
    elif args.image:
        image = Path(args.image).resolve()
        news = NewsPost(
            image_path=str(image),
            headline=args.headline,
            body=args.body,
            source=args.source,
        )
    else:
        print("Error: provide --image or --case-id", file=sys.stderr)
        return 1

    if not image.is_file():
        print(f"Error: image not found: {image}", file=sys.stderr)
        return 1

    registry = ToolRegistry(
        qwen_base=args.qwen_url,
        trufor_base=args.trufor_url,
        rag_base=args.rag_url,
    )
    loop = ExecutionLoop(registry=registry, max_new_tokens=args.max_new_tokens)

    try:
        if args.baseline:
            state = await loop.run_baseline(str(image), news=news if news.has_text() else None)
        else:
            state = await loop.run_harness(str(image), news=news if news.has_text() else None)
    except RuntimeError as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        return 1

    report = {
        "run_id": state.run_id,
        "mode": state.mode,
        "verdict": state.verdict,
        "issue_type": state.issue_type,
        "confidence": state.confidence,
        "reasoning": state.reasoning,
        "evidence_chain": state.evidence_chain,
        "forgery_score": state.forgery_score,
        "anchors_count": len(state.anchors),
        "headline": state.headline,
        "body": state.body,
        "source": state.source,
        "fake_type": state.fake_type,
        "fusion_parse_ok": not any(
            "Fusion output JSON parse failed" in e for e in state.errors
        ),
        "raw_fusion_output": state.raw_fusion_output,
        "errors": state.errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
