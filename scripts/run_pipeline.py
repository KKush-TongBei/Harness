#!/usr/bin/env python3
"""Run full Harness pipeline on a single image."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.loop import ExecutionLoop
from harness.tools.registry import ToolRegistry


async def main() -> int:
    parser = argparse.ArgumentParser(description="Harness MVP single-image pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--qwen-url", default="http://127.0.0.1:8000")
    parser.add_argument("--trufor-url", default="http://127.0.0.1:8001")
    parser.add_argument("--rag-url", default="http://127.0.0.1:8002")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--baseline", action="store_true", help="Run baseline mode only")
    args = parser.parse_args()

    image = Path(args.image).resolve()
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
            state = await loop.run_baseline(str(image))
        else:
            state = await loop.run_harness(str(image))
    except RuntimeError as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        return 1

    report = {
        "run_id": state.run_id,
        "mode": state.mode,
        "verdict": state.verdict,
        "confidence": state.confidence,
        "reasoning": state.reasoning,
        "evidence_chain": state.evidence_chain,
        "forgery_score": state.forgery_score,
        "anchors_count": len(state.anchors),
        "errors": state.errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
