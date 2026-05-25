#!/usr/bin/env python3
"""A/B test: Baseline vs Harness+RAG on test case suite."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.evaluation import verdict_matches_expected
from harness.loop import ExecutionLoop
from harness.state import StateStorage
from harness.tools.registry import ToolRegistry


async def run_case(
    loop: ExecutionLoop,
    case: dict,
    cases_dir: Path,
) -> dict:
    case_id = case["id"]
    image_path = cases_dir / case["image"]
    expected = case.get("expected_verdict", "authentic")

    baseline_state = await loop.run_baseline(str(image_path), save=False)
    baseline_state.run_id = f"baseline_{case_id}"
    StateStorage(ROOT / "outputs" / "ab_test").save(baseline_state)

    harness_state = await loop.run_harness(str(image_path), save=False)
    harness_state.run_id = f"harness_{case_id}"
    StateStorage(ROOT / "outputs" / "ab_test").save(harness_state)

    baseline_correct = verdict_matches_expected(baseline_state.verdict, expected)
    harness_correct = verdict_matches_expected(harness_state.verdict, expected)
    fp_corrected = (
        not baseline_correct
        and baseline_state.verdict in ("fake", "suspicious")
        and harness_correct
        and expected == "authentic"
    )

    return {
        "case_id": case_id,
        "category": case.get("category", ""),
        "expected": expected,
        "baseline_verdict": baseline_state.verdict,
        "baseline_confidence": baseline_state.confidence,
        "harness_verdict": harness_state.verdict,
        "harness_confidence": harness_state.confidence,
        "trufor_score": harness_state.forgery_score,
        "anchors_count": len(harness_state.anchors),
        "baseline_correct": baseline_correct,
        "harness_correct": harness_correct,
        "fp_corrected": fp_corrected,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Harness A/B ablation test")
    parser.add_argument(
        "--cases",
        default=str(ROOT / "data" / "test_cases" / "metadata.json"),
        help="Path to test cases metadata.json",
    )
    parser.add_argument("--qwen-url", default="http://127.0.0.1:8000")
    parser.add_argument("--trufor-url", default="http://127.0.0.1:8001")
    parser.add_argument("--rag-url", default="http://127.0.0.1:8002")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="Max cases (0=all)")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.is_file():
        print(f"Error: cases file not found: {cases_path}", file=sys.stderr)
        print("Run: python scripts/generate_test_cases.py", file=sys.stderr)
        return 1

    cases_dir = cases_path.parent
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    if args.limit > 0:
        cases = cases[: args.limit]

    registry = ToolRegistry(
        qwen_base=args.qwen_url,
        trufor_base=args.trufor_url,
        rag_base=args.rag_url,
    )
    loop = ExecutionLoop(registry=registry, max_new_tokens=args.max_new_tokens)

    health = await registry.health_check()
    if not health.get("all_ok"):
        print(f"Services not ready: {json.dumps(health, ensure_ascii=False)}", file=sys.stderr)
        print("Run: bash scripts/start_services.sh", file=sys.stderr)
        return 1

    out_dir = ROOT / "outputs" / "ab_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"

    rows: list[dict] = []
    for case in cases:
        print(f"Running case: {case['id']} ...")
        try:
            row = await run_case(loop, case, cases_dir)
            rows.append(row)
            print(
                f"  baseline={row['baseline_verdict']} harness={row['harness_verdict']} "
                f"expected={row['expected']} fp_corrected={row['fp_corrected']}"
            )
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            rows.append({"case_id": case["id"], "error": str(e)})

    if rows:
        fieldnames = list(rows[0].keys())
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    baseline_ok = sum(1 for r in rows if r.get("baseline_correct"))
    harness_ok = sum(1 for r in rows if r.get("harness_correct"))
    fp_fixed = sum(1 for r in rows if r.get("fp_corrected"))

    print()
    print(f"Summary written to: {summary_path}")
    print(f"Baseline correct: {baseline_ok}/{len(rows)}")
    print(f"harness correct: {harness_ok}/{len(rows)}")
    print(f"false positives corrected by harness: {fp_fixed}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
