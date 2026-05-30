#!/usr/bin/env python3
"""Build a small Harness-compatible metadata subset from local weibo_dataset/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.weibo_loader import DEFAULT_DATASET_DIR, DEFAULT_SUBSET_PATH, write_subset_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Build weibo test subset metadata for Harness")
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Path to weibo_dataset/ root",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SUBSET_PATH),
        help="Output metadata JSON path",
    )
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=10,
        help="Max rumor and nonrumor samples each (0 = all with local images)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f"Error: dataset dir not found: {dataset_dir}", file=sys.stderr)
        print("Place the Weibo dataset at weibo_dataset/ and retry.", file=sys.stderr)
        return 1

    output = write_subset_metadata(
        Path(args.output),
        dataset_dir,
        split=args.split,
        limit_per_class=args.limit_per_class,
        seed=args.seed,
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
