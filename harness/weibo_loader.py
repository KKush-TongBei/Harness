from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = ROOT / "weibo_dataset"
DEFAULT_SUBSET_PATH = ROOT / "data" / "weibo_test_subset.json"



def _image_filename_from_url(url: str) -> str:
    path = urlparse(url.strip()).path
    return Path(path).name


def _resolve_local_image(urls_line: str, image_dir: Path) -> Path | None:
    for part in urls_line.split("|"):
        part = part.strip()
        if not part or part.lower() == "null":
            continue
        if part.startswith("http://") or part.startswith("https://"):
            candidate = image_dir / _image_filename_from_url(part)
        else:
            candidate = image_dir / Path(part).name
        if candidate.is_file():
            return candidate.resolve()
    return None


def _parse_meta_line(line: str) -> dict[str, str]:
    parts = line.split("|")
    weibo_id = parts[0].strip() if parts else ""
    username = parts[1].strip() if len(parts) > 1 else ""
    if username.lower() == "null":
        username = ""
    return {"weibo_id": weibo_id, "username": username}


def iter_weibo_records(
    tweet_file: Path,
    image_dir: Path,
    *,
    require_local_image: bool = True,
) -> Iterator[dict[str, Any]]:
    lines = tweet_file.read_text(encoding="utf-8").splitlines()
    i = 0
    while i + 2 < len(lines):
        meta_line = lines[i].strip()
        urls_line = lines[i + 1].strip()
        text_line = lines[i + 2].strip()
        i += 3

        if not meta_line:
            continue

        meta = _parse_meta_line(meta_line)
        local_image = _resolve_local_image(urls_line, image_dir)
        if require_local_image and local_image is None:
            continue

        yield {
            "weibo_id": meta["weibo_id"],
            "username": meta["username"],
            "text": text_line,
            "image_path": str(local_image) if local_image else "",
            "tweet_file": tweet_file.name,
        }


def _headline_from_text(text: str, max_len: int = 80) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return "微博帖子"
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"


def record_to_case(
    record: dict[str, Any],
    *,
    label: str,
    index: int,
) -> dict[str, Any]:
    is_rumor = label == "rumor"
    weibo_id = record.get("weibo_id") or f"idx{index}"
    text = str(record.get("text", ""))
    username = str(record.get("username", ""))
    image_path = Path(str(record.get("image_path", "")))
    rel_image = image_path.relative_to(ROOT).as_posix() if image_path.is_absolute() else str(record.get("image_path", ""))

    return {
        "id": f"weibo_{label}_{weibo_id}",
        "image": rel_image,
        "expected_verdict": "fake" if is_rumor else "authentic",
        "category": "微博谣言" if is_rumor else "微博非谣言",
        "fake_type": "misleading_text" if is_rumor else "matching",
        "headline": _headline_from_text(text),
        "body": text,
        "source": username or "微博",
        "notes": f"Weibo test split ({label}); id={weibo_id}",
        "dataset": "weibo",
        "weibo_id": weibo_id,
    }


def load_weibo_cases(
    dataset_dir: Path | None = None,
    *,
    split: str = "test",
    limit_per_class: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    dataset_dir = (dataset_dir or DEFAULT_DATASET_DIR).resolve()
    tweets_dir = dataset_dir / "tweets"
    specs = {
        "test": [
            ("rumor", tweets_dir / "test_rumor.txt", dataset_dir / "rumor_images"),
            ("nonrumor", tweets_dir / "test_nonrumor.txt", dataset_dir / "nonrumor_images"),
        ],
        "train": [
            ("rumor", tweets_dir / "train_rumor.txt", dataset_dir / "rumor_images"),
            ("nonrumor", tweets_dir / "train_nonrumor.txt", dataset_dir / "nonrumor_images"),
        ],
    }
    if split not in specs:
        raise ValueError(f"Unsupported split: {split}")

    cases: list[dict[str, Any]] = []
    for label, tweet_file, image_dir in specs[split]:
        records = list(
            iter_weibo_records(tweet_file, image_dir, require_local_image=True)
        )
        if limit_per_class > 0 and len(records) > limit_per_class:
            import random

            rng = random.Random(seed)
            records = rng.sample(records, limit_per_class)
        for idx, record in enumerate(records):
            cases.append(record_to_case(record, label=label, index=idx))
    return cases


def build_subset_metadata(
    dataset_dir: Path | None = None,
    *,
    split: str = "test",
    limit_per_class: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    cases = load_weibo_cases(
        dataset_dir,
        split=split,
        limit_per_class=limit_per_class,
        seed=seed,
    )
    return {
        "dataset": "weibo",
        "split": split,
        "limit_per_class": limit_per_class,
        "seed": seed,
        "cases": cases,
    }


def write_subset_metadata(
    output_path: Path | None = None,
    dataset_dir: Path | None = None,
    *,
    split: str = "test",
    limit_per_class: int = 10,
    seed: int = 42,
) -> Path:
    output_path = (output_path or DEFAULT_SUBSET_PATH).resolve()
    metadata = build_subset_metadata(
        dataset_dir,
        split=split,
        limit_per_class=limit_per_class,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def resolve_case_image(case: dict[str, Any], metadata_path: Path) -> Path:
    image = case.get("image", "")
    image_path = Path(image)
    if image_path.is_absolute():
        return image_path

    metadata_dir = metadata_path.parent.resolve()
    candidate = (metadata_dir / image_path).resolve()
    if candidate.is_file():
        return candidate

    root_candidate = (ROOT / image_path).resolve()
    if root_candidate.is_file():
        return root_candidate

    raise FileNotFoundError(f"Image not found for case {case.get('id')}: {image}")
