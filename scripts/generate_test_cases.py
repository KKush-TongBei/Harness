#!/usr/bin/env python3
"""Generate synthetic test images for Harness MVP false-positive scenarios."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "test_cases"

CASES = [
    {
        "id": "military_demo_01",
        "image": "military_demo_01.jpg",
        "expected_verdict": "authentic",
        "category": "合法军事装备展示",
        "notes": "模拟公开日坦克展示场景，TruFor 可能因压缩误报",
        "bg": (60, 80, 50),
        "text": "Military Open Day\nTank Display\nOfficial Press Photo",
    },
    {
        "id": "naval_parade_01",
        "image": "naval_parade_01.jpg",
        "expected_verdict": "authentic",
        "category": "海军舰艇开放日",
        "notes": "军舰与军港场景",
        "bg": (30, 60, 100),
        "text": "Naval Parade\nWarship Open Day\nAuthorized Photo",
    },
    {
        "id": "historical_war_01",
        "image": "historical_war_01.jpg",
        "expected_verdict": "authentic",
        "category": "历史战争纪实摄影",
        "notes": "低分辨率胶片颗粒模拟",
        "bg": (90, 90, 90),
        "text": "Historical Archive\nWWII Documentary\nFilm Grain Photo",
    },
    {
        "id": "press_conference_01",
        "image": "press_conference_01.jpg",
        "expected_verdict": "authentic",
        "category": "政府新闻发布会",
        "notes": "官方发布会标准构图",
        "bg": (40, 40, 80),
        "text": "Press Conference\nGovernment Official\nState Media Photo",
    },
    {
        "id": "disaster_relief_01",
        "image": "disaster_relief_01.jpg",
        "expected_verdict": "authentic",
        "category": "灾害救援新闻",
        "notes": "救援现场记者拍摄",
        "bg": (70, 90, 120),
        "text": "Disaster Relief\nRescue Operation\nNews Report Photo",
    },
    {
        "id": "sports_event_01",
        "image": "sports_event_01.jpg",
        "expected_verdict": "authentic",
        "category": "体育赛事转播",
        "notes": "JPEG 压缩与运动场景",
        "bg": (20, 120, 60),
        "text": "Sports Event\nOlympic Stadium\nLive Broadcast",
    },
    {
        "id": "medical_public_01",
        "image": "medical_public_01.jpg",
        "expected_verdict": "authentic",
        "category": "公共卫生科普",
        "notes": "官方健康宣传配图",
        "bg": (200, 230, 240),
        "text": "Public Health\nMedical Staff\nOfficial Campaign",
    },
    {
        "id": "social_compress_01",
        "image": "social_compress_01.jpg",
        "expected_verdict": "authentic",
        "category": "社交媒体压缩图",
        "notes": "多次转发压缩痕迹",
        "bg": (180, 160, 140),
        "text": "Social Media\nRe-shared Image\nHeavy Compression",
    },
]


def _make_image(case: dict) -> Image.Image:
    w, h = 640, 480
    img = Image.new("RGB", (w, h), case["bg"])
    draw = ImageDraw.Draw(img)

    # Simple geometric "scene" elements
    draw.rectangle([80, 200, 560, 420], outline=(255, 255, 255), width=3)
    draw.ellipse([250, 80, 390, 180], fill=(200, 200, 200))

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    for i, line in enumerate(case["text"].split("\n")):
        draw.text((100, 30 + i * 30), line, fill=(255, 255, 255), font=font)

    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_cases = []

    for case in CASES:
        img = _make_image(case)
        out_path = OUT_DIR / case["image"]
        img.save(out_path, quality=75 if "compress" in case["id"] else 90)
        metadata_cases.append(
            {
                "id": case["id"],
                "image": case["image"],
                "expected_verdict": case["expected_verdict"],
                "category": case["category"],
                "notes": case["notes"],
            }
        )
        print(f"Created {out_path}")

    metadata = {"cases": metadata_cases}
    meta_path = OUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path} ({len(metadata_cases)} cases)")


if __name__ == "__main__":
    main()
