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
        "fake_type": "matching",
        "headline": "某军区举行公开日，主战坦克与装甲部队向民众展示",
        "body": "官方媒体报道称，此次开放日活动为年度例行公开军事展示，现场拍摄合法合规。",
        "source": "新华社",
        "notes": "模拟公开日坦克展示场景，TruFor 可能因压缩误报",
        "bg": (60, 80, 50),
        "text": "Military Open Day\nTank Display\nOfficial Press Photo",
    },
    {
        "id": "naval_parade_01",
        "image": "naval_parade_01.jpg",
        "expected_verdict": "authentic",
        "category": "海军舰艇开放日",
        "fake_type": "matching",
        "headline": "海军成立纪念日舰艇开放活动纪实",
        "body": "多艘军舰靠港向公众开放参观，水兵列队欢迎市民登舰。",
        "source": "人民日报",
        "notes": "军舰与军港场景",
        "bg": (30, 60, 100),
        "text": "Naval Parade\nWarship Open Day\nAuthorized Photo",
    },
    {
        "id": "historical_war_01",
        "image": "historical_war_01.jpg",
        "expected_verdict": "authentic",
        "category": "历史战争纪实摄影",
        "fake_type": "matching",
        "headline": "档案馆公布二战时期战场纪实照片",
        "body": "国家档案馆发布一批历史战争纪实影像，胶片颗粒与扫描痕迹为原始档案特征。",
        "source": "国家档案馆",
        "notes": "低分辨率胶片颗粒模拟",
        "bg": (90, 90, 90),
        "text": "Historical Archive\nWWII Documentary\nFilm Grain Photo",
    },
    {
        "id": "press_conference_01",
        "image": "press_conference_01.jpg",
        "expected_verdict": "authentic",
        "category": "政府新闻发布会",
        "fake_type": "matching",
        "headline": "国务院新闻办公室举行例行新闻发布会",
        "body": "发言人在 podium 前介绍政策，媒体席记者现场提问报道。",
        "source": "国务院新闻办公室",
        "notes": "官方发布会标准构图",
        "bg": (40, 40, 80),
        "text": "Press Conference\nGovernment Official\nState Media Photo",
    },
    {
        "id": "disaster_relief_01",
        "image": "disaster_relief_01.jpg",
        "expected_verdict": "authentic",
        "category": "灾害救援新闻",
        "fake_type": "matching",
        "headline": "洪涝灾害救援现场：救援队伍转移受灾群众",
        "body": "记者在现场拍摄救援进展，画面显示临时安置点与志愿者协助。",
        "source": "央视新闻",
        "notes": "救援现场记者拍摄",
        "bg": (70, 90, 120),
        "text": "Disaster Relief\nRescue Operation\nNews Report Photo",
    },
    {
        "id": "sports_event_01",
        "image": "sports_event_01.jpg",
        "expected_verdict": "authentic",
        "category": "体育赛事转播",
        "fake_type": "matching",
        "headline": "国际体育赛事现场转播画面",
        "body": "电视转播截图显示运动员在场馆比赛，画面含运动模糊与压缩痕迹。",
        "source": "央视体育",
        "notes": "JPEG 压缩与运动场景",
        "bg": (20, 120, 60),
        "text": "Sports Event\nOlympic Stadium\nLive Broadcast",
    },
    {
        "id": "medical_public_01",
        "image": "medical_public_01.jpg",
        "expected_verdict": "authentic",
        "category": "公共卫生科普",
        "fake_type": "matching",
        "headline": "国家卫健委发布公共卫生科普宣传配图",
        "body": "配图展示医护人员开展健康科普与疫苗接种宣传。",
        "source": "国家卫健委",
        "notes": "官方健康宣传配图",
        "bg": (200, 230, 240),
        "text": "Public Health\nMedical Staff\nOfficial Campaign",
    },
    {
        "id": "social_compress_01",
        "image": "social_compress_01.jpg",
        "expected_verdict": "authentic",
        "category": "社交媒体压缩图",
        "fake_type": "matching",
        "headline": "社交平台多次转发的活动照片",
        "body": "网友转发分享的活动现场图，画质因多次压缩出现块效应，属正常传播现象。",
        "source": "微博用户分享",
        "notes": "多次转发压缩痕迹",
        "bg": (180, 160, 140),
        "text": "Social Media\nRe-shared Image\nHeavy Compression",
    },
]


def _make_image(case: dict) -> Image.Image:
    w, h = 640, 480
    img = Image.new("RGB", (w, h), case["bg"])
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 200, 560, 420], outline=(255, 255, 255), width=3)
    draw.ellipse([250, 80, 390, 180], fill=(200, 200, 200))

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    for i, line in enumerate(case["text"].split("\n")):
        draw.text((100, 30 + i * 30), line, fill=(255, 255, 255), font=font)

    return img


def _metadata_entry(case: dict) -> dict:
    return {
        "id": case["id"],
        "image": case["image"],
        "expected_verdict": case["expected_verdict"],
        "category": case["category"],
        "fake_type": case["fake_type"],
        "headline": case["headline"],
        "body": case["body"],
        "source": case["source"],
        "notes": case["notes"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_cases = []

    for case in CASES:
        img = _make_image(case)
        out_path = OUT_DIR / case["image"]
        img.save(out_path, quality=75 if "compress" in case["id"] else 90)
        metadata_cases.append(_metadata_entry(case))
        print(f"Created {out_path}")

    meta_path = OUT_DIR / "metadata.json"
    if meta_path.is_file():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        real_cases = [
            c for c in data.get("cases", []) if c.get("image", "").startswith("real_news/")
        ]
        metadata_cases.extend(real_cases)

    metadata = {"cases": metadata_cases}
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {meta_path} ({len(metadata_cases)} cases)")


if __name__ == "__main__":
    main()
