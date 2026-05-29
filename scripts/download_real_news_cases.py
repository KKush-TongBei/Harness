#!/usr/bin/env python3
"""Download curated real-news test images into data/test_cases/real_news/."""

from __future__ import annotations

import json
import re
import ssl
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "test_cases" / "real_news"
METADATA_PATH = ROOT / "data" / "test_cases" / "metadata.json"
SYNTHETIC_IMAGES = {
    "military_demo_01.jpg",
    "naval_parade_01.jpg",
    "historical_war_01.jpg",
    "press_conference_01.jpg",
    "disaster_relief_01.jpg",
    "sports_event_01.jpg",
    "medical_public_01.jpg",
    "social_compress_01.jpg",
}

# Verified downloadable URLs (Unsplash / Picsum / news og:image).
CASES: list[dict[str, str]] = [
    {
        "id": "real_press_briefing_01",
        "filename": "real_press_briefing_01.jpg",
        "image_url": "https://images.unsplash.com/photo-1556761175-b413da4baf72?w=960&q=80",
        "expected_verdict": "authentic",
        "category": "新闻发布会",
        "notes": "媒体发布会现场真实摄影",
        "source": "Unsplash",
    },
    {
        "id": "real_hospital_01",
        "filename": "real_hospital_01.jpg",
        "image_url": "https://images.unsplash.com/photo-1579684385127-1ef15d508118?w=960&q=80",
        "expected_verdict": "authentic",
        "category": "医疗新闻",
        "notes": "医院/医护场景真实摄影",
        "source": "Unsplash / NCI",
    },
    {
        "id": "real_sports_stadium_01",
        "filename": "real_sports_stadium_01.jpg",
        "image_url": "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=960&q=80",
        "expected_verdict": "authentic",
        "category": "体育新闻",
        "notes": "田径/体育赛事现场真实摄影",
        "source": "Unsplash",
    },
    {
        "id": "real_protest_news_01",
        "filename": "real_protest_news_01.jpg",
        "image_url": "https://images.unsplash.com/photo-1529107386315-e1a2ed48a620?w=960&q=80",
        "expected_verdict": "authentic",
        "category": "社会新闻",
        "notes": "街头集会/抗议报道常见场景",
        "source": "Unsplash",
    },
    {
        "id": "real_tv_studio_01",
        "filename": "real_tv_studio_01.jpg",
        "image_url": "https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=960&q=80",
        "expected_verdict": "authentic",
        "category": "电视新闻",
        "notes": "电视演播室/新闻工作场景",
        "source": "Unsplash",
    },
    {
        "id": "real_crowd_event_01",
        "filename": "real_crowd_event_01.jpg",
        "image_url": "https://images.unsplash.com/photo-1520607162513-77705c0f0d4a?w=960&q=80",
        "expected_verdict": "authentic",
        "category": "活动报道",
        "notes": "大型活动现场真实摄影",
        "source": "Unsplash",
    },
    {
        "id": "real_city_press_01",
        "filename": "real_city_press_01.jpg",
        "image_url": "https://picsum.photos/id/1015/960/640",
        "expected_verdict": "authentic",
        "category": "城市新闻",
        "notes": "城市景观真实摄影，常见新闻配图背景",
        "source": "Picsum Photos",
    },
    {
        "id": "fake_misleading_headline_01",
        "filename": "fake_misleading_headline_01.jpg",
        "image_url": "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=960&q=80",
        "expected_verdict": "fake",
        "category": "误导性信息",
        "notes": "信息过载/屏幕场景，模拟谣言传播语境",
        "source": "Unsplash",
    },
    {
        "id": "fake_gossipcop_variety_01",
        "filename": "fake_gossipcop_variety_01.jpg",
        "news_url": "https://variety.com/2017/biz/news/tax-march-donald-trump-protest-1202031487/",
        "expected_verdict": "fake",
        "category": "GossipCop 假新闻",
        "notes": "GossipCop 标注 fake 的 Variety 报道配图",
        "source": "GossipCop / Variety",
    },
    {
        "id": "fake_social_share_01",
        "filename": "fake_social_share_01.jpg",
        "image_url": "https://images.unsplash.com/photo-1611605698335-8b1569810432?w=960&q=80",
        "expected_verdict": "fake",
        "category": "社交媒体谣言",
        "notes": "社交媒体/手机分享场景，常见谣言传播载体",
        "source": "Unsplash",
    },
]

REAL_NEWS_TEXT: dict[str, dict[str, str]] = {
    "real_press_briefing_01": {
        "fake_type": "matching",
        "headline": "企业高管在媒体发布会上介绍新产品",
        "body": "发布会现场记者举机拍摄，演讲者在台前介绍年度业务计划。",
        "source": "财经通讯社",
    },
    "real_hospital_01": {
        "fake_type": "matching",
        "headline": "医院开展日常诊疗与护理工作",
        "body": "纪实摄影记录医护人员在病房内照料患者，场景为常规医疗工作。",
        "source": "健康时报",
    },
    "real_sports_stadium_01": {
        "fake_type": "matching",
        "headline": "田径选手在体育场参加赛事",
        "body": "现场摄影捕捉运动员在跑道与看台前热身准备，为常规体育赛事报道。",
        "source": "体育周报",
    },
    "real_protest_news_01": {
        "fake_type": "matching",
        "headline": "市民走上街头表达诉求，记者现场报道",
        "body": "通讯社记者拍摄集会现场，人群举牌游行，为合法新闻报道场景。",
        "source": "路透社",
    },
    "real_tv_studio_01": {
        "fake_type": "matching",
        "headline": "电视台新闻演播室工作场景",
        "body": "导播间与演播室灯光、摄像机就位，工作人员准备晚间新闻播出。",
        "source": "央视新闻",
    },
    "real_crowd_event_01": {
        "fake_type": "matching",
        "headline": "大型公共活动现场人群聚集",
        "body": "记者拍摄节日活动现场，观众密集参与互动，为常规活动报道配图。",
        "source": "新华社",
    },
    "real_city_press_01": {
        "fake_type": "matching",
        "headline": "城市天际线与街景成为经济报道配图",
        "body": "摄影记者拍摄城市建筑群与街景，用于城市发展专题报道。",
        "source": "新华社",
    },
    "fake_misleading_headline_01": {
        "fake_type": "new_text_old_image",
        "headline": "突发！本市今夜将发生特大事故，政府紧急封锁消息",
        "body": "刚刚得到内部消息，现场已被封锁，速转！配图仅为信息过载的屏幕截图，与标题声称事件无关。",
        "source": "未知微信群",
    },
    "fake_gossipcop_variety_01": {
        "fake_type": "misleading_text",
        "headline": "独家！某一线明星秘密结婚，内部人士曝光细节",
        "body": "未经证实的娱乐八卦标题，配图仅为普通活动或杂志风格照片，无法支撑爆料内容。",
        "source": "八卦自媒体",
    },
    "fake_social_share_01": {
        "fake_type": "text_image_mismatch",
        "headline": "暴雨导致某体育场严重内涝，观众被困",
        "body": "朋友圈疯传：现场惨不忍睹！但配图实际为手机社交分享界面或无关场景，与暴雨灾害描述不符。",
        "source": "微信朋友圈",
    },
}

_OG_IMAGE_RES = (
    re.compile(
        r'<meta[^>]+property=["\'](?:og:image|twitter:image)["\'][^>]+content=["\']([^"\']+)["\']',
        re.IGNORECASE,
    ),
    re.compile(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\'](?:og:image|twitter:image)["\']',
        re.IGNORECASE,
    ),
)


def _fetch_bytes(url: str, timeout: int = 90) -> bytes:
    ctx = ssl.create_default_context()
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Harness test downloader/1.0)",
            "Accept": "image/*,*/*",
        },
    )
    with urlopen(req, timeout=timeout, context=ctx) as resp:
        data = resp.read()
    if len(data) < 2048:
        raise ValueError(f"response too small ({len(data)} bytes)")
    return data


def _extract_og_image(html: str) -> str | None:
    for pattern in _OG_IMAGE_RES:
        match = pattern.search(html)
        if match:
            return match.group(1).strip()
    return None


def _resolve_image_url(case: dict[str, str]) -> str:
    if case.get("image_url"):
        return case["image_url"]
    news_url = case.get("news_url")
    if not news_url:
        raise ValueError(f"No image_url or news_url for case {case['id']}")
    html = _fetch_bytes(news_url).decode("utf-8", errors="ignore")
    og = _extract_og_image(html)
    if not og:
        raise ValueError(f"Could not extract og:image from {news_url}")
    if og.startswith("//"):
        return "https:" + og
    if og.startswith("/"):
        return urljoin(news_url, og)
    return og


def _download_case(case: dict[str, str], retries: int = 3) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / case["filename"]
    url = _resolve_image_url(case)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"  [{attempt}/{retries}] {case['id']} <- {url[:100]}...")
            data = _fetch_bytes(url)
            out_path.write_bytes(data)
            print(f"  saved {out_path.name} ({len(data) // 1024} KB)")
            return out_path
        except (URLError, ValueError, OSError) as exc:
            last_err = exc
            time.sleep(2 * attempt)
    raise last_err or RuntimeError(f"download failed for {case['id']}")


def _build_metadata_entry(case: dict[str, str]) -> dict[str, str]:
    text = REAL_NEWS_TEXT.get(case["id"], {})
    return {
        "id": case["id"],
        "image": f"real_news/{case['filename']}",
        "expected_verdict": case["expected_verdict"],
        "category": case["category"],
        "fake_type": text.get("fake_type", "matching"),
        "headline": text.get("headline", ""),
        "body": text.get("body", ""),
        "source": text.get("source", case.get("source", "")),
        "notes": f"{case['notes']} [{case['source']}]",
    }


def _merge_metadata(real_entries: list[dict[str, str]]) -> None:
    if METADATA_PATH.is_file():
        data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    else:
        data = {"cases": []}

    synthetic = [
        c for c in data.get("cases", []) if Path(c.get("image", "")).name in SYNTHETIC_IMAGES
    ]
    data["cases"] = synthetic + real_entries
    METADATA_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Updated {METADATA_PATH} ({len(synthetic)} synthetic + {len(real_entries)} real)")


def main() -> int:
    print(f"Downloading {len(CASES)} real-news test images to {OUT_DIR}")
    entries: list[dict[str, str]] = []
    failed: list[str] = []

    for case in CASES:
        try:
            _download_case(case)
            entries.append(_build_metadata_entry(case))
        except (URLError, ValueError, OSError) as exc:
            print(f"  [FAIL] {case['id']}: {exc}", file=sys.stderr)
            failed.append(case["id"])

    if len(entries) < 8:
        print(f"Too few images downloaded ({len(entries)}).", file=sys.stderr)
        return 1

    _merge_metadata(entries)

    if failed:
        print(f"Warning: failed cases: {', '.join(failed)}", file=sys.stderr)
    print(f"Done. {len(entries)}/{len(CASES)} images ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
