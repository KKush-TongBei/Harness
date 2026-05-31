from __future__ import annotations

from harness.tools.rag_client import Anchor

RISK_TERMS: tuple[str, ...] = (
    "手机",
    "屏幕",
    "截图",
    "八卦",
    "娱乐",
    "杂志",
    "聊天",
    "转发",
    "朋友圈",
    "微信",
    "显示器",
    "电脑屏幕",
    "明星",
    "小报",
    "移花接木",
    "旧图",
    "图文不符",
    "不匹配",
    "速转",
    "扩散",
    "求扩散",
    "突发",
    "去世",
    "爆炸",
    "人贩子",
    "phone",
    "mobile",
    "screen",
    "screenshot",
    "gossip",
    "celebrity",
    "headline",
    "misleading",
    "app",
    "message",
    "variety",
)

OFFICIAL_SUPPORT_TERMS: tuple[str, ...] = (
    "公安部",
    "刑侦局",
    "中新网",
    "央视新闻",
    "辟谣平台",
    "经视直播",
    "楚天都市报",
    "新华社",
    "人民日报",
    "国家卫健委",
    "国务院",
)


def _combined_text(description: str, news_text: str = "") -> str:
    return f"{description}\n{news_text}".strip()


def _contains_official_source(text: str) -> bool:
    return any(term in text for term in OFFICIAL_SUPPORT_TERMS)


def build_official_support_probe(description: str, news_text: str = "") -> str | None:
    """Build supplemental RAG query when text mentions official media or agencies."""
    combined = _combined_text(description, news_text)
    if not combined:
        return None
    found = [term for term in OFFICIAL_SUPPORT_TERMS if term in combined]
    if not found:
        return None
    return " ".join(found) + " 官方 正规媒体 微博转载 合法 新闻报道 反诈提醒"


def build_misinformation_probe(description: str, news_text: str = "") -> str | None:
    """Build supplemental RAG query when description or news text mentions rumor cues."""
    combined = _combined_text(description, news_text)
    if not combined:
        return None
    if _contains_official_source(combined):
        return None
    found = [term for term in RISK_TERMS if term.lower() in combined.lower()]
    if not found:
        return None
    return " ".join(found) + " 谣言传播 未经核实 误导性信息 图文不符 新文旧图"


def merge_anchors(primary: list[Anchor], extra: list[Anchor]) -> list[Anchor]:
    """Merge anchor lists by id, keeping the higher retrieval score."""
    by_key: dict[str, Anchor] = {}
    for anchor in primary + extra:
        key = anchor.id or anchor.title
        existing = by_key.get(key)
        if existing is None or anchor.score > existing.score:
            by_key[key] = anchor
    merged = list(by_key.values())
    merged.sort(key=lambda a: a.score, reverse=True)
    return merged
