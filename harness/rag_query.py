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
)


def build_misinformation_probe(description: str) -> str | None:
    """Build supplemental RAG query when description mentions rumor-carrier cues."""
    if not description.strip():
        return None
    found = [term for term in RISK_TERMS if term.lower() in description.lower()]
    if not found:
        return None
    return " ".join(found) + " 谣言传播 未经核实 误导性信息"


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
