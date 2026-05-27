"""Mock RAG server: keyword-based anchor retrieval from local JSON knowledge base."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Mock RAG Knowledge Base", version="0.1.0")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_KB = _REPO_ROOT / "data" / "knowledge_base.json"
_KB_PATH = Path(os.environ.get("RAG_KB_PATH", str(_DEFAULT_KB)))

_kb_cache: list[dict[str, Any]] | None = None

MIN_SUPPORT_SCORE = float(os.environ.get("RAG_MIN_SUPPORT_SCORE", "1.0"))
MIN_WARNING_SCORE = float(os.environ.get("RAG_MIN_WARNING_SCORE", "2.0"))


def _load_kb() -> list[dict[str, Any]]:
    global _kb_cache
    if _kb_cache is not None:
        return _kb_cache
    if not _KB_PATH.is_file():
        raise FileNotFoundError(f"Knowledge base not found: {_KB_PATH}")
    data = json.loads(_KB_PATH.read_text(encoding="utf-8"))
    anchors = []
    for anchor in data.get("anchors", []):
        item = dict(anchor)
        item.setdefault("anchor_type", "support")
        anchors.append(item)
    _kb_cache = anchors
    return _kb_cache


def _tokenize(text: str) -> set[str]:
    text = text.lower()
    tokens = set(re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text))
    return {t for t in tokens if len(t) >= 2}


def _score_anchor(query: str, anchor: dict[str, Any]) -> float:
    query_lower = query.lower()
    score = 0.0

    keywords = [str(k).lower() for k in anchor.get("keywords", [])]
    for kw in keywords:
        if kw and kw in query_lower:
            score += 1.0

    q_tokens = _tokenize(query)
    anchor_text = " ".join(
        [
            anchor.get("title", ""),
            anchor.get("summary", ""),
            " ".join(anchor.get("keywords", [])),
        ]
    )
    a_tokens = _tokenize(anchor_text)
    if q_tokens and a_tokens:
        overlap = q_tokens & a_tokens
        score += len(overlap) * 0.5

    return score


def _score_all(query: str, kb: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for anchor in kb:
        s = _score_anchor(query, anchor)
        if s > 0:
            item = dict(anchor)
            item["score"] = round(s, 4)
            scored.append(item)
    return scored


def select_anchors(
    scored: list[dict[str, Any]],
    top_k: int = 3,
    *,
    min_support_score: float = MIN_SUPPORT_SCORE,
    min_warning_score: float = MIN_WARNING_SCORE,
) -> list[dict[str, Any]]:
    """Prefer strong support anchors; prioritize warning when it beats weak support."""
    support = [a for a in scored if a.get("anchor_type", "support") != "warning"]
    warning = [a for a in scored if a.get("anchor_type") == "warning"]

    support.sort(key=lambda x: x["score"], reverse=True)
    warning.sort(key=lambda x: x["score"], reverse=True)

    best_support = support[0]["score"] if support else 0.0
    best_warning = warning[0]["score"] if warning else 0.0

    if (
        warning
        and best_warning >= min_warning_score
        and best_warning > best_support
    ):
        result = [warning[0]]
        if support and len(result) < top_k:
            result.append(support[0])
        return result[:top_k]

    strong = [a for a in support if a["score"] > min_support_score]
    if strong:
        return strong[:top_k]

    result: list[dict[str, Any]] = []
    if warning and best_warning >= min_warning_score:
        result.append(warning[0])
    for anchor in support:
        if len(result) >= top_k:
            break
        if anchor not in result:
            result.append(anchor)
    return result[:top_k]


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=10)


class SearchResponse(BaseModel):
    anchors: list[dict[str, Any]]
    query: str


@app.get("/health")
def health() -> dict[str, str]:
    _load_kb()
    return {"status": "ok", "kb_path": str(_KB_PATH)}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    kb = _load_kb()
    scored = _score_all(req.query, kb)
    top = select_anchors(scored, req.top_k)
    return SearchResponse(anchors=top, query=req.query)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("RAG_PORT", "8002"))
    uvicorn.run(app, host="127.0.0.1", port=port)
