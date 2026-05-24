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


def _load_kb() -> list[dict[str, Any]]:
    global _kb_cache
    if _kb_cache is not None:
        return _kb_cache
    if not _KB_PATH.is_file():
        raise FileNotFoundError(f"Knowledge base not found: {_KB_PATH}")
    data = json.loads(_KB_PATH.read_text(encoding="utf-8"))
    _kb_cache = list(data.get("anchors", []))
    return _kb_cache


def _tokenize(text: str) -> set[str]:
    text = text.lower()
    tokens = set(re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text))
    return {t for t in tokens if len(t) >= 2}


def _score_anchor(query: str, anchor: dict[str, Any]) -> float:
    query_lower = query.lower()
    score = 0.0

    # Substring keyword match (works well for Chinese without word boundaries)
    keywords = [str(k).lower() for k in anchor.get("keywords", [])]
    for kw in keywords:
        if kw and kw in query_lower:
            score += 1.0

    # Token overlap for English / mixed text
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
    scored = []
    for anchor in kb:
        s = _score_anchor(req.query, anchor)
        if s > 0:
            item = dict(anchor)
            item["score"] = round(s, 4)
            scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[: req.top_k]
    return SearchResponse(anchors=top, query=req.query)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("RAG_PORT", "8002"))
    uvicorn.run(app, host="127.0.0.1", port=port)
