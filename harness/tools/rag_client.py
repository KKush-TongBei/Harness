from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class Anchor:
    title: str
    source: str
    date: str
    summary: str
    keywords: list[str]
    score: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Anchor":
        return cls(
            title=str(data.get("title", "")),
            source=str(data.get("source", "")),
            date=str(data.get("date", "")),
            summary=str(data.get("summary", "")),
            keywords=[str(k) for k in data.get("keywords", [])],
            score=float(data.get("score", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "source": self.source,
            "date": self.date,
            "summary": self.summary,
            "keywords": self.keywords,
            "score": self.score,
        }


class RagClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8002", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    async def search(self, query: str, top_k: int = 3) -> list[Anchor]:
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            resp = await client.post(
                f"{self.base_url}/search",
                json={"query": query, "top_k": top_k},
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"RAG search failed ({resp.status_code}): {resp.text}")
            data = resp.json()
            anchors = data.get("anchors", [])
            return [Anchor.from_dict(a) for a in anchors]
