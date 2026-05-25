from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.tools.qwen_client import QwenClient
from harness.tools.rag_client import Anchor, RagClient
from harness.tools.trufor_client import TruForClient


@dataclass
class ToolRegistry:
    qwen_base: str = "http://127.0.0.1:8000"
    trufor_base: str = "http://127.0.0.1:8001"
    rag_base: str = "http://127.0.0.1:8002"
    qwen_timeout: float = 1800.0
    trufor_timeout: float = 120.0
    rag_timeout: float = 30.0

    _qwen: QwenClient = field(init=False, repr=False)
    _trufor: TruForClient = field(init=False, repr=False)
    _rag: RagClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._qwen = QwenClient(self.qwen_base, timeout=self.qwen_timeout)
        self._trufor = TruForClient(self.trufor_base, timeout=self.trufor_timeout)
        self._rag = RagClient(self.rag_base, timeout=self.rag_timeout)

    async def describe_image(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        return await self._qwen.describe(image_path, prompt, max_new_tokens)

    async def forgery_score(self, image_path: str) -> float:
        return await self._trufor.score(image_path)

    async def search_anchors(self, query: str, top_k: int = 3) -> list[Anchor]:
        return await self._rag.search(query, top_k=top_k)

    async def health_check(self) -> dict[str, Any]:
        status: dict[str, Any] = {"qwen": False, "trufor": False, "rag": False, "details": {}}

        try:
            qwen_h = await self._qwen.health()
            status["qwen"] = qwen_h.get("status") == "ok"
            status["details"]["qwen"] = qwen_h
        except Exception as e:
            status["details"]["qwen"] = str(e)

        try:
            status["trufor"] = await self._trufor.health()
            status["details"]["trufor"] = "reachable" if status["trufor"] else "unreachable"
        except Exception as e:
            status["details"]["trufor"] = str(e)

        try:
            rag_h = await self._rag.health()
            status["rag"] = rag_h.get("status") == "ok"
            status["details"]["rag"] = rag_h
        except Exception as e:
            status["details"]["rag"] = str(e)

        status["all_ok"] = all([status["qwen"], status["trufor"], status["rag"]])
        return status
