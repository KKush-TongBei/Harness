from __future__ import annotations

from pathlib import Path

import httpx


class TruForClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8001", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def health(self) -> bool:
        """TruFor has no /health; probe with OPTIONS or treat 405/422 as alive."""
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            try:
                resp = await client.post(f"{self.base_url}/score")
                return resp.status_code in (400, 422, 503)
            except httpx.ConnectError:
                return False

    async def score(self, image_path: str | Path) -> float:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            with path.open("rb") as f:
                resp = await client.post(
                    f"{self.base_url}/score",
                    files={"file": (path.name, f, "application/octet-stream")},
                )
            if resp.status_code >= 400:
                detail = resp.text
                try:
                    detail = resp.json()
                except Exception:
                    pass
                raise RuntimeError(f"TruFor score failed ({resp.status_code}): {detail}")
            data = resp.json()
            return float(data["score"])
