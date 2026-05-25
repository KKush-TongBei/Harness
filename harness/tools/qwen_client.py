from __future__ import annotations

from pathlib import Path

import httpx

_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def _mime_for(path: Path) -> str:
    return _MIME.get(path.suffix.lower(), "image/jpeg")


class QwenClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 1800.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    async def describe(
        self,
        image_path: str | Path,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            with path.open("rb") as f:
                resp = await client.post(
                    f"{self.base_url}/v1/describe",
                    files={"image": (path.name, f, _mime_for(path))},
                    data={"prompt": prompt, "max_new_tokens": str(max_new_tokens)},
                )
            if resp.status_code >= 400:
                detail = resp.text
                try:
                    detail = resp.json()
                except Exception:
                    pass
                raise RuntimeError(f"Qwen describe failed ({resp.status_code}): {detail}")
            data = resp.json()
            return str(data.get("text", ""))
