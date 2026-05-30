from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class NewsPost:
    image_path: str
    headline: str = ""
    body: str = ""
    source: str = ""
    fake_type: str = "matching"

    def has_text(self) -> bool:
        return bool(self.headline.strip() or self.body.strip() or self.source.strip())

    def rag_query_suffix(self) -> str:
        parts: list[str] = []
        if self.headline.strip():
            parts.append(f"标题:{self.headline.strip()}")
        if self.body.strip():
            parts.append(f"正文:{self.body.strip()}")
        if self.source.strip():
            parts.append(f"来源:{self.source.strip()}")
        return "\n".join(parts)

    @classmethod
    def from_case_dict(
        cls,
        case: dict[str, Any],
        cases_dir: Path,
        *,
        metadata_path: Path | None = None,
    ) -> "NewsPost":
        image = case.get("image", "")
        if metadata_path is not None and case.get("dataset") == "weibo":
            from harness.weibo_loader import resolve_case_image

            image_path = str(resolve_case_image(case, metadata_path))
        else:
            image_path = str((cases_dir / image).resolve())
        return cls(
            image_path=image_path,
            headline=str(case.get("headline", "")),
            body=str(case.get("body", "")),
            source=str(case.get("source", "")),
            fake_type=str(case.get("fake_type", "matching")),
        )

    @classmethod
    def from_metadata(cls, metadata_path: Path, case_id: str) -> "NewsPost":
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        for case in data.get("cases", []):
            if case.get("id") == case_id:
                return cls.from_case_dict(
                    case, metadata_path.parent, metadata_path=metadata_path
                )
        raise KeyError(f"Case not found in metadata: {case_id}")
