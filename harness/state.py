from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4


Mode = Literal["harness", "baseline"]


@dataclass
class PipelineState:
    input_image: str
    mode: Mode = "harness"
    run_id: str = field(default_factory=lambda: uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    description: str | None = None
    anchors: list[dict[str, Any]] = field(default_factory=list)
    forgery_score: float | None = None
    verdict: str | None = None
    confidence: float | None = None
    reasoning: str | None = None
    evidence_chain: list[str] = field(default_factory=list)
    raw_fusion_output: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StateStorage:
    """Persist pipeline state and reports to outputs/."""

    def __init__(self, output_root: str | Path = "outputs"):
        self.output_root = Path(output_root)

    def run_dir(self, state: PipelineState) -> Path:
        sub = "ab_test" if state.mode == "baseline" else state.run_id
        if state.mode == "harness":
            return self.output_root / state.run_id
        return self.output_root / sub / state.run_id

    def save(self, state: PipelineState) -> Path:
        run_dir = self.run_dir(state)
        run_dir.mkdir(parents=True, exist_ok=True)

        state_path = run_dir / "state.json"
        state_path.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report = {
            "run_id": state.run_id,
            "mode": state.mode,
            "input_image": state.input_image,
            "verdict": state.verdict,
            "confidence": state.confidence,
            "reasoning": state.reasoning,
            "evidence_chain": state.evidence_chain,
            "forgery_score": state.forgery_score,
            "anchors_count": len(state.anchors),
            "timestamp": state.timestamp,
        }
        report_path = run_dir / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return run_dir
