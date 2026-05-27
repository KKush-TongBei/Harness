from __future__ import annotations

import asyncio
from pathlib import Path

from harness.context import ContextManager
from harness.evaluation import parse_model_output
from harness.fusion_policy import apply_fusion_policy
from harness.state import PipelineState, StateStorage
from harness.tools.registry import ToolRegistry


class ExecutionLoop:
    """Four-phase Harness pipeline orchestrator."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        context: ContextManager | None = None,
        storage: StateStorage | None = None,
        max_new_tokens: int = 512,
    ):
        self.registry = registry or ToolRegistry()
        self.context = context or ContextManager()
        self.storage = storage or StateStorage()
        self.max_new_tokens = max_new_tokens

    async def run_harness(self, image_path: str | Path, save: bool = True) -> PipelineState:
        path = str(Path(image_path).resolve())
        state = PipelineState(input_image=path, mode="harness")

        # Phase 1: Init + health check
        health = await self.registry.health_check()
        if not health.get("all_ok"):
            state.errors.append(f"Service health check failed: {health}")
            if save:
                self.storage.save(state)
            raise RuntimeError(f"Services not ready: {health}")

        # Phase 2: Visual translation + RAG
        state.description = await self.registry.describe_image(
            path,
            self.context.describe_prompt,
            max_new_tokens=self.max_new_tokens,
        )

        anchors = await self.registry.search_anchors(state.description)
        state.anchors = [a.to_dict() for a in anchors]

        # Phase 3: Forgery detection (parallel with phase 2 in production; serial for clarity)
        state.forgery_score = await self.registry.forgery_score(path)

        # Phase 4: Fusion verdict
        fusion_prompt = self.context.build_fusion_prompt(
            description=state.description,
            anchors=state.anchors,
            score=state.forgery_score,
        )
        raw = await self.registry.describe_image(
            path,
            fusion_prompt,
            max_new_tokens=self.max_new_tokens,
        )
        state.raw_fusion_output = raw
        result = parse_model_output(raw)
        if not result.parse_ok:
            state.errors.append("Fusion output JSON parse failed; used fallback heuristics")
        result = apply_fusion_policy(state.anchors, state.forgery_score, result)
        state.verdict = result.verdict
        state.confidence = result.confidence
        state.reasoning = result.reasoning
        state.evidence_chain = result.evidence_chain

        if save:
            self.storage.save(state)
        return state

    async def run_baseline(self, image_path: str | Path, save: bool = True) -> PipelineState:
        path = str(Path(image_path).resolve())
        state = PipelineState(input_image=path, mode="baseline")

        health = await self.registry.health_check()
        if not health.get("qwen"):
            state.errors.append(f"Qwen not ready: {health}")
            if save:
                self.storage.save(state)
            raise RuntimeError(f"Qwen service not ready: {health}")

        raw = await self.registry.describe_image(
            path,
            self.context.baseline_prompt,
            max_new_tokens=self.max_new_tokens,
        )
        state.raw_fusion_output = raw
        result = parse_model_output(raw)
        state.verdict = result.verdict
        state.confidence = result.confidence
        state.reasoning = result.reasoning
        state.evidence_chain = result.evidence_chain

        if save:
            self.storage.save(state)
        return state

    async def run_parallel_detect_and_rag(
        self, image_path: str, description: str
    ) -> tuple[list, float]:
        """Optional parallel Step2+3 helper."""
        anchors_task = self.registry.search_anchors(description)
        score_task = self.registry.forgery_score(image_path)
        anchors, score = await asyncio.gather(anchors_task, score_task)
        return [a.to_dict() for a in anchors], score
