from __future__ import annotations

import asyncio
from pathlib import Path

from harness.context import ContextManager
from harness.evaluation import parse_model_output
from harness.fusion_policy import apply_fusion_policy
from harness.news_post import NewsPost
from harness.rag_query import (
    build_misinformation_probe,
    build_official_support_probe,
    merge_anchors,
)
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

    def _apply_news_to_state(self, state: PipelineState, news: NewsPost | None) -> None:
        if news is None:
            return
        state.headline = news.headline
        state.body = news.body
        state.source = news.source
        state.fake_type = news.fake_type

    def _rag_query(self, description: str, news: NewsPost | None) -> str:
        query = description
        if news and news.has_text():
            suffix = news.rag_query_suffix()
            if suffix:
                query = f"{query}\n{suffix}"
        return query

    async def run_harness(
        self,
        image_path: str | Path,
        news: NewsPost | None = None,
        save: bool = True,
    ) -> PipelineState:
        path = str(Path(image_path).resolve())
        state = PipelineState(input_image=path, mode="harness")
        self._apply_news_to_state(state, news)

        health = await self.registry.health_check()
        if not health.get("all_ok"):
            state.errors.append(f"Service health check failed: {health}")
            if save:
                self.storage.save(state)
            raise RuntimeError(f"Services not ready: {health}")

        describe_prompt = self.context.build_describe_prompt(news)
        state.description = await self.registry.describe_image(
            path,
            describe_prompt,
            max_new_tokens=self.max_new_tokens,
        )

        rag_query = self._rag_query(state.description, news)
        anchors = await self.registry.search_anchors(rag_query)
        probe_text = state.description
        if news and news.has_text():
            probe_text = f"{probe_text}\n{news.headline}\n{news.body}"
        official_probe = build_official_support_probe(probe_text)
        if official_probe:
            extra = await self.registry.search_anchors(official_probe, top_k=2)
            anchors = merge_anchors(anchors, extra)
        probe = build_misinformation_probe(probe_text)
        if probe:
            extra = await self.registry.search_anchors(probe, top_k=2)
            anchors = merge_anchors(anchors, extra)
        state.anchors = [a.to_dict() for a in anchors]

        state.forgery_score = await self.registry.forgery_score(path)

        fusion_prompt = self.context.build_fusion_prompt(
            description=state.description,
            anchors=state.anchors,
            score=state.forgery_score,
            news=news,
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
        result = apply_fusion_policy(state.anchors, state.forgery_score, result, news=news)
        state.verdict = result.verdict
        state.issue_type = result.issue_type
        state.confidence = result.confidence
        state.reasoning = result.reasoning
        state.evidence_chain = result.evidence_chain

        if save:
            self.storage.save(state)
        return state

    async def run_baseline(
        self,
        image_path: str | Path,
        news: NewsPost | None = None,
        save: bool = True,
    ) -> PipelineState:
        path = str(Path(image_path).resolve())
        state = PipelineState(input_image=path, mode="baseline")
        self._apply_news_to_state(state, news)

        health = await self.registry.health_check()
        if not health.get("qwen"):
            state.errors.append(f"Qwen not ready: {health}")
            if save:
                self.storage.save(state)
            raise RuntimeError(f"Qwen service not ready: {health}")

        baseline_prompt = self.context.build_baseline_prompt(news)
        raw = await self.registry.describe_image(
            path,
            baseline_prompt,
            max_new_tokens=self.max_new_tokens,
        )
        state.raw_fusion_output = raw
        result = parse_model_output(raw)
        state.verdict = result.verdict
        state.issue_type = result.issue_type
        state.confidence = result.confidence
        state.reasoning = result.reasoning
        state.evidence_chain = result.evidence_chain

        if save:
            self.storage.save(state)
        return state

    async def run_parallel_detect_and_rag(
        self, image_path: str, description: str
    ) -> tuple[list, float]:
        anchors_task = self.registry.search_anchors(description)
        score_task = self.registry.forgery_score(image_path)
        anchors, score = await asyncio.gather(anchors_task, score_task)
        return [a.to_dict() for a in anchors], score
