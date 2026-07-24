"""Shared benchmark matrix types."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..llm import LLMClient
from .identity import RunIdentity


@dataclass(frozen=True)
class AlgorithmProfile:
    """Named algorithm configuration used by benchmark runs."""

    name: str
    memory_algorithm: str
    search_params: dict[str, Any] = field(default_factory=dict)
    project_override_config: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    """Mutable per-run context shared by adapters and manifest writing."""

    identity: RunIdentity
    request_ids: dict[str, list[str]] = field(
        default_factory=lambda: {
            "add": [],
            "search": [],
            "answer": [],
            "eval": [],
        }
    )
    request_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record_request_id(self, stage: str, request_id: str | None) -> None:
        """Remember one server-generated request id for a benchmark stage."""
        if not request_id:
            return
        if stage not in self.request_ids:
            self.request_ids[stage] = []
        index = len(self.request_ids[stage]) + 1
        self.request_ids[stage].append(request_id)
        self.request_metadata[request_id] = {
            "benchmark": self.identity.benchmark,
            "run_id": self.identity.run_id,
            "stage": stage,
            "index": index,
        }



@dataclass(frozen=True)
class BenchmarkSpec:
    """Normalized config for one benchmark."""

    name: str
    dataset: str
    memory_algorithm: str
    profile: str | None = None
    limit: int | None = None
    search_params: dict[str, Any] = field(default_factory=dict)
    execution_params: dict[str, Any] = field(default_factory=dict)
    project_override_config: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunnerConfig:
    """Runtime knobs loaded from benchmark matrix YAML."""

    base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 600.0
    top_k: int = 50
    search_strategy: str | None = None
    rerank: bool = True
    max_conv_concurrency: int = 2
    max_qa_concurrency: int = 8
    max_search_concurrency: int | None = None
    max_score_concurrency: int | None = None
    add: bool = True
    score: bool = True
    show_progress: bool = True
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_temperature: float | None = 0.0
    llm_max_tokens: int | None = None
    llm_timeout: float = 600.0
    answer_llm_model: str | None = None
    answer_llm_api_key: str | None = None
    answer_llm_base_url: str | None = None
    judge_llm_model: str | None = None
    judge_llm_api_key: str | None = None
    judge_llm_base_url: str | None = None




class BenchmarkAdapter(Protocol):
    """Benchmark adapter protocol."""

    name: str

    async def run(
        self,
        *,
        memory: Any,
        answer_llm: LLMClient,
        judge_llm: LLMClient,
        ctx: RunContext,
        bench_config: BenchmarkSpec,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        """Run one benchmark and return a JSON-serializable result."""
        ...
