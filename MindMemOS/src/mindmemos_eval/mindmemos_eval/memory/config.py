"""Benchmark matrix config loading."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .base import AlgorithmProfile, BenchmarkSpec, RunnerConfig

SUPPORTED_MEMORY_ALGORITHMS = frozenset({"vanilla", "schema"})
SUPPORTED_BENCHMARK_SEARCH_STRATEGIES = frozenset({"vanilla", "schema"})


def load_algorithm_profiles(path: str | Path) -> dict[str, AlgorithmProfile]:
    """Load named algorithm profiles from YAML."""
    data = load_benchmark_matrix_config(path)
    profiles = data.get("algorithm_profiles") or {}
    if not isinstance(profiles, Mapping):
        raise ValueError("algorithm_profiles must be a mapping")

    loaded: dict[str, AlgorithmProfile] = {}
    for name, raw in profiles.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"algorithm profile {name!r} must be a mapping")
        memory_algorithm = validate_memory_algorithm(raw.get("memory_algorithm"))
        search_params = dict(raw.get("search_params") or {})
        project_override_config = _optional_mapping(
            raw.get("project_override_config"), field_name=f"{name}.project_override_config"
        )
        loaded[str(name)] = AlgorithmProfile(
            name=str(name),
            memory_algorithm=memory_algorithm,
            search_params=search_params,
            project_override_config=project_override_config,
            raw=dict(raw),
        )
    return loaded


def load_benchmark_specs(
    path: str | Path,
    *,
    algorithm_override: str | None = None,
    memory_algorithm_override: str | None = None,
) -> dict[str, BenchmarkSpec]:
    """Load benchmark specs from YAML."""
    data = load_benchmark_matrix_config(path)
    benchmarks = data.get("benchmarks") or {}
    execution_params = data.get("execution_params") or {}
    if execution_params and not isinstance(execution_params, Mapping):
        raise ValueError("execution_params must be a mapping")
    profiles = load_algorithm_profiles(path)
    specs: dict[str, BenchmarkSpec] = {}
    selected_algorithm = validate_memory_algorithm(algorithm_override) if algorithm_override is not None else None
    legacy_override = (
        validate_memory_algorithm(memory_algorithm_override) if memory_algorithm_override is not None else None
    )
    for name, raw in benchmarks.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"benchmark {name!r} must be a mapping")
        dataset = str(raw.get("dataset") or "")
        if not dataset:
            raise ValueError(f"benchmark {name!r} is missing dataset")

        default_algorithm = (
            validate_memory_algorithm(raw.get("default_algorithm")) if raw.get("default_algorithm") else None
        )
        profile_name = selected_algorithm or default_algorithm or (str(raw["profile"]) if raw.get("profile") else None)
        profile = profiles.get(profile_name) if profile_name else None
        if profile_name and profile is None:
            raise ValueError(f"benchmark {name!r} references unknown algorithm profile {profile_name!r}")

        if legacy_override is not None:
            memory_algorithm = legacy_override
        elif profile is not None:
            memory_algorithm = profile.memory_algorithm
        else:
            memory_algorithm = validate_memory_algorithm(raw.get("memory_algorithm"))
        search_params = dict(profile.search_params if profile is not None else {})
        search_params.update(dict(raw.get("search_params") or {}))
        benchmark_execution_params = {}
        if name in execution_params:
            raw_execution_params = execution_params[name]
            if not isinstance(raw_execution_params, Mapping):
                raise ValueError(f"execution_params.{name} must be a mapping")
            benchmark_execution_params = dict(raw_execution_params)
        project_override_config = (
            _optional_mapping(raw.get("project_override_config"), field_name=f"{name}.project_override_config")
            if raw.get("project_override_config") is not None
            else (profile.project_override_config if profile is not None else None)
        )
        specs[str(name)] = BenchmarkSpec(
            name=str(name),
            dataset=dataset,
            memory_algorithm=memory_algorithm,
            profile=profile_name,
            limit=_optional_int(raw.get("limit")),
            search_params=search_params,
            execution_params=benchmark_execution_params,
            project_override_config=project_override_config,
            raw=dict(raw),
        )
    return specs


def load_benchmark_matrix_config(path: str | Path) -> dict[str, Any]:
    """Load the raw benchmark matrix YAML."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError("benchmark matrix config must be a mapping")
    return dict(data)


def load_runner_config(path: str | Path) -> RunnerConfig:
    """Load runner defaults from the benchmark matrix YAML."""
    data = load_benchmark_matrix_config(path)
    raw = data.get("runner") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("runner config must be a mapping")
    llm = dict(raw.get("llm") or {})
    answer_llm = dict(raw.get("answer_llm") or {})
    judge_llm = dict(raw.get("judge_llm") or {})
    return RunnerConfig(
        base_url=str(raw.get("base_url") or "http://127.0.0.1:8000"),
        timeout_seconds=float(raw.get("timeout_seconds", 600.0)),
        top_k=int(raw.get("top_k", 50)),
        search_strategy=str(raw["search_strategy"]) if raw.get("search_strategy") else None,
        rerank=bool(raw.get("rerank", True)),
        max_conv_concurrency=int(raw.get("max_conv_concurrency", 2)),
        max_qa_concurrency=int(raw.get("max_qa_concurrency", 8)),
        max_search_concurrency=_optional_int(raw.get("max_search_concurrency")),
        max_score_concurrency=_optional_int(raw.get("max_score_concurrency")),
        add=bool(raw.get("add", True)),
        score=bool(raw.get("score", True)),
        show_progress=bool(raw.get("show_progress", True)),
        llm_model=str(llm.get("model") or os.getenv("MINDMEMOS_EVAL_LLM_MODEL") or "gpt-4o-mini"),
        llm_api_key=llm.get("api_key") or os.getenv("MINDMEMOS_EVAL_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        llm_base_url=llm.get("base_url") or os.getenv("MINDMEMOS_EVAL_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        llm_temperature=_optional_float(llm.get("temperature"), default=0.0),
        llm_max_tokens=_optional_int(llm.get("max_tokens")),
        llm_timeout=float(llm.get("timeout", 600.0)),
        answer_llm_model=answer_llm.get("model"),
        answer_llm_api_key=answer_llm.get("api_key"),
        answer_llm_base_url=answer_llm.get("base_url"),
        judge_llm_model=judge_llm.get("model"),
        judge_llm_api_key=judge_llm.get("api_key"),
        judge_llm_base_url=judge_llm.get("base_url"),
    )


def validate_memory_algorithm(value: Any) -> str:
    """Validate a configured memory algorithm."""
    algorithm = str(value or "").strip().lower()
    if algorithm not in SUPPORTED_MEMORY_ALGORITHMS:
        supported = ", ".join(sorted(SUPPORTED_MEMORY_ALGORITHMS))
        raise ValueError(f"unsupported memory_algorithm {algorithm!r}; expected one of: {supported}")
    return algorithm


def to_public_search_strategy(value: Any) -> str:
    """Map benchmark search strategy config to the public API search mode.

    Benchmark configs use ``vanilla`` / ``schema`` to describe the intended
    engine. The public API receives ``fast`` because the actual engine is
    selected server-side from ``memory_algorithm`` on the API key.
    """
    strategy = str(value or "vanilla").strip().lower()
    if strategy not in SUPPORTED_BENCHMARK_SEARCH_STRATEGIES:
        supported = ", ".join(sorted(SUPPORTED_BENCHMARK_SEARCH_STRATEGIES))
        raise ValueError(f"unsupported search_strategy {strategy!r}; expected one of: {supported}")
    return "fast"


def resolve_public_search_strategy(value: Any) -> str:
    """Resolve a public API search mode from algorithm profile configuration."""

    strategy = str(value or "fast").strip().lower()
    if strategy in {"fast", "agentic"}:
        return strategy
    return to_public_search_strategy(strategy)


def override_spec_memory_algorithm(spec: BenchmarkSpec, memory_algorithm: str | None) -> BenchmarkSpec:
    """Return a spec with a command-line memory algorithm override applied."""
    if memory_algorithm is None:
        return spec
    algorithm = validate_memory_algorithm(memory_algorithm)
    return BenchmarkSpec(
        name=spec.name,
        dataset=spec.dataset,
        memory_algorithm=algorithm,
        profile=spec.profile,
        limit=spec.limit,
        search_params=dict(spec.search_params),
        execution_params=dict(spec.execution_params),
        project_override_config=spec.project_override_config,
        raw={**spec.raw, "memory_algorithm": algorithm},
    )


def _parse_benchmark_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any, *, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    return float(value)


def _optional_mapping(value: Any, *, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return dict(value)


def _option(args: argparse.Namespace, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _merged_runner_config(args: argparse.Namespace) -> RunnerConfig:
    config = load_runner_config(args.benchmark_config)
    overrides = {
        "base_url": _option(args, "base_url"),
        "timeout_seconds": _option(args, "timeout_seconds"),
        "top_k": _option(args, "top_k"),
        "search_strategy": _option(args, "search_strategy"),
        "rerank": _option(args, "rerank"),
        "max_conv_concurrency": _option(args, "max_conv_concurrency"),
        "max_qa_concurrency": _option(args, "max_qa_concurrency"),
        "max_search_concurrency": _option(args, "max_search_concurrency"),
        "max_score_concurrency": _option(args, "max_score_concurrency"),
        "add": _option(args, "add"),
        "score": _option(args, "score"),
        "show_progress": _option(args, "show_progress"),
        "llm_model": _option(args, "llm_model"),
        "llm_api_key": _option(args, "llm_api_key"),
        "llm_base_url": _option(args, "llm_base_url"),
        "llm_temperature": _option(args, "llm_temperature"),
        "llm_max_tokens": _option(args, "llm_max_tokens"),
        "llm_timeout": _option(args, "llm_timeout"),
        "answer_llm_model": _option(args, "answer_llm_model"),
        "answer_llm_api_key": _option(args, "answer_llm_api_key"),
        "answer_llm_base_url": _option(args, "answer_llm_base_url"),
        "judge_llm_model": _option(args, "judge_llm_model"),
        "judge_llm_api_key": _option(args, "judge_llm_api_key"),
        "judge_llm_base_url": _option(args, "judge_llm_base_url"),
    }
    data = config.__dict__.copy()
    data.update({key: value for key, value in overrides.items() if value is not None})
    return RunnerConfig(**data)
