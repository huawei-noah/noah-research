"""Pipeline class registration and factory helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

PipelineType = str

_VALID_PIPELINE_TYPES = {"add", "search", "get", "delete", "update", "feedback", "dreaming", "skill_evolve"}
_PIPELINE_REGISTRY: dict[PipelineType, dict[str, type[Any]]] = {}
_BUILTINS_LOADED = False


def register(*, type: PipelineType, name: str):
    """Register a pipeline class under a type/name pair."""

    if type not in _VALID_PIPELINE_TYPES:
        valid = ", ".join(sorted(_VALID_PIPELINE_TYPES))
        raise ValueError(f"Unknown pipeline type {type!r}. Valid pipeline types: {valid}")
    if not name:
        raise ValueError("pipeline name must not be empty")

    def decorator(cls: type[Any]) -> type[Any]:
        pipelines = _PIPELINE_REGISTRY.setdefault(type, {})
        if name in pipelines:
            raise ValueError(f"{type} pipeline {name!r} is already registered")
        pipelines[name] = cls
        return cls

    return decorator


def create_pipeline(*, type: PipelineType, name: str, **kwargs: Any) -> Any:
    """Create a registered pipeline by type/name."""

    load_builtin_pipelines()
    pipeline_cls = _PIPELINE_REGISTRY.get(type, {}).get(name)
    if pipeline_cls is None:
        available = ", ".join(sorted(_PIPELINE_REGISTRY.get(type, {}))) or "<none>"
        raise ValueError(f"Unknown {type} pipeline {name!r}. Available {type} pipelines: {available}")
    return pipeline_cls(**kwargs)


def load_builtin_pipelines() -> None:
    """Import built-in pipeline modules so their decorators run."""

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    for module_name in (
        ".add.default",
        ".add.schema",
        ".add.vanilla",
        ".delete.default",
        ".dreaming.default",
        ".feedback.default",
        ".get.default",
        ".search.default",
        ".search.pipeline",
        ".skill.evolution",
        ".update.default",
    ):
        import_module(module_name, package=__package__)

    _BUILTINS_LOADED = True
