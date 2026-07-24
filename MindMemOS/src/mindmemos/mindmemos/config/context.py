import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ..errors import ConfigNotInitializedError
from .app import MemoryConfig, build_config
from .base import safe_dict

_global_config: MemoryConfig | None = None
_current: ContextVar[MemoryConfig | None] = ContextVar("config_context", default=None)
_current_overrides: ContextVar["ConfigOverrides | None"] = ContextVar("config_overrides_context", default=None)


@dataclass(frozen=True)
class ConfigOverrides:
    """Request-scoped config fragments applied on top of the process base config."""

    tenant_config: dict[str, Any] | None = None
    project_config: dict[str, Any] | None = None

    def is_empty(self) -> bool:
        return not self.tenant_config and not self.project_config


def init_config(config_name: str = "product", config_path: str | Path | None = None):
    global _global_config
    _global_config = build_config(config_name, config_path)
    _current.set(_global_config)
    _current_overrides.set(None)


def init_config_from_env() -> str:
    """Initialize config from MINDMEMOS_CONFIG_PATH or MINDMEMOS_CONFIG_NAME."""

    config_path = os.getenv("MINDMEMOS_CONFIG_PATH") or None
    config_name = os.getenv("MINDMEMOS_CONFIG_NAME") or "dev"
    init_config(config_name=config_name, config_path=config_path)
    return str(config_path or config_name)


def get_config() -> MemoryConfig:
    bound = _current.get()
    if bound:
        return bound
    if _global_config is None:
        raise ConfigNotInitializedError
    return _global_config


def update_config(
    tenant_config: dict[str, Any] | None = None,
    project_config: dict[str, Any] | None = None,
) -> None:
    """Merge tenant + project on top of the static base and bind the result.

    Per spec, called once per request cycle by the gateway-aware entry point
    (e.g. a FastAPI `Depends` or worker init). Tenant is applied before
    project, so project wins on conflicts.

    Inputs are pure data — this function never queries a DB or cache itself.
    """
    _current.set(_build_scoped_config(tenant_config, project_config))
    overrides = ConfigOverrides(tenant_config=tenant_config, project_config=project_config)
    _current_overrides.set(None if overrides.is_empty() else overrides)


def get_config_overrides() -> ConfigOverrides | None:
    """Return request-scoped config fragments currently bound to this context."""

    return _current_overrides.get()


@contextmanager
def bind_config_overrides(
    tenant_config: dict[str, Any] | None = None,
    project_config: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Temporarily bind config overrides, then restore the previous context."""

    cfg_token = _current.set(_build_scoped_config(tenant_config, project_config))
    overrides = ConfigOverrides(tenant_config=tenant_config, project_config=project_config)
    overrides_token = _current_overrides.set(None if overrides.is_empty() else overrides)
    try:
        yield
    finally:
        _current.reset(cfg_token)
        _current_overrides.reset(overrides_token)


def _build_scoped_config(
    tenant_config: dict[str, Any] | None = None,
    project_config: dict[str, Any] | None = None,
) -> MemoryConfig:
    cfg = _global_config
    if cfg is None:
        raise ConfigNotInitializedError
    if tenant_config:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(tenant_config))
    if project_config:
        cfg = OmegaConf.merge(cfg, project_config)
    return cfg


def dump_config() -> dict[str, Any]:
    """Safe export of the current config; SENSITIVE fields are masked."""
    return safe_dict(get_config())


def reset_config() -> None:
    """Test helper: drop the context binding and rebuild the static base."""
    global _global_config
    _global_config = None
    _current.set(None)
    _current_overrides.set(None)
