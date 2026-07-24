"""Request-scoped config propagation over Kafka headers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ...config import bind_config_overrides, get_config_overrides

CONFIG_CONTEXT_HEADER = "x-mm-config-context"


def inject_config_context(headers: dict[str, str] | None = None) -> dict[str, str]:
    """Attach current config overrides to Kafka headers when present."""

    output = dict(headers or {})
    overrides = get_config_overrides()
    if overrides is None or overrides.is_empty():
        return output

    output[CONFIG_CONTEXT_HEADER] = json.dumps(
        {
            "tenant_config": overrides.tenant_config,
            "project_config": overrides.project_config,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return output


@contextmanager
def bind_config_context_from_headers(headers: dict[str, str]) -> Iterator[None]:
    """Bind config overrides encoded in Kafka headers for this handler call."""

    raw = headers.get(CONFIG_CONTEXT_HEADER)
    if not raw:
        yield
        return

    parsed = _decode_config_context(raw)
    with bind_config_overrides(
        tenant_config=parsed.get("tenant_config"),
        project_config=parsed.get("project_config"),
    ):
        yield


def _decode_config_context(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid kafka config context header") from exc
    if not isinstance(parsed, dict):
        raise ValueError("invalid kafka config context header")
    tenant_config = parsed.get("tenant_config")
    project_config = parsed.get("project_config")
    if tenant_config is not None and not isinstance(tenant_config, dict):
        raise ValueError("invalid kafka config context header")
    if project_config is not None and not isinstance(project_config, dict):
        raise ValueError("invalid kafka config context header")
    return parsed
