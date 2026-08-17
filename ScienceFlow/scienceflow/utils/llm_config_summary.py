# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

"""Helpers for displaying LLM runtime configuration without secrets."""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


_STAGE_ENV = {
    "code": {
        "model": "CODE_MODEL",
        "models": "CODE_MODELS",
        "api_key": "CODE_API_KEY",
        "api_keys": "CODE_API_KEYS",
        "base_url": "CODE_BASE_URL",
        "base_urls": "CODE_BASE_URLS",
        "routing_mode": "CODE_LLM_ROUTING_MODE",
        "sticky_id": "CODE_LLM_STICKY_ID",
        "sticky_primary_index": "CODE_LLM_STICKY_PRIMARY_INDEX",
    },
    "feedback": {
        "model": "FEEDBACK_MODEL",
        "models": "FEEDBACK_MODELS",
        "api_key": "FEEDBACK_API_KEY",
        "api_keys": "FEEDBACK_API_KEYS",
        "base_url": "FEEDBACK_BASE_URL",
        "base_urls": "FEEDBACK_BASE_URLS",
        "routing_mode": "FEEDBACK_LLM_ROUTING_MODE",
        "sticky_id": "FEEDBACK_LLM_STICKY_ID",
        "sticky_primary_index": "FEEDBACK_LLM_STICKY_PRIMARY_INDEX",
    },
}


def split_csvish(value: Any) -> list[str]:
    """Return non-empty values from list/scalar comma or whitespace input."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[,\s]+", text) if part.strip()]


def sanitize_base_url(value: Any) -> str:
    """Return a display-safe URL without userinfo, query, or fragment."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = urlsplit(text)
    except ValueError:
        return _truncate(text)
    if not parsed.scheme or not parsed.netloc:
        return _truncate(text.split("?", 1)[0].split("#", 1)[0])
    host = parsed.hostname or parsed.netloc.rsplit("@", 1)[-1]
    if parsed.port:
        host = f"{host}:{parsed.port}"
    path = parsed.path.rstrip("/")
    return _truncate(urlunsplit((parsed.scheme, host, path, "", "")))


def summarize_llm_config_from_env(env: dict[str, str], *, source: str = "env") -> dict[str, Any]:
    """Build a secret-free summary matching config env override semantics."""
    stages: dict[str, Any] = {}
    for stage_name, names in _STAGE_ENV.items():
        stages[stage_name] = _summarize_env_stage(env, names, source=source)
    return {"source": source, **stages}


def summarize_llm_config_from_resolved_config(data: dict[str, Any]) -> dict[str, Any]:
    """Build a secret-free LLM summary from ``resolved_config.yaml`` contents."""
    agent = data.get("agent") if isinstance(data, dict) else None
    if not isinstance(agent, dict):
        return {}
    stages: dict[str, Any] = {}
    for stage_name in ("code", "feedback"):
        stage = agent.get(stage_name)
        if isinstance(stage, dict):
            stages[stage_name] = _summarize_mapping_stage(stage, source="resolved_config")
    return {"source": "resolved_config", **stages} if stages else {}




def summarize_llm_config_from_agent_patch(
    agent_patch: dict[str, Any] | None,
    *,
    source: str = "parallel_runner_agent",
) -> dict[str, Any]:
    """Build a secret-free LLM summary from a parallel manifest ``agent`` patch."""
    if not isinstance(agent_patch, dict):
        return {}
    agent: dict[str, Any] = {}
    for stage_name in ("code", "feedback"):
        stage = agent_patch.get(stage_name)
        if isinstance(stage, dict):
            agent[stage_name] = _expand_env_values(stage)
    summary = summarize_llm_config_from_resolved_config({"agent": agent})
    if not summary:
        return {}
    summary["source"] = source
    for stage_name in ("code", "feedback"):
        stage = summary.get(stage_name)
        if isinstance(stage, dict):
            stage["source"] = source
    return summary


def merge_llm_config_summary(
    base: dict[str, Any],
    override: dict[str, Any],
    *,
    source: str = "merged",
) -> dict[str, Any]:
    """Merge stage summaries, with non-empty override stages winning."""
    out = dict(base or {})
    for stage_name in ("code", "feedback"):
        stage = override.get(stage_name) if isinstance(override, dict) else None
        if isinstance(stage, dict) and stage:
            out[stage_name] = stage
    if out:
        out["source"] = source
    return out

def format_llm_config_summary(summary: dict[str, Any]) -> str:
    """Compact one-cell display string for monitor tables."""
    if not isinstance(summary, dict) or not summary:
        return ""
    lines: list[str] = []
    for stage_name in ("code", "feedback"):
        stage = summary.get(stage_name)
        if isinstance(stage, dict):
            text = _format_stage(stage_name, stage)
            if text:
                lines.append(text)
    seen = summary.get("observed_models")
    if isinstance(seen, list) and seen:
        lines.append("seen=" + ",".join(_safe_model_name(x) for x in seen[:4]))
        if len(seen) > 4:
            lines[-1] += f"+{len(seen) - 4}"
    return "\n".join(lines)


def observed_models_from_log_file(path: Any, *, max_chars: int = 2_000_000) -> list[str]:
    """Extract model names from ``[LLM] <model>: ...`` log lines."""
    text = _read_head_tail(Path(str(path)), max_chars=max_chars) if str(path or "").strip() else ""
    if not text:
        return []
    counts: Counter[str] = Counter()
    for match in re.finditer(r"\[LLM\]\s+([^:\n\r]+):", text):
        name = _safe_model_name(match.group(1))
        if name:
            counts[name] += 1
    return [name for name, _ in counts.most_common()]


def merge_observed_models(*groups: Any) -> list[str]:
    """Merge observed model names preserving first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        if isinstance(group, dict):
            values = group.keys()
        elif isinstance(group, (list, tuple, set)):
            values = group
        else:
            values = []
        for raw in values:
            name = _safe_model_name(raw)
            if name and name not in seen:
                out.append(name)
                seen.add(name)
    return out


def _expand_env_values(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_values(item) for key, item in value.items()}
    return value


def _summarize_env_stage(env: dict[str, str], names: dict[str, str], *, source: str) -> dict[str, Any]:
    models = split_csvish(env.get(names["models"]))
    model = str(env.get(names["model"]) or "").strip()
    keys = split_csvish(env.get(names["api_keys"]))
    if not keys and str(env.get(names["api_key"]) or "").strip():
        keys = ["<set>"]
    if not keys:
        keys = split_csvish(env.get("API_KEYS"))
    if not keys and str(env.get("API_KEY") or "").strip():
        keys = ["<set>"]
    urls = split_csvish(env.get(names["base_urls"]))
    if not urls and str(env.get(names["base_url"]) or "").strip():
        urls = [str(env.get(names["base_url"]) or "").strip()]
    if not urls:
        urls = split_csvish(env.get("BASE_URLS"))
    if not urls and str(env.get("BASE_URL") or "").strip():
        urls = [str(env.get("BASE_URL") or "").strip()]
    routing_mode = str(env.get(names["routing_mode"]) or env.get("SCIENCEFLOW_LLM_ROUTING_MODE") or "").strip()
    sticky_id_set = bool(str(env.get(names["sticky_id"]) or env.get("SCIENCEFLOW_LLM_STICKY_ID") or "").strip())
    sticky_primary = str(
        env.get(names["sticky_primary_index"]) or env.get("SCIENCEFLOW_LLM_STICKY_PRIMARY_INDEX") or ""
    ).strip()
    return _stage_summary(
        model=model,
        models=models,
        keys=keys,
        urls=urls,
        routing_mode=routing_mode,
        sticky_id_set=sticky_id_set,
        sticky_primary_index=sticky_primary,
        source=source,
    )


def _summarize_mapping_stage(stage: dict[str, Any], *, source: str) -> dict[str, Any]:
    keys = split_csvish(stage.get("api_keys"))
    if not keys and str(stage.get("api_key") or "").strip():
        keys = ["<set>"]
    urls = split_csvish(stage.get("base_urls"))
    if not urls and str(stage.get("base_url") or "").strip():
        urls = [str(stage.get("base_url") or "").strip()]
    return _stage_summary(
        model=str(stage.get("model") or "").strip(),
        models=split_csvish(stage.get("models")),
        keys=keys,
        urls=urls,
        routing_mode=str(stage.get("api_routing_mode") or "").strip(),
        sticky_id_set=bool(str(stage.get("api_sticky_id") or "").strip()),
        sticky_primary_index=stage.get("api_sticky_primary_index"),
        source=source,
    )


def _stage_summary(
    *,
    model: str,
    models: list[str],
    keys: list[str],
    urls: list[str],
    routing_mode: str,
    sticky_id_set: bool,
    sticky_primary_index: Any,
    source: str,
) -> dict[str, Any]:
    key_count = len(keys)
    url_count = len(urls)
    endpoint_count = max(key_count, url_count)
    if key_count > 1 and url_count == 1:
        endpoint_count = key_count
    return {
        "model": _safe_model_name(model),
        "models": [_safe_model_name(x) for x in models if _safe_model_name(x)],
        "key_count": key_count,
        "base_url": sanitize_base_url(urls[0]) if urls else "",
        "base_url_count": url_count,
        "endpoint_count": endpoint_count,
        "routing_mode": routing_mode,
        "sticky_id_set": sticky_id_set,
        "sticky_primary_index": sticky_primary_index if sticky_primary_index not in (None, "") else None,
        "source": source,
    }


def _format_stage(stage_name: str, stage: dict[str, Any]) -> str:
    model = _safe_model_name(stage.get("model")) or "?"
    models = stage.get("models") if isinstance(stage.get("models"), list) else []
    if models:
        model += f"+{len(models)}"
    parts = [f"{stage_name}={model}"]
    key_count = _positive_int(stage.get("key_count"))
    endpoint_count = _positive_int(stage.get("endpoint_count"))
    if key_count:
        if endpoint_count and endpoint_count != key_count:
            parts.append(f"keys={key_count}/ep={endpoint_count}")
        else:
            parts.append(f"keys={key_count}")
    url = sanitize_base_url(stage.get("base_url"))
    url_count = _positive_int(stage.get("base_url_count"))
    if url:
        if url_count and url_count > 1:
            parts.append(f"url={url}+{url_count - 1}")
        else:
            parts.append(f"url={url}")
    route = str(stage.get("routing_mode") or "").strip()
    if route:
        parts.append(f"route={route}")
    primary = stage.get("sticky_primary_index")
    if primary is not None:
        parts.append(f"primary={primary}")
    return " ".join(parts)


def _safe_model_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"[^A-Za-z0-9._:/+-]", "_", text)[:80]


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _truncate(value: str, max_len: int = 96) -> str:
    text = value.strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def _read_head_tail(path: Path, *, max_chars: int) -> str:
    try:
        if not path.is_file():
            return ""
        size = path.stat().st_size
        if size <= max_chars:
            return path.read_text(encoding="utf-8", errors="replace")
        half = max_chars // 2
        with path.open("rb") as f:
            head = f.read(half)
            f.seek(max(0, size - half))
            tail = f.read(half)
        return (
            head.decode("utf-8", errors="replace")
            + "\n"
            + tail.decode("utf-8", errors="replace")
        )
    except OSError:
        return ""
