from types import SimpleNamespace

import mindmemos.workers.runtime as runtime
import pytest


@pytest.mark.asyncio
async def test_worker_runtime_initializes_config_from_env(monkeypatch) -> None:
    calls: list[str] = []

    def init_config_from_env() -> str:
        calls.append("config")
        return "worker-config"

    monkeypatch.setattr(runtime, "init_config_from_env", init_config_from_env)
    monkeypatch.setattr(
        runtime,
        "get_config",
        lambda: SimpleNamespace(telemetry=SimpleNamespace(log_level="INFO")),
    )
    monkeypatch.setattr(runtime, "configure_logging", lambda *, level: calls.append(f"logging:{level}"))
    monkeypatch.setattr(runtime, "configure_tracing", lambda telemetry: calls.append("tracing"))

    async def ensure_database_schema() -> None:
        calls.append("schema")

    monkeypatch.setattr(runtime, "ensure_database_schema", ensure_database_schema)
    monkeypatch.setattr(runtime, "init_llm_client", lambda: calls.append("llm"))
    monkeypatch.setattr(runtime, "init_embed_client", lambda: calls.append("embed"))

    async def validate_embedding_dimension() -> None:
        calls.append("embedding-check")

    monkeypatch.setattr(runtime, "validate_embedding_dimension", validate_embedding_dimension)

    await runtime.start_worker_runtime()

    assert calls == ["config", "logging:INFO", "tracing", "schema", "llm", "embed", "embedding-check"]
