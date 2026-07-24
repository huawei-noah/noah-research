import json
from types import SimpleNamespace

import mindmemos.workers.schema_add_drain as schema_add_drain
import pytest

from mindmemos.infra.kafka import ConsumedMessage


def make_message(body: dict) -> ConsumedMessage:
    return ConsumedMessage(
        topic=schema_add_drain.TOPIC,
        partition=0,
        offset=1,
        key=None,
        value=json.dumps(body).encode("utf-8"),
    )


@pytest.mark.asyncio
async def test_schema_add_drain_worker_calls_schema_pipeline_drain(monkeypatch) -> None:
    calls = []
    pipeline_names = []

    class ConfiguredSchemaPipeline:
        async def drain_buffer(self, context, *, consistency=None, force=False, trigger_record_id=None):
            calls.append(
                SimpleNamespace(
                    context=context, consistency=consistency, force=force, trigger_record_id=trigger_record_id
                )
            )

    monkeypatch.setattr(
        schema_add_drain,
        "create_pipeline",
        lambda *, type, name: pipeline_names.append(name) or ConfiguredSchemaPipeline(),
    )

    await schema_add_drain.handle_schema_add_drain(
        make_message(
            {
                "context": {
                    "request_id": "req-1",
                    "account_id": "acc-1",
                    "project_id": "proj-1",
                    "api_key_uuid": "key-1",
                    "memory_algorithm": "schema",
                    "user_id": "user-1",
                    "session_id": "session-1",
                },
                "force": True,
                "consistency": "strong",
                "trigger_record_id": "trace-1",
            }
        )
    )

    assert len(calls) == 1
    assert pipeline_names == ["schema_add"]
    assert calls[0].context.project_id == "proj-1"
    assert calls[0].consistency == "strong"
    assert calls[0].force is True
    assert calls[0].trigger_record_id == "trace-1"


@pytest.mark.asyncio
async def test_schema_add_drain_worker_rejects_pipeline_without_drain(monkeypatch) -> None:
    class MissingDrainPipeline:
        pass

    monkeypatch.setattr(schema_add_drain, "create_pipeline", lambda *, type, name: MissingDrainPipeline())

    with pytest.raises(RuntimeError, match="support drain_buffer"):
        await schema_add_drain.handle_schema_add_drain(
            make_message(
                {
                    "context": {
                        "request_id": "req-1",
                        "account_id": "acc-1",
                        "project_id": "proj-1",
                        "api_key_uuid": "key-1",
                        "memory_algorithm": "schema",
                        "user_id": "user-1",
                        "session_id": "session-1",
                    }
                }
            )
        )
