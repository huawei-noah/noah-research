import json
from types import SimpleNamespace

import mindmemos.workers.schema_add_episode as schema_add_episode
import pytest

from mindmemos.infra.kafka import ConsumedMessage


def make_message(body: dict) -> ConsumedMessage:
    return ConsumedMessage(
        topic=schema_add_episode.TOPIC,
        partition=0,
        offset=1,
        key=None,
        value=json.dumps(body).encode("utf-8"),
    )


@pytest.mark.asyncio
async def test_schema_add_episode_worker_calls_schema_pipeline_generate(monkeypatch) -> None:
    calls = []
    pipeline_names = []

    class ConfiguredSchemaPipeline:
        async def generate_episode(
            self, context, add_record_ids, *, episode_id, consistency=None, trigger_record_id=None
        ):
            calls.append(
                SimpleNamespace(
                    context=context,
                    add_record_ids=add_record_ids,
                    episode_id=episode_id,
                    consistency=consistency,
                    trigger_record_id=trigger_record_id,
                )
            )

    monkeypatch.setattr(
        schema_add_episode,
        "create_pipeline",
        lambda *, type, name: pipeline_names.append(name) or ConfiguredSchemaPipeline(),
    )

    await schema_add_episode.handle_schema_add_episode(
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
                "add_record_ids": ["record-1"],
                "episode_id": "episode-1",
                "consistency": "strong",
                "trigger_record_id": "trace-1",
            }
        )
    )

    assert pipeline_names == ["schema_add"]
    assert len(calls) == 1
    assert calls[0].context.project_id == "proj-1"
    assert calls[0].add_record_ids == ["record-1"]
    assert calls[0].episode_id == "episode-1"
    assert calls[0].consistency == "strong"
    assert calls[0].trigger_record_id == "trace-1"


@pytest.mark.asyncio
async def test_schema_add_episode_worker_rejects_pipeline_without_generate(monkeypatch) -> None:
    class MissingGeneratePipeline:
        pass

    monkeypatch.setattr(schema_add_episode, "create_pipeline", lambda *, type, name: MissingGeneratePipeline())

    with pytest.raises(RuntimeError, match="support generate_episode"):
        await schema_add_episode.handle_schema_add_episode(
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
                    "add_record_ids": ["record-1"],
                    "episode_id": "episode-1",
                }
            )
        )
