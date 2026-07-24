import json
from types import SimpleNamespace

import mindmemos.workers.skill_evolve as skill_evolve
import pytest
from mindmemos.infra.kafka import ConsumedMessage


def make_message(body: dict) -> ConsumedMessage:
    return ConsumedMessage(
        topic=skill_evolve.TOPIC,
        partition=0,
        offset=1,
        key=None,
        value=json.dumps(body).encode("utf-8"),
    )


@pytest.mark.asyncio
async def test_skill_evolve_worker_runs_configured_pipeline(monkeypatch) -> None:
    calls = []
    pipeline_names = []

    class ConfiguredPipeline:
        async def evolve(self, *, project_id: str, cloud_skill_id: str):
            calls.append((project_id, cloud_skill_id))
            return SimpleNamespace(evolved=False, new_version_id=None)

    monkeypatch.setattr(
        skill_evolve,
        "get_config",
        lambda: SimpleNamespace(pipelines={"skill_evolve": "trace_v2_summary"}),
    )
    monkeypatch.setattr(
        skill_evolve,
        "create_pipeline",
        lambda *, type, name: pipeline_names.append((type, name)) or ConfiguredPipeline(),
    )

    await skill_evolve.handle_skill_evolve(
        make_message(
            {
                "request_id": "req-1",
                "account_id": "acc-1",
                "project_id": "proj-1",
                "cloud_skill_id": "skill-1",
            }
        )
    )

    assert pipeline_names == [("skill_evolve", "trace_v2_summary")]
    assert calls == [("proj-1", "skill-1")]
