"""Route tests for ``POST /v1/skills/register`` (design §5.2).

Drives the real :class:`SkillService` over an in-memory Qdrant-backed
``SkillVersionStore``, with auth stubbed via dependency override.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from mindmemos.api.deps import get_request_context
from mindmemos.api.schemas import AuthContext
from mindmemos.api.services import get_skill_service
from mindmemos.api.services.skill_service import SkillService
from mindmemos.api.skill_routes import router
from mindmemos.components.skill import compute_content_hash, serialize_bundle
from mindmemos.config import QdrantConfig
from mindmemos.errors import ApiError
from mindmemos.infra.db import SkillVersionRepository
from mindmemos.infra.db.qdrant import QdrantStore
from mindmemos.pipelines.skill import SKILL_EVOLVE_TOPIC, SkillVersionStore
from mindmemos.typing.skill import SkillEvolveResult
from qdrant_client import AsyncQdrantClient


def bundle(text: str) -> str:
    return serialize_bundle({"SKILL.md": text})


class _StubEvolver:
    """Records the evolve call args and returns a canned result."""

    def __init__(self, result: SkillEvolveResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    async def evolve(self, *, project_id: str, cloud_skill_id: str) -> SkillEvolveResult:
        self.calls.append((project_id, cloud_skill_id))
        return self.result


@pytest_asyncio.fixture
async def client():
    qclient = AsyncQdrantClient(":memory:")
    cfg = QdrantConfig(
        url="http://unused",
        add_record_collection="test_add_record",
        skill_version_collection="test_skill_version",
        skill_blob_collection="test_skill_blob",
        skill_trace_pending_collection="test_skill_trace_pending",
        vector_size=2,
    )
    qdrant = QdrantStore(cfg, client=qclient)
    await qdrant.ensure_schema()
    skill_repo = SkillVersionRepository(cfg, engine=qdrant.engine)
    service = SkillService(store=SkillVersionStore(skill_repo=skill_repo, add_record_repo=qdrant.add_record))

    app = FastAPI()

    @app.exception_handler(ApiError)
    async def _handle_api_error(request, exc):
        return JSONResponse(
            status_code=exc.status_code, content={"code": exc.code, "message": exc.message, "data": None}
        )

    app.include_router(router)
    app.dependency_overrides[get_skill_service] = lambda: service
    app.dependency_overrides[get_request_context] = lambda: AuthContext(
        request_id="req-1",
        account_id="acct",
        project_id="proj",
        api_key_uuid="key",
        memory_algorithm="schema",
        scopes=["memory:read", "memory:write"],
    )
    try:
        yield TestClient(app)
    finally:
        await qclient.close()


def test_register_returns_version_envelope(client):
    resp = client.post("/v1/skills/register", json={"name": "prd-writer", "content": bundle("hello")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == "ok"
    assert body["request_id"] == "req-1"
    data = body["data"]
    assert data["content_hash"] == compute_content_hash({"SKILL.md": "hello"})
    assert data["status"] == "observed"
    assert data["cloud_skill_id"]
    assert data["version_id"]


def test_register_is_idempotent_over_http(client):
    payload = {"name": "prd-writer", "content": bundle("hello")}
    first = client.post("/v1/skills/register", json=payload).json()["data"]
    second = client.post("/v1/skills/register", json=payload).json()["data"]
    assert first["version_id"] == second["version_id"]
    assert first["cloud_skill_id"] == second["cloud_skill_id"]


def test_list_detail_versions_and_content_over_http(client):
    first = client.post("/v1/skills/register", json={"name": "prd-writer", "content": bundle("v1")}).json()["data"]
    second = client.post(
        "/v1/skills/register",
        json={"name": "prd-writer", "content": bundle("v2"), "parent_version_id": first["version_id"]},
    ).json()["data"]

    listing = client.get("/v1/skills")
    assert listing.status_code == 200
    skills = listing.json()["data"]["skills"]
    assert len(skills) == 1
    assert skills[0]["cloud_skill_id"] == first["cloud_skill_id"]
    assert skills[0]["latest_version"]["version_id"] == second["version_id"]
    assert skills[0]["published_head"] is None

    detail = client.post(f"/v1/skills/{first['cloud_skill_id']}/get")
    assert detail.status_code == 200
    assert detail.json()["data"]["latest_version"]["version_id"] == second["version_id"]

    versions = client.get(f"/v1/skills/{first['cloud_skill_id']}/versions")
    assert versions.status_code == 200
    assert [item["version_id"] for item in versions.json()["data"]["versions"]] == [
        first["version_id"],
        second["version_id"],
    ]

    content = client.get(f"/v1/skills/{first['cloud_skill_id']}/versions/{second['version_id']}/content")
    assert content.status_code == 200
    assert content.json()["data"]["version"]["version_id"] == second["version_id"]
    assert content.json()["data"]["content"] == bundle("v2")


def test_sync_reports_no_published_head_over_http(client):
    version = client.post("/v1/skills/register", json={"name": "prd-writer", "content": bundle("v1")}).json()["data"]

    resp = client.post(
        "/v1/skills/sync",
        json=[{"cloud_skill_id": version["cloud_skill_id"], "local_version_id": version["version_id"]}],
    )

    assert resp.status_code == 200
    result = resp.json()["data"]["results"][0]
    assert result["has_update"] is False
    assert result["published_head"] is None
    assert result["gating_status"] == "no_published_head"


def test_delete_unmanages_skill_over_http(client):
    version = client.post("/v1/skills/register", json={"name": "prd-writer", "content": bundle("v1")}).json()["data"]

    resp = client.post(f"/v1/skills/{version['cloud_skill_id']}/delete")
    assert resp.status_code == 200

    detail = client.post(f"/v1/skills/{version['cloud_skill_id']}/get")
    assert detail.status_code == 404
    assert detail.json()["code"] == "skill.not_found"


def test_register_unknown_parent_is_404(client):
    resp = client.post(
        "/v1/skills/register",
        json={"name": "prd-writer", "content": bundle("x"), "parent_version_id": "missing"},
    )
    assert resp.status_code == 404
    assert resp.json()["code"] == "skill.parent_not_found"


def test_register_empty_bundle_is_400(client):
    resp = client.post("/v1/skills/register", json={"name": "x", "content": "[]"})
    assert resp.status_code == 400
    assert resp.json()["code"] == "skill.invalid_bundle"


def test_read_unknown_skill_is_404(client):
    resp = client.post("/v1/skills/missing/get")
    assert resp.status_code == 404
    assert resp.json()["code"] == "skill.not_found"


def test_sync_unknown_skill_is_404(client):
    resp = client.post("/v1/skills/sync", json=[{"cloud_skill_id": "missing", "local_version_id": "v1"}])
    assert resp.status_code == 404
    assert resp.json()["code"] == "skill.not_found"


def test_sync_rejects_object_body(client):
    resp = client.post("/v1/skills/sync", json={"items": []})
    assert resp.status_code == 422


@pytest.mark.parametrize("missing", ["name", "content"])
def test_register_rejects_missing_required_fields(client, missing):
    payload = {"name": "x", "content": bundle("y")}
    del payload[missing]
    resp = client.post("/v1/skills/register", json=payload)
    assert resp.status_code == 422


def _evolve_app(evolver) -> TestClient:
    service = SkillService(store=SkillVersionStore(skill_repo=object(), add_record_repo=object()), evolver=evolver)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_skill_service] = lambda: service
    app.dependency_overrides[get_request_context] = lambda: AuthContext(
        request_id="req-1",
        account_id="acct",
        project_id="proj",
        api_key_uuid="key",
        memory_algorithm="schema",
        scopes=["memory:write"],
    )
    return TestClient(app)


def test_evolve_returns_result_envelope():
    evolver = _StubEvolver(
        SkillEvolveResult(
            cloud_skill_id="cs-1",
            evolved=True,
            pending_count=5,
            threshold=4,
            new_version_id="v-new",
            new_version_ids=["v-new"],
            summarized_count=5,
            consumed_count=5,
        )
    )
    resp = _evolve_app(evolver).post("/v1/skills/evolve", json={"cloud_skill_id": "cs-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == "ok"
    assert body["data"]["evolved"] is True
    assert body["data"]["new_version_id"] == "v-new"
    assert evolver.calls == [("proj", "cs-1")]


def test_evolve_below_threshold_envelope():
    evolver = _StubEvolver(SkillEvolveResult(cloud_skill_id="cs-1", evolved=False, pending_count=2, threshold=4))
    resp = _evolve_app(evolver).post("/v1/skills/evolve", json={"cloud_skill_id": "cs-1"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["evolved"] is False
    assert data["pending_count"] == 2
    assert data["new_version_id"] is None


def test_evolve_async_queues_kafka(monkeypatch):
    evolver = _StubEvolver(SkillEvolveResult(cloud_skill_id="x", evolved=False, pending_count=0, threshold=4))
    producer = AsyncMock()
    monkeypatch.setattr("mindmemos.api.services.skill_service.get_producer", lambda: producer)

    resp = _evolve_app(evolver).post("/v1/skills/evolve", json={"cloud_skill_id": "cs-1", "mode": "async"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == "queued"
    assert body["data"]["status"] == "queued"
    assert evolver.calls == []
    producer.send.assert_awaited_once()
    assert producer.send.call_args.args[0] == SKILL_EVOLVE_TOPIC
    assert producer.send.call_args.kwargs["dispatch_key"] == "proj:cs-1"
    assert producer.send.call_args.kwargs["value"]["cloud_skill_id"] == "cs-1"
    assert producer.send.call_args.kwargs["value"]["project_id"] == "proj"


def test_evolve_requires_cloud_skill_id():
    evolver = _StubEvolver(SkillEvolveResult(cloud_skill_id="x", evolved=False, pending_count=0, threshold=4))
    resp = _evolve_app(evolver).post("/v1/skills/evolve", json={})
    assert resp.status_code == 422
