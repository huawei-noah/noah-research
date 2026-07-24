"""Tests for HttpTransport envelope handling and MemoryClient add/search."""

from __future__ import annotations

import json

import httpx
import pytest
from mindmemos_sdk.errors import ApiError, AuthRequiredError, MindMemOSSDKError, TransportError
from mindmemos_sdk.memory import (
    DialogueMessage,
    FileMessage,
    MemoryClient,
    TextMessage,
    UrlMessage,
)
from mindmemos_sdk.skills import HashState, SkillContext, SkillRecord
from mindmemos_sdk.transport import HttpTransport


def _transport(handler, *, api_key="mk_test", **kwargs) -> HttpTransport:
    """构造一个走 MockTransport 的 HttpTransport。"""
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return HttpTransport(base_url="https://api.test", api_key=api_key, client=client, **kwargs)


def test_add_sync_returns_memories():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "code": "ok",
                "message": "",
                "request_id": "req-1",
                "data": {"memories": [{"operation": "add", "content": "hello", "memory_id": "m1"}]},
            },
        )

    client = MemoryClient(_transport(handler), default_user_id="u_1")
    result = client.add(messages=[TextMessage(text="hello")])

    assert captured["url"] == "https://api.test/v1/memory/add"
    assert captured["auth"] == "Bearer mk_test"
    assert captured["body"]["user_id"] == "u_1"
    assert captured["body"]["messages"] == [{"text": "hello"}]
    assert captured["body"]["mode"] == "sync"
    assert result.code == "ok"
    assert result.request_id == "req-1"
    assert len(result.memories) == 1
    assert result.memories[0].memory_id == "m1"


def test_add_typed_messages_serialized():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "ok", "data": {"memories": []}})

    client = MemoryClient(_transport(handler), default_user_id="u_1")
    client.add(
        messages=[
            DialogueMessage(role="user", content="hi", timestamp=1700000000000),
            UrlMessage(url="https://example.com"),
            FileMessage(file_name="a.pdf", file_path="/tmp/a.pdf"),
            TextMessage(text="note"),
            {"text": "raw-dict-still-ok"},
        ],
    )

    msgs = captured["body"]["messages"]
    assert msgs[0] == {"role": "user", "content": "hi", "timestamp": 1700000000000}
    assert msgs[1] == {"url": "https://example.com"}
    assert msgs[2] == {"file_name": "a.pdf", "file_path": "/tmp/a.pdf", "file_type": ""}
    assert msgs[3] == {"text": "note"}
    assert msgs[4] == {"text": "raw-dict-still-ok"}


def test_add_sends_skill_context_when_provided():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "ok", "data": {"memories": []}})

    client = MemoryClient(_transport(handler), default_user_id="u_1")
    client.add(
        messages=[TextMessage(text="hello")],
        skill_context=[
            {
                "name": "demo",
                "content_hash": "hash-1",
                "base_version_id": "v1",
                "usage": "modified",
            }
        ],
    )

    assert captured["body"]["skill_context"] == [
        {
            "name": "demo",
            "content_hash": "hash-1",
            "base_version_id": "v1",
            "usage": "modified",
        }
    ]


def test_add_sends_score_and_task_id_when_provided():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "ok", "data": {"memories": []}})

    client = MemoryClient(_transport(handler), default_user_id="u_1")
    client.add(
        messages=[TextMessage(text="hello")],
        score=0.75,
        task_id="task-1",
    )

    assert captured["body"]["score"] == 0.75
    assert captured["body"]["task_id"] == "task-1"


def test_dialogue_message_accepts_named_speaker_without_timestamp():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "ok", "data": {"memories": []}})

    client = MemoryClient(_transport(handler), default_user_id="u_1")
    client.add(messages=[DialogueMessage(role="Melanie", content="hi")])

    assert captured["body"]["messages"] == [{"role": "Melanie", "content": "hi", "timestamp": None}]


def test_add_auto_detects_and_ensures_registered_skill(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "ok", "data": {"memories": []}})

    class FakeRegistry:
        def list(self):
            return [
                SkillRecord(
                    skill_id="sk_1",
                    path=str(tmp_path / "demo"),
                    skill_name="demo",
                    cloud_skill_id="cloud-1",
                    base_version_id="v1",
                    content_hash="hash-1",
                    hash_state=HashState.CONFIRMED,
                )
            ]

        def get_by_path(self, _path):
            return self.list()[0]

    class FakeSkills:
        registry = FakeRegistry()

        def __init__(self):
            self.flush_called = False

        def skill_id_for_context(self, context):
            return "sk_1" if context.name == "demo" else None

        def ensure_skill_context(self, skill_id, *, usage=None):
            assert skill_id == "sk_1"
            return SkillContext(name="demo", content_hash="hash-confirmed", base_version_id="v1", usage=usage)

        def flush_pending_uploads(self):
            self.flush_called = True
            return []

    skills = FakeSkills()
    client = MemoryClient(_transport(handler), default_user_id="u_1", skill_manager=skills)
    skill_path = tmp_path / "demo" / "SKILL.md"
    client.add(
        messages=[
            DialogueMessage(role="assistant", content=f'[tool_call] read({{"path":"{skill_path}"}})', timestamp=1),
            DialogueMessage(role="tool", content="name: demo\n\nBody\n", timestamp=2),
        ]
    )

    assert captured["body"]["skill_context"] == [
        {
            "name": "demo",
            "content_hash": "hash-confirmed",
            "base_version_id": "v1",
            "usage": "injected",
        }
    ]
    assert skills.flush_called is True


def test_add_async_queued_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "queued", "request_id": "req-2", "data": {"memories": []}})

    client = MemoryClient(_transport(handler), default_user_id="u_1")
    result = client.add(messages=[TextMessage(text="hi")], mode="async")
    assert result.code == "queued"
    assert result.memories == []


def test_search_returns_hits_and_sends_params():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "code": "ok",
                "request_id": "req-3",
                "data": {
                    "memories": [
                        {
                            "id": "m1",
                            "memory": "cat",
                            "last_update_at": "2026-06-11 00:00:00",
                            "lineage": {
                                "role": "current",
                                "derived_from_memory_ids": ["old-m1"],
                                "derived_to_memory_ids": [],
                            },
                        }
                    ]
                },
            },
        )

    client = MemoryClient(_transport(handler), default_user_id="u_1")
    result = client.search("pets", top_k=5)
    assert captured["body"]["query"] == "pets"
    assert captured["body"]["top_k"] == 5
    assert captured["body"]["user_id"] == "u_1"
    assert result.memories[0].id == "m1"
    assert result.memories[0].memory == "cat"
    assert result.memories[0].lineage is not None
    assert result.memories[0].lineage.role == "current"
    assert result.memories[0].lineage.derived_from_memory_ids == ["old-m1"]


def test_get_sends_body_and_returns_hits():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "code": "ok",
                "request_id": "req-get",
                "data": {"memories": [{"id": "m1", "memory": "cat", "last_update_at": "2026-06-11 00:00:00"}]},
            },
        )

    client = MemoryClient(_transport(handler), default_user_id="u_1")
    result = client.get(filters={"app_id": "a1"}, top_k=5)

    assert captured["url"] == "https://api.test/v1/memory/get"
    # get carries no actor identity; only filters/top_k.
    assert captured["body"] == {"filters": {"app_id": "a1"}, "top_k": 5}
    assert result.request_id == "req-get"
    assert result.memories[0].id == "m1"


def test_get_omits_empty_optional_fields():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "ok", "data": {"memories": []}})

    client = MemoryClient(_transport(handler))
    client.get()
    assert captured["body"] == {}


def test_update_sends_body_and_returns_status():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "ok", "request_id": "req-up", "data": None})

    client = MemoryClient(_transport(handler))
    result = client.update("m1", "new content")

    assert captured["url"] == "https://api.test/v1/memory/update"
    assert captured["body"] == {"memory_id": "m1", "content": "new content"}
    assert result.code == "ok"
    assert result.request_id == "req-up"


def test_delete_sends_memory_id():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "ok", "data": None})

    client = MemoryClient(_transport(handler))
    result = client.delete("m1")

    assert captured["url"] == "https://api.test/v1/memory/delete"
    assert captured["body"] == {"memory_id": "m1"}
    assert result.code == "ok"


def test_delete_soft_error_raises_api_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "error", "message": "memory not found", "data": None})

    client = MemoryClient(_transport(handler))
    with pytest.raises(ApiError) as ei:
        client.delete("missing")
    assert ei.value.code == "error"
    assert "not found" in str(ei.value)


def test_feedback_omits_none_text():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "ok", "data": None})

    client = MemoryClient(_transport(handler))
    client.feedback()
    assert captured["url"] == "https://api.test/v1/memory/feedback"
    assert captured["body"] == {}


def test_feedback_sends_explicit_text():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "ok", "message": "thanks", "data": None})

    client = MemoryClient(_transport(handler))
    result = client.feedback(feedback="great recall")
    assert captured["body"] == {"feedback": "great recall"}
    assert result.message == "thanks"


def test_feedback_sends_actor_identity():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "ok", "data": None})

    client = MemoryClient(_transport(handler), default_user_id="u-default", default_app_id="app-default")
    client.feedback(feedback="great recall", agent_id="agent-1", session_id="session-1")

    assert captured["body"] == {
        "user_id": "u-default",
        "app_id": "app-default",
        "agent_id": "agent-1",
        "session_id": "session-1",
        "feedback": "great recall",
    }


def test_dreaming_sends_async_mode_by_default():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "ok", "data": None})

    client = MemoryClient(_transport(handler))
    result = client.dreaming()
    assert captured["url"] == "https://api.test/v1/memory/dreaming"
    assert captured["body"] == {"mode": "async"}
    assert result.code == "ok"


def test_dreaming_sends_sync_mode():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "ok", "data": None})

    client = MemoryClient(_transport(handler))
    result = client.dreaming(mode="sync")
    assert captured["body"] == {"mode": "sync"}
    assert result.code == "ok"


def test_dreaming_sends_actor_identity():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "ok", "data": None})

    client = MemoryClient(_transport(handler), default_user_id="u-default", default_app_id="app-default")
    client.dreaming(agent_id="agent-1", session_id="session-1")

    assert captured["body"] == {
        "mode": "async",
        "user_id": "u-default",
        "app_id": "app-default",
        "agent_id": "agent-1",
        "session_id": "session-1",
    }


def test_add_requires_user_id():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - not reached
        return httpx.Response(200, json={"code": "ok", "data": {}})

    client = MemoryClient(_transport(handler))
    with pytest.raises(MindMemOSSDKError):
        client.add(messages=[TextMessage(text="x")])


def test_add_rejects_empty_messages():
    client = MemoryClient(_transport(lambda r: httpx.Response(200, json={"code": "ok"})), default_user_id="u")
    with pytest.raises(MindMemOSSDKError):
        client.add(messages=[])


def test_user_id_override():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "ok", "data": {"memories": []}})

    client = MemoryClient(_transport(handler), default_user_id="u_default")
    client.search("q", user_id="u_override")
    assert captured["body"]["user_id"] == "u_override"


def test_error_envelope_raises_api_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"code": "memory.not_found", "message": "nope", "request_id": "req-err", "data": None},
        )

    client = MemoryClient(_transport(handler), default_user_id="u_1")
    with pytest.raises(ApiError) as ei:
        client.search("q")
    assert ei.value.code == "memory.not_found"
    assert ei.value.request_id == "req-err"
    assert "nope" in str(ei.value)


def test_http_500_raises_api_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"code": "error", "message": "boom"})

    client = MemoryClient(_transport(handler), default_user_id="u_1")
    with pytest.raises(ApiError) as ei:
        client.search("q")
    assert ei.value.status_code == 500


def test_missing_api_key_raises_auth_required():
    transport = HttpTransport(base_url="https://api.test", api_key=None)
    with pytest.raises(AuthRequiredError):
        transport.post_envelope("/v1/memory/search", json={})


def test_transport_get_envelope_sends_params_and_headers():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["request_id"] = request.headers.get("X-Request-Id")
        return httpx.Response(200, json={"code": "ok", "request_id": "req-get", "data": {"items": [1]}})

    envelope = _transport(handler).get_envelope(
        "/v1/skills/cloud-1/versions",
        params={"since": "2026-06-16T00:00:00Z"},
        request_id="req-client",
    )

    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.test/v1/skills/cloud-1/versions?since=2026-06-16T00%3A00%3A00Z"
    assert captured["auth"] == "Bearer mk_test"
    assert captured["request_id"] == "req-client"
    assert envelope.data == {"items": [1]}


def test_transport_delete_envelope_sends_no_body_by_default():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = request.content
        return httpx.Response(200, json={"code": "ok", "data": None})

    envelope = _transport(handler).delete_envelope("/v1/skills/cloud-1")

    assert captured["method"] == "DELETE"
    assert captured["url"] == "https://api.test/v1/skills/cloud-1"
    assert captured["body"] == b""
    assert envelope.code == "ok"


def test_transport_post_envelope_accepts_top_level_array_json():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": "ok", "data": {"results": []}})

    envelope = _transport(handler).post_envelope(
        "/v1/skills/sync",
        json=[{"cloud_skill_id": "cloud-1", "local_version_id": "version-1"}],
    )

    assert captured["body"] == [{"cloud_skill_id": "cloud-1", "local_version_id": "version-1"}]
    assert envelope.data == {"results": []}


def test_network_error_retries_then_transport_error():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("down")

    client = MemoryClient(_transport(handler, max_retries=2), default_user_id="u_1")
    with pytest.raises(TransportError):
        client.search("q")
    assert calls["n"] == 3  # initial try + 2 retries
