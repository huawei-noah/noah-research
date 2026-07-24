"""Tests for the new ``mindmemos memory`` CLI subcommands.

These cover argument wiring, the status-line formatter, and handler dispatch with
a fake client so no network or local config is touched.
"""

from __future__ import annotations

import json

import pytest
from mindmemos_sdk.skills import RollbackPlan, SkillDiffResult, SkillRecord
from mindmemos_sdk.skills.models import HashState, LocalSkillVersion, SkillOrigin, SkillVersionStatus

from mindmemos_sdk import cli
from mindmemos_sdk.memory import AddResult, DialogueMessage, GetResult, MemorySearchHit, SearchResult, StatusResult


class _FakeMemory:
    """Records the last call and returns a canned result."""

    def __init__(self, result: object) -> None:
        self._result = result
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, method: str, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        return self._result

    def add(self, *args, **kwargs):
        return self._record("add", *args, **kwargs)

    def search(self, *args, **kwargs):
        return self._record("search", *args, **kwargs)

    def get(self, *args, **kwargs):
        return self._record("get", *args, **kwargs)

    def update(self, *args, **kwargs):
        return self._record("update", *args, **kwargs)

    def delete(self, *args, **kwargs):
        return self._record("delete", *args, **kwargs)

    def feedback(self, *args, **kwargs):
        return self._record("feedback", *args, **kwargs)

    def dreaming(self, *args, **kwargs):
        return self._record("dreaming", *args, **kwargs)


class _FakeClient:
    def __init__(self, result: object) -> None:
        self.memory = _FakeMemory(result)

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def fake_client(monkeypatch):
    """Patch ``_build_client`` and expose the fake for assertions."""

    holder: dict[str, _FakeClient] = {}

    def factory(result: object) -> _FakeClient:
        client = _FakeClient(result)
        holder["client"] = client
        monkeypatch.setattr(cli, "_build_client", lambda: client)
        return client

    factory.holder = holder  # type: ignore[attr-defined]
    return factory


def _run(argv: list[str]) -> int:
    return cli.main(argv)


class _FakeSkills:
    def __init__(self, result: object) -> None:
        self._result = result
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, method: str, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        return self._result

    def register(self, *args, **kwargs):
        return self._record("register", *args, **kwargs)

    def list(self):
        return self._record("list")

    def show(self, *args, **kwargs):
        return self._record("show", *args, **kwargs)

    def pull(self, *args, **kwargs):
        return self._record("pull", *args, **kwargs)

    def push(self, *args, **kwargs):
        return self._record("push", *args, **kwargs)

    def plan_update(self, *args, **kwargs):
        return self._record("plan_update", *args, **kwargs)

    def history(self, *args, **kwargs):
        return self._record("history", *args, **kwargs)

    def plan_rollback(self, *args, **kwargs):
        return self._record("plan_rollback", *args, **kwargs)

    def apply_checkout(self, *args, **kwargs):
        return self._record("apply_checkout", *args, **kwargs)

    def diff(self, *args, **kwargs):
        return self._record("diff", *args, **kwargs)

    def unregister(self, *args, **kwargs):
        return self._record("unregister", *args, **kwargs)


@pytest.fixture
def fake_skills(monkeypatch):
    holder: dict[str, _FakeSkills] = {}

    def factory(result: object) -> _FakeSkills:
        manager = _FakeSkills(result)
        holder["manager"] = manager
        monkeypatch.setattr(cli, "_build_skill_manager", lambda *, require_api_key: manager)
        return manager

    factory.holder = holder  # type: ignore[attr-defined]
    return factory


def _skill_record() -> SkillRecord:
    return SkillRecord(
        skill_id="sk_1",
        alias="demo-main",
        path="/tmp/demo",
        skill_name="demo",
        cloud_skill_id="cloud-1",
        base_version_id="v1",
        content_hash="hash-1",
        hash_state=HashState.CONFIRMED,
        version_label="1.0.0",
        updated_at="2026-06-16T00:00:00Z",
    )


def _skill_record_at(path: str) -> SkillRecord:
    return _skill_record().model_copy(update={"path": path})


def _rollback_plan() -> RollbackPlan:
    return RollbackPlan(
        skill_id="sk_1",
        path="/tmp/demo",
        from_version_id="v2",
        to_version_id="v1",
        from_content_hash="hash-2",
        to_content_hash="hash-1",
        files=["SKILL.md"],
        backup_path="/tmp/backups/demo/20260616T000000Z",
    )


def test_status_line_includes_target_message_and_request_id():
    result = StatusResult(code="ok", request_id="req-9", message="done")
    line = cli._status_line("Updated", "m1", result)
    assert line == "Updated m1. done (request_id=req-9)"


def test_status_line_without_target_or_extras():
    assert cli._status_line("Dreaming triggered", None, StatusResult()) == "Dreaming triggered."


def test_skill_register_prints_saved_record(fake_skills, capsys):
    fake_skills(_skill_record())

    rc = _run(
        ["skill", "register", "/tmp/demo/SKILL.md", "--name", "demo2", "--alias", "demo-main", "--version", "2.0.0"]
    )

    assert rc == 0
    assert fake_skills.holder["manager"].calls[0] == (
        "register",
        ("/tmp/demo/SKILL.md",),
        {"name": "demo2", "version_label": "2.0.0", "alias": "demo-main"},
    )
    out = capsys.readouterr().out
    assert "Registered demo" in out
    assert "alias:          demo-main" in out


def test_skill_list_and_show(fake_skills, capsys):
    fake_skills([_skill_record()])

    assert _run(["skill", "list"]) == 0
    out = capsys.readouterr().out
    assert "skill_id  alias      name  base_version_id  cloud_skill_id  hash_state  path" in out
    assert "sk_1      demo-main  demo  v1               cloud-1         confirmed   /tmp/demo" in out

    fake_skills(_skill_record())
    assert _run(["skill", "show", "demo-main"]) == 0
    out = capsys.readouterr().out
    assert "skill_id:       sk_1" in out
    assert "alias:          demo-main" in out
    assert "hash_state:     confirmed" in out


def test_skill_pull_and_history(fake_skills, capsys):
    version = LocalSkillVersion(
        version_id="v2",
        parent_version_id="v1",
        version_label="1.1.0",
        status=SkillVersionStatus.PUBLISHED,
        origin=SkillOrigin.CLOUD,
        content_hash="hash-2",
        created_at="2026-06-16T00:01:00Z",
    )
    fake_skills([version])

    assert _run(["skill", "pull", "sk_1"]) == 0
    assert "Pulled 1 version(s)." in capsys.readouterr().out

    fake_skills([version])
    assert _run(["skill", "history", "sk_1"]) == 0
    assert "v2 parent=v1 status=published" in capsys.readouterr().out


def test_skill_push_prints_new_version(fake_skills, capsys):
    fake_skills(_skill_record().model_copy(update={"base_version_id": "v2", "content_hash": "hash-2"}))

    rc = _run(["skill", "push", "demo-main"])

    assert rc == 0
    assert fake_skills.holder["manager"].calls == [("push", ("demo-main",), {})]
    out = capsys.readouterr().out
    assert "Pushed demo (sk_1) to v2." in out
    assert "content_hash: hash-2" in out


def test_skill_update_with_yes_applies_checkout(fake_skills, capsys):
    manager = fake_skills(_rollback_plan())

    def apply_checkout(plan):
        manager.calls.append(("apply_checkout", (plan,), {}))
        return _skill_record().model_copy(update={"base_version_id": "v2"})

    manager.apply_checkout = apply_checkout

    rc = _run(["skill", "update", "sk_1", "--yes"])

    assert rc == 0
    assert manager.calls[0] == ("plan_update", ("sk_1",), {})
    assert manager.calls[1][0] == "apply_checkout"
    assert "Updated demo (sk_1) to v2." in capsys.readouterr().out


def test_skill_rollback_requires_confirmation(monkeypatch, fake_skills, capsys):
    fake_skills(_rollback_plan())
    monkeypatch.setattr(cli, "_prompt", lambda _msg: "n")

    rc = _run(["skill", "rollback", "sk_1", "--to", "v1"])

    assert rc == 1
    assert fake_skills.holder["manager"].calls == [
        ("plan_rollback", ("sk_1",), {"version_id": "v1"}),
    ]
    out = capsys.readouterr().out
    assert "Rollback plan:" in out
    assert "Aborted." in out


def test_skill_rollback_with_yes_applies_checkout(fake_skills, capsys):
    manager = fake_skills(_rollback_plan())

    def apply_checkout(plan):
        manager.calls.append(("apply_checkout", (plan,), {}))
        return _skill_record().model_copy(update={"base_version_id": "v1"})

    manager.apply_checkout = apply_checkout

    rc = _run(["skill", "rollback", "sk_1", "--to", "v1", "--yes"])

    assert rc == 0
    assert manager.calls[0] == ("plan_rollback", ("sk_1",), {"version_id": "v1"})
    assert manager.calls[1][0] == "apply_checkout"
    assert "Rolled back demo (sk_1) to v1." in capsys.readouterr().out


def test_skill_diff_prints_unified_diff(fake_skills, capsys):
    fake_skills(
        SkillDiffResult(
            skill_id="sk_1",
            from_version_id="v1",
            to_version_id="v2",
            diff="--- v1/SKILL.md\n+++ v2/SKILL.md\n+new\n",
        )
    )

    rc = _run(["skill", "diff", "sk_1", "--from", "v1", "--to", "v2"])

    assert rc == 0
    assert fake_skills.holder["manager"].calls == [
        ("diff", ("sk_1",), {"from_version_id": "v1", "to_version_id": "v2"}),
    ]
    assert "+new" in capsys.readouterr().out


def test_skill_unregister_requires_confirmation(monkeypatch, fake_skills, capsys):
    fake_skills(_skill_record())
    monkeypatch.setattr(cli, "_prompt", lambda _msg: "n")

    rc = _run(["skill", "unregister", "sk_1"])

    assert rc == 1
    assert fake_skills.holder["manager"].calls == []
    assert "Aborted." in capsys.readouterr().out


def test_skill_unregister_with_yes(fake_skills, capsys):
    fake_skills(_skill_record())

    rc = _run(["skill", "unregister", "sk_1", "--yes"])

    assert rc == 0
    assert fake_skills.holder["manager"].calls == [("unregister", ("sk_1",), {})]
    assert "Unregistered demo" in capsys.readouterr().out


def test_skill_unregister_delete_files_with_yes(fake_skills, tmp_path, capsys):
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("name: demo\n", encoding="utf-8")
    fake_skills(_skill_record_at(str(skill_dir)))

    rc = _run(["skill", "unregister", "sk_1", "--delete-files", "--yes"])

    assert rc == 0
    assert not skill_dir.exists()
    assert "Deleted" in capsys.readouterr().out


def test_memory_add_builds_dialogue_message(fake_client):
    fake_client(AddResult(code="ok"))
    rc = _run(
        [
            "memory",
            "add",
            "--content",
            "我喜欢咖啡",
            "--role",
            "assistant",
            "--user-id",
            "u1",
            "--app-id",
            "app1",
            "--agent-id",
            "agent1",
            "--session-id",
            "s1",
            "--metadata-json",
            '{"source": "cli"}',
        ]
    )
    assert rc == 0
    calls = fake_client.holder["client"].memory.calls
    assert len(calls) == 1
    name, _args, kwargs = calls[0]
    assert name == "add"
    assert kwargs["mode"] == "sync"
    assert kwargs["user_id"] == "u1"
    assert kwargs["app_id"] == "app1"
    assert kwargs["agent_id"] == "agent1"
    assert kwargs["session_id"] == "s1"
    assert kwargs["metadata"] == {"source": "cli"}
    (message,) = kwargs["messages"]
    assert isinstance(message, DialogueMessage)
    assert message.role == "assistant"
    assert message.content == "我喜欢咖啡"
    assert message.timestamp > 0


def test_memory_add_accepts_skill_context_json(fake_client):
    fake_client(AddResult(code="ok"))
    context = [{"name": "demo", "content_hash": "hash-1", "base_version_id": "v1"}]

    rc = _run(["memory", "add", "--content", "hi", "--skill-context-json", json.dumps(context)])

    assert rc == 0
    _name, _args, kwargs = fake_client.holder["client"].memory.calls[0]
    assert kwargs["skill_context"] == context


def test_memory_add_defaults_role_to_user(fake_client):
    fake_client(AddResult(code="ok"))
    rc = _run(["memory", "add", "--content", "hi", "--async"])
    assert rc == 0
    _name, _args, kwargs = fake_client.holder["client"].memory.calls[0]
    assert kwargs["mode"] == "async"
    assert kwargs["messages"][0].role == "user"


def test_memory_add_accepts_messages_json(fake_client):
    fake_client(AddResult(code="queued", request_id="req-add"))
    messages = [{"role": "user", "content": "hi", "timestamp": 1700000000000}]

    rc = _run(["memory", "add", "--messages-json", json.dumps(messages), "--async", "--json"])

    assert rc == 0
    _name, _args, kwargs = fake_client.holder["client"].memory.calls[0]
    assert kwargs["mode"] == "async"
    assert kwargs["messages"] == messages


def test_memory_add_rejects_invalid_metadata_json(capsys):
    rc = _run(["memory", "add", "--content", "hi", "--metadata-json", "[1, 2]"])
    assert rc == 2
    assert "--metadata-json must be a JSON object" in capsys.readouterr().out


def test_memory_add_requires_content_or_messages_json(capsys):
    rc = _run(["memory", "add"])
    assert rc == 2
    assert "either --content or --messages-json is required" in capsys.readouterr().out


def test_memory_add_rejects_invalid_messages_json(capsys):
    rc = _run(["memory", "add", "--messages-json", '{"role": "user"}'])
    assert rc == 2
    assert "messages JSON must be a non-empty JSON array" in capsys.readouterr().out


def test_memory_search_json_output(fake_client, capsys):
    fake_client(SearchResult(memories=[MemorySearchHit(id="m1", memory="likes tea")]))

    rc = _run(
        [
            "memory",
            "search",
            "tea",
            "--top-k",
            "4",
            "--user-id",
            "u1",
            "--app-id",
            "app1",
            "--agent-id",
            "agent1",
            "--session-id",
            "s1",
            "--search-strategy",
            "agentic",
            "--rerank",
            "--filter",
            '{"memory_type": "semantic"}',
            "--json",
        ]
    )

    assert rc == 0
    assert fake_client.holder["client"].memory.calls == [
        (
            "search",
            ("tea",),
            {
                "top_k": 4,
                "user_id": "u1",
                "search_strategy": "agentic",
                "rerank": True,
                "filters": {"memory_type": "semantic"},
                "app_id": "app1",
                "agent_id": "agent1",
                "session_id": "s1",
            },
        )
    ]
    payload = json.loads(capsys.readouterr().out)
    assert payload["memories"][0]["id"] == "m1"


def test_memory_search_invalid_filter_json_fails_fast(capsys):
    rc = _run(["memory", "search", "tea", "--filter", "{not json}"])
    assert rc == 2
    assert "invalid --filter JSON" in capsys.readouterr().out


def test_memory_update_invokes_client(fake_client, capsys):
    fake_client(StatusResult(code="ok", request_id="req-up"))
    rc = _run(["memory", "update", "m1", "--content", "new text"])
    assert rc == 0
    client = fake_client.holder["client"]
    assert client.memory.calls == [("update", ("m1", "new text"), {})]
    assert "Updated m1." in capsys.readouterr().out


def test_memory_delete_requires_confirmation(monkeypatch, fake_client, capsys):
    fake_client(StatusResult())
    monkeypatch.setattr(cli, "_prompt", lambda _msg: "n")
    rc = _run(["memory", "delete", "m1"])
    assert rc == 1
    # Declining the prompt must not hit the client.
    assert fake_client.holder["client"].memory.calls == []
    assert "Aborted." in capsys.readouterr().out


def test_memory_delete_with_yes_skips_prompt(fake_client, capsys):
    fake_client(StatusResult(code="ok"))
    rc = _run(["memory", "delete", "m1", "--yes"])
    assert rc == 0
    assert fake_client.holder["client"].memory.calls == [("delete", ("m1",), {})]
    assert "Deleted m1." in capsys.readouterr().out


def test_memory_feedback_passes_text(fake_client):
    fake_client(StatusResult(code="ok"))
    rc = _run(["memory", "feedback", "--text", "good"])
    assert rc == 0
    assert fake_client.holder["client"].memory.calls == [("feedback", (), {"feedback": "good"})]


def test_memory_dreaming_invokes_client(fake_client, capsys):
    fake_client(StatusResult(code="ok"))
    rc = _run(["memory", "dreaming"])
    assert rc == 0
    assert fake_client.holder["client"].memory.calls == [("dreaming", (), {"mode": "async"})]
    assert "Dreaming triggered." in capsys.readouterr().out


def test_memory_dreaming_sync_passes_mode(fake_client):
    fake_client(StatusResult(code="ok"))
    rc = _run(["memory", "dreaming", "--sync"])
    assert rc == 0
    assert fake_client.holder["client"].memory.calls == [("dreaming", (), {"mode": "sync"})]


def test_memory_dreaming_rejects_conflicting_modes(fake_client, capsys):
    fake_client(StatusResult(code="ok"))
    rc = _run(["memory", "dreaming", "--sync", "--async"])
    assert rc == 2
    assert fake_client.holder["client"].memory.calls == []
    assert "--sync and --async are mutually exclusive" in capsys.readouterr().err


def test_memory_get_passes_filter_and_top_k(fake_client):
    fake_client(GetResult(memories=[MemorySearchHit(id="m1", memory="cat")]))
    rc = _run(["memory", "get", "--filter", '{"app_id": "a1"}', "--top-k", "3"])
    assert rc == 0
    assert fake_client.holder["client"].memory.calls == [
        ("get", (), {"filters": {"app_id": "a1"}, "top_k": 3})
    ]


def test_memory_get_invalid_filter_json_fails_fast(capsys):
    rc = _run(["memory", "get", "--filter", "{not json}"])
    assert rc == 2
    assert "invalid --filter JSON" in capsys.readouterr().out


def test_memory_get_non_object_filter_rejected(capsys):
    rc = _run(["memory", "get", "--filter", "[1, 2]"])
    assert rc == 2
    assert "must be a JSON object" in capsys.readouterr().out
