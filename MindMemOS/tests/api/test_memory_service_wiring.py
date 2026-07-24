from pathlib import Path
from types import SimpleNamespace

import pytest
from mindmemos.api.schemas import AddRequest, AuthContext
from mindmemos.pipelines.delete.default import DefaultDeletePipeline
from mindmemos.pipelines.feedback.default import DefaultFeedbackPipeline
from mindmemos.pipelines.get.default import DefaultGetPipeline
from mindmemos.pipelines.registry import register
from mindmemos.pipelines.search.pipeline import SearchPipelineImpl
from mindmemos.pipelines.update.default import DefaultUpdatePipeline
from mindmemos.typing.memory import DialogueMessage, MemoryRequestContext
from mindmemos.typing.service import AddPipelineInput, AddPipelineSyncResult, MemoryAddEventItem

from mindmemos.api.services import memory_service
from mindmemos.config import init_config, reset_config


def test_get_memory_service_wires_configured_non_algorithm_pipelines() -> None:
    try:
        init_config(config_path="config/mindmemos/dev.example.yaml")
        memory_service._service = None

        service = memory_service.get_memory_service()

        assert service._add is None
        assert service._search is None
        assert service._get is None
        assert service._delete is None
        assert service._update is None
        assert service._feedback is None
        assert service._pipeline("_add") is None
        assert isinstance(service._pipeline("_search"), SearchPipelineImpl)
        assert isinstance(service._pipeline("_get"), DefaultGetPipeline)
        assert isinstance(service._pipeline("_delete"), DefaultDeletePipeline)
        assert isinstance(service._pipeline("_update"), DefaultUpdatePipeline)
        assert isinstance(service._pipeline("_feedback"), DefaultFeedbackPipeline)
    finally:
        memory_service._service = None
        reset_config()


def test_checked_in_dev_config_loads_and_selects_default_search_pipelines() -> None:
    try:
        init_config(config_path="config/mindmemos/dev.yaml")
        memory_service._service = None

        service = memory_service.get_memory_service()

        assert isinstance(service._pipeline("_search"), SearchPipelineImpl)
    finally:
        memory_service._service = None
        reset_config()


def test_dev_config_does_not_select_add_algorithm(tmp_path) -> None:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
pipelines:
  get: default_get
  delete: default_delete
  update: default_update
  feedback: default_feedback
""",
        encoding="utf-8",
    )

    try:
        init_config(config_path=config_path)
        memory_service._service = None

        service = memory_service.get_memory_service()

        assert service._add is None
        assert service._pipeline("_add") is None
        assert isinstance(service._pipeline("_search"), SearchPipelineImpl)
    finally:
        memory_service._service = None
        reset_config()


def test_get_memory_service_selects_non_algorithm_pipelines_from_config(tmp_path) -> None:
    @register(type="get", name="custom_get")
    class CustomGetPipeline:
        pass

    @register(type="delete", name="custom_delete")
    class CustomDeletePipeline:
        pass

    @register(type="update", name="custom_update")
    class CustomUpdatePipeline:
        pass

    @register(type="feedback", name="custom_feedback")
    class CustomFeedbackPipeline:
        pass

    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
pipelines:
  get: custom_get
  delete: custom_delete
  update: custom_update
  feedback: custom_feedback
""",
        encoding="utf-8",
    )

    try:
        init_config(config_path=config_path)
        memory_service._service = None

        service = memory_service.get_memory_service()

        assert service._pipeline("_add") is None
        assert isinstance(service._pipeline("_search"), SearchPipelineImpl)
        assert isinstance(service._pipeline("_get"), CustomGetPipeline)
        assert isinstance(service._pipeline("_delete"), CustomDeletePipeline)
        assert isinstance(service._pipeline("_update"), CustomUpdatePipeline)
        assert isinstance(service._pipeline("_feedback"), CustomFeedbackPipeline)
    finally:
        memory_service._service = None
        reset_config()


def test_get_memory_service_rejects_unknown_pipeline_name_on_use(tmp_path) -> None:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
pipelines:
  get: missing_get
""",
        encoding="utf-8",
    )

    try:
        init_config(config_path=config_path)
        memory_service._service = None

        service = memory_service.get_memory_service()
        with pytest.raises(ValueError, match="Unknown get pipeline"):
            service._pipeline("_get")
    finally:
        memory_service._service = None
        reset_config()


@pytest.mark.asyncio
async def test_add_request_does_not_eagerly_create_unrelated_feedback_pipeline(tmp_path, monkeypatch) -> None:
    @register(type="add", name="isolated_add_for_wiring_test")
    class IsolatedAddPipeline:
        async def add_sync(
            self, inp: AddPipelineInput, context: MemoryRequestContext, *, add_record_id: str | None = None
        ) -> AddPipelineSyncResult:
            return AddPipelineSyncResult(
                status="ok",
                memories=[MemoryAddEventItem(operation="add", content=inp.messages[0].content)],
            )

        async def add_async(
            self, inp: AddPipelineInput, context: MemoryRequestContext, *, add_record_id: str | None = None
        ):
            raise AssertionError("unexpected async add")

        async def has_pending(self, ctx: MemoryRequestContext) -> bool:
            return False

    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
pipelines:
  get: default_get
  delete: default_delete
  update: default_update
  feedback: missing_feedback
""",
        encoding="utf-8",
    )

    try:
        init_config(config_path=config_path)
        memory_service._service = None
        monkeypatch.setattr(
            memory_service,
            "binding_for_memory_algorithm",
            lambda _algorithm: SimpleNamespace(add_pipeline="isolated_add_for_wiring_test", search_pipeline="schema"),
        )
        service = memory_service.get_memory_service()

        result = await service.add(
            AuthContext(
                request_id="00000000-0000-0000-0000-000000000001",
                account_id="acc-1",
                project_id="proj-1",
                api_key_uuid="key-1",
                memory_algorithm="schema",
            ),
            AddRequest(
                user_id="user-1",
                session_id="session-1",
                messages=[DialogueMessage(role="user", content="remember local wheels", timestamp=1770000000000)],
            ),
        )

        assert result.status == "ok"
        assert result.memories[0].content == "remember local wheels"
        assert service._feedback is None
    finally:
        memory_service._service = None
        reset_config()


def test_memory_api_does_not_expose_project_clear_route() -> None:
    routes_source = (Path(memory_service.__file__).resolve().parents[1] / "routes.py").read_text(encoding="utf-8")

    assert '@router.post("/clear"' not in routes_source
    assert "ClearResponse" not in routes_source


def test_memory_service_does_not_expose_project_clear_facade() -> None:
    assert not hasattr(memory_service.MemoryService, "clear")
