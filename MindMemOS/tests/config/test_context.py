from mindmemos.config import (
    bind_config_overrides,
    get_config,
    get_config_overrides,
    init_config,
    reset_config,
    update_config,
)
from mindmemos.config import context as config_context


def test_get_config_falls_back_to_global_config_when_request_context_is_empty() -> None:
    try:
        init_config(config_path="config/mindmemos/dev.example.yaml")
        config_context._current.set(None)

        cfg = get_config()

        assert cfg.auth.mode == "api_key"
    finally:
        reset_config()


def test_update_config_tracks_request_overrides() -> None:
    try:
        init_config(config_path="config/mindmemos/dev.example.yaml")

        update_config(
            tenant_config={"pipelines": {"get": "tenant_get"}},
            project_config={"pipelines": {"get": "project_get"}},
        )

        overrides = get_config_overrides()
        assert get_config().pipelines["get"] == "project_get"
        assert overrides is not None
        assert overrides.tenant_config == {"pipelines": {"get": "tenant_get"}}
        assert overrides.project_config == {"pipelines": {"get": "project_get"}}
    finally:
        reset_config()


def test_bind_config_overrides_restores_previous_context() -> None:
    try:
        init_config(config_path="config/mindmemos/dev.example.yaml")
        update_config(project_config={"pipelines": {"get": "outer_get"}})

        with bind_config_overrides(project_config={"pipelines": {"get": "inner_get"}}):
            assert get_config().pipelines["get"] == "inner_get"
            assert get_config_overrides().project_config == {"pipelines": {"get": "inner_get"}}

        assert get_config().pipelines["get"] == "outer_get"
        assert get_config_overrides().project_config == {"pipelines": {"get": "outer_get"}}
    finally:
        reset_config()
