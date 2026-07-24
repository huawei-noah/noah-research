from mindmemos.config.app import ModelEndpointConfig, ModelRouterConfig
from mindmemos.llm import router


def test_build_router_assigns_unique_ids_to_duplicate_deployments(monkeypatch) -> None:
    captured = {}

    class FakeRouter:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(router, "Router", FakeRouter)

    cfg = ModelRouterConfig(
        endpoints=[
            ModelEndpointConfig(model="gpt-test", api_key="sk", api_base="https://example.test/v1"),
            ModelEndpointConfig(model="gpt-test", api_key="sk", api_base="https://example.test/v1"),
            ModelEndpointConfig(model="gpt-test", api_key="sk", api_base="https://example.test/v1"),
        ]
    )

    router.build_router(cfg, "chat")

    assert [item["model_info"]["id"] for item in captured["model_list"]] == [
        "gpt-test@https://example.test/v1#1",
        "gpt-test@https://example.test/v1#2",
        "gpt-test@https://example.test/v1#3",
    ]


def test_build_router_passes_endpoint_num_retries_to_litellm_params(monkeypatch) -> None:
    captured = {}

    class FakeRouter:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(router, "Router", FakeRouter)

    cfg = ModelRouterConfig(
        endpoints=[
            ModelEndpointConfig(
                model="gpt-test",
                api_key="sk",
                api_base="https://example.test/v1",
                num_retries=7,
            )
        ]
    )

    router.build_router(cfg, "chat")

    assert captured["num_retries"] == 7
    assert captured["model_list"][0]["litellm_params"]["num_retries"] == 7


def test_litellm_param_fields_do_not_allow_stream() -> None:
    assert "stream" not in router._LITELLM_PARAM_FIELDS


def test_build_litellm_params_keeps_real_endpoint_values() -> None:
    params = router.build_litellm_params(
        ModelEndpointConfig(
            model="cohere/qwen3-reranker",
            api_key="sk-xxx",
            api_base="https://your-rerank-service/v1",
            timeout=600,
        )
    )

    assert params["model"] == "cohere/qwen3-reranker"
    assert params["api_key"] == "sk-xxx"
    assert params["api_base"] == "https://your-rerank-service/v1"
    assert params["timeout"] == 600


def test_get_router_caches_by_resolved_config(monkeypatch) -> None:
    instances = []

    class FakeRouter:
        def __init__(self, **kwargs) -> None:
            instances.append(self)

    monkeypatch.setattr(router, "Router", FakeRouter)
    router.clear_router_cache()

    cfg = ModelRouterConfig(endpoints=[ModelEndpointConfig(model="m", api_key="sk", api_base="https://a.test/v1")])
    try:
        first, _ = router.get_router(cfg, "chat")
        second, _ = router.get_router(cfg, "chat")

        assert first is second
        assert len(instances) == 1
    finally:
        router.clear_router_cache()


def test_get_router_builds_separate_router_for_different_config(monkeypatch) -> None:
    instances = []

    class FakeRouter:
        def __init__(self, **kwargs) -> None:
            instances.append(self)

    monkeypatch.setattr(router, "Router", FakeRouter)
    router.clear_router_cache()

    cfg_a = ModelRouterConfig(endpoints=[ModelEndpointConfig(model="m", api_key="sk", api_base="https://a.test/v1")])
    cfg_b = ModelRouterConfig(endpoints=[ModelEndpointConfig(model="m", api_key="sk", api_base="https://b.test/v1")])
    try:
        router.get_router(cfg_a, "chat")
        router.get_router(cfg_b, "chat")

        assert len(instances) == 2
    finally:
        router.clear_router_cache()


def test_clear_router_cache_forces_rebuild(monkeypatch) -> None:
    instances = []

    class FakeRouter:
        def __init__(self, **kwargs) -> None:
            instances.append(self)

    monkeypatch.setattr(router, "Router", FakeRouter)
    router.clear_router_cache()

    cfg = ModelRouterConfig(endpoints=[ModelEndpointConfig(model="m", api_key="sk", api_base="https://a.test/v1")])
    try:
        router.get_router(cfg, "chat")
        router.clear_router_cache()
        router.get_router(cfg, "chat")

        assert len(instances) == 2
    finally:
        router.clear_router_cache()


def test_build_router_does_not_install_global_litellm_session(monkeypatch) -> None:
    class FakeRouter:
        def __init__(self, **kwargs) -> None:
            pass

    sentinel = object()
    monkeypatch.setattr(router, "Router", FakeRouter)
    monkeypatch.setattr(router.litellm, "aclient_session", sentinel)

    cfg = ModelRouterConfig(
        endpoints=[ModelEndpointConfig(model="gpt-test", api_key="sk", api_base="https://example.test/v1")]
    )

    router.build_router(cfg, "chat")

    assert router.litellm.aclient_session is sentinel


def test_build_litellm_params_injects_allowed_openai_params_for_whitelisted_model() -> None:
    ep = ModelEndpointConfig(
        model="openai/doubao-embedding-large",
        api_key="sk",
        api_base="https://x.test/v1",
        dimensions=1024,
    )

    params = router.build_litellm_params(ep, dimensions_supported_models=["doubao-embedding"])

    assert params["dimensions"] == 1024
    assert params["allowed_openai_params"] == ["dimensions"]


def test_build_litellm_params_omits_allowed_openai_params_for_non_whitelisted_model() -> None:
    ep = ModelEndpointConfig(
        model="openai/some-other-embedding",
        api_key="sk",
        api_base="https://x.test/v1",
        dimensions=1024,
    )

    params = router.build_litellm_params(ep, dimensions_supported_models=["doubao-embedding"])

    assert "allowed_openai_params" not in params


def test_build_litellm_params_omits_allowed_openai_params_when_dimensions_unset() -> None:
    ep = ModelEndpointConfig(
        model="openai/doubao-embedding-large",
        api_key="sk",
        api_base="https://x.test/v1",
    )

    params = router.build_litellm_params(ep, dimensions_supported_models=["doubao-embedding"])

    assert "allowed_openai_params" not in params


def test_build_litellm_params_whitelist_matches_after_provider_prefix_strip() -> None:
    ep = ModelEndpointConfig(
        model="openai/qwen3-embedding-text-002b",
        api_key="sk",
        api_base="https://x.test/v1",
        dimensions=1024,
    )

    params = router.build_litellm_params(ep, dimensions_supported_models=["qwen3"])

    assert params["allowed_openai_params"] == ["dimensions"]


def test_build_router_passes_whitelist_to_endpoint_litellm_params(monkeypatch) -> None:
    captured = {}

    class FakeRouter:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(router, "Router", FakeRouter)

    cfg = ModelRouterConfig(
        endpoints=[
            ModelEndpointConfig(
                model="openai/doubao-embedding-large",
                api_key="sk",
                api_base="https://x.test/v1",
                dimensions=1024,
            )
        ],
        dimensions_supported_models=["doubao-embedding"],
    )

    router.build_router(cfg, "embedding")

    assert captured["model_list"][0]["litellm_params"]["allowed_openai_params"] == ["dimensions"]


def test_get_router_rebuilds_when_whitelist_changes(monkeypatch) -> None:
    instances = []

    class FakeRouter:
        def __init__(self, **kwargs) -> None:
            instances.append(self)

    monkeypatch.setattr(router, "Router", FakeRouter)
    router.clear_router_cache()

    base_ep = ModelEndpointConfig(
        model="openai/doubao-embedding-large",
        api_key="sk",
        api_base="https://x.test/v1",
        dimensions=1024,
    )
    try:
        router.get_router(ModelRouterConfig(endpoints=[base_ep], dimensions_supported_models=[]), "embedding")
        router.get_router(
            ModelRouterConfig(endpoints=[base_ep], dimensions_supported_models=["doubao-embedding"]),
            "embedding",
        )

        # Whitelist change -> different cache key -> rebuilt router.
        assert len(instances) == 2
    finally:
        router.clear_router_cache()
