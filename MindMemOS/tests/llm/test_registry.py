import pytest

from mindmemos.config import init_config, reset_config, update_config
from mindmemos.llm import registry


def test_get_llm_client_creates_new_client_each_call(monkeypatch) -> None:
    created = []
    sentinel_router = object()

    class FakeLLMClient:
        ALIAS = "chat"

        def __init__(self, router, *, default_model=None, max_attempts=3) -> None:
            self.router = router
            created.append(self)

    monkeypatch.setattr(registry, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(registry, "get_router", lambda router_cfg, alias, **kw: (sentinel_router, 0))

    try:
        init_config(config_path="config/mindmemos/dev.example.yaml")

        first = registry.get_llm_client()
        second = registry.get_llm_client()

        # Wrapper is per-request; the Router behind it is the shared/cached one.
        assert first is not second
        assert created == [first, second]
        assert first.router is sentinel_router
        assert second.router is sentinel_router
    finally:
        reset_config()


def test_get_llm_client_uses_request_scoped_config(monkeypatch) -> None:
    seen_models: list[str] = []

    class FakeLLMClient:
        ALIAS = "chat"

        def __init__(self, router, *, default_model=None, max_attempts=3) -> None:
            pass

    def fake_get_router(router_cfg, alias, *, num_retries=None):
        seen_models.append(router_cfg.endpoints[0].model)
        return object(), 0

    monkeypatch.setattr(registry, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(registry, "get_router", fake_get_router)

    try:
        init_config(config_path="config/mindmemos/dev.example.yaml")
        registry.get_llm_client()

        update_config(
            project_config={
                "chat_model_router": {
                    "endpoints": [
                        {
                            "model": "project-chat-model",
                            "api_key": "sk-project",
                            "api_base": "https://project.example.test/v1",
                        }
                    ]
                }
            }
        )
        registry.get_llm_client()

        # Request-scoped override reaches the router factory, which keys its cache on it.
        assert seen_models == ["gpt-4.1-mini", "project-chat-model"]
    finally:
        reset_config()


def test_get_rerank_client_creates_fresh_embed_client(monkeypatch) -> None:
    created_embeds = []

    class FakeEmbedClient:
        ALIAS = "embedding"

        def __init__(self, router, *, default_model=None) -> None:
            created_embeds.append(self)

    class FakeRerankClient:
        ALIAS = "rerank"

        def __init__(self, router, *, embed_client, **kwargs) -> None:
            self.embed_client = embed_client

    monkeypatch.setattr(registry, "EmbedClient", FakeEmbedClient)
    monkeypatch.setattr(registry, "RerankClient", FakeRerankClient)
    monkeypatch.setattr(registry, "get_router", lambda router_cfg, alias, **kw: (object(), 0))

    try:
        init_config(config_path="config/mindmemos/dev.example.yaml")

        first = registry.get_rerank_client()
        second = registry.get_rerank_client()

        assert first.embed_client is not second.embed_client
        assert created_embeds == [first.embed_client, second.embed_client]
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_close_llm_clients_closes_litellm_cache(monkeypatch) -> None:
    calls = []

    async def fake_close() -> None:
        calls.append("closed")

    cleared = []
    monkeypatch.setattr(registry, "clear_router_cache", lambda: cleared.append(True))
    monkeypatch.setattr(registry.litellm, "close_litellm_async_clients", fake_close)
    registry.litellm.aclient_session = object()

    await registry.close_llm_clients()

    assert calls == ["closed"]
    assert cleared == [True]
    assert registry.litellm.aclient_session is None
