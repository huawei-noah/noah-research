from types import SimpleNamespace

import pytest
from mindmemos.config import get_config, init_config, reset_config
from mindmemos.errors import EmbeddingDimensionError, InvalidConfigError
from mindmemos.llm import registry
from mindmemos.llm.embedding import EmbedClient
from mindmemos.llm.registry import PROBE_TEXT, validate_embedding_dimension


class FixedDimEmbedRouter:
    """Fake litellm router returning a single vector of a configurable dimension."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    async def aembedding(self, **kwargs):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1] * self.dim)],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=None, total_tokens=1),
            model="embedding",
        )


@pytest.mark.asyncio
async def test_validate_raises_invalid_config_when_dimensions_differ_from_vector_size(monkeypatch) -> None:
    def boom():
        raise AssertionError("get_embed_client must not be called on a static config error")

    monkeypatch.setattr(registry, "get_embed_client", boom)

    try:
        init_config(config_path="config/mindmemos/dev.example.yaml")
        # dev.example.yaml has vector_size=1024; force a mismatching dimensions value.
        get_config().embed_model_router.endpoints[0].dimensions = 512

        with pytest.raises(InvalidConfigError, match="dimensions"):
            await validate_embedding_dimension()
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_validate_probe_raises_when_provider_returns_wrong_dimension(monkeypatch) -> None:
    # Bug scenario: dimensions silently dropped, provider returns native 2560 != vector_size 1024.
    fake_client = EmbedClient(FixedDimEmbedRouter(dim=2560))
    monkeypatch.setattr(registry, "get_embed_client", lambda: fake_client)

    try:
        init_config(config_path="config/mindmemos/dev.example.yaml")
        # dev.example.yaml: dimensions=1024 == vector_size=1024, so static precheck passes
        # and the probe reaches embed(), which measures the actual (wrong) dimension.

        with pytest.raises(EmbeddingDimensionError) as exc_info:
            await validate_embedding_dimension()

        assert exc_info.value.expected == 1024
        assert exc_info.value.actual == 2560
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_validate_skips_probe_when_no_endpoints_configured(monkeypatch) -> None:
    def boom():
        raise AssertionError("get_embed_client must not be called when no endpoints configured")

    monkeypatch.setattr(registry, "get_embed_client", boom)

    try:
        init_config(config_path="config/mindmemos/dev.example.yaml")
        get_config().embed_model_router.endpoints.clear()

        await validate_embedding_dimension()  # no raise, no embed call
    finally:
        reset_config()


def test_probe_text_is_short_constant() -> None:
    assert PROBE_TEXT == "ping"
