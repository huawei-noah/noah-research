from types import SimpleNamespace

import pytest
from mindmemos.errors import EmbeddingDimensionError
from mindmemos.llm.embedding import EmbedClient


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
async def test_embed_raises_dimension_error_when_vector_length_mismatch() -> None:
    client = EmbedClient(FixedDimEmbedRouter(dim=2))

    with pytest.raises(EmbeddingDimensionError) as exc_info:
        await client.embed(task="test", text="hi", expected_dim=1024)

    assert exc_info.value.expected == 1024
    assert exc_info.value.actual == 2
    assert exc_info.value.model == "embedding"
    assert exc_info.value.task == "test"


@pytest.mark.asyncio
async def test_embed_passes_when_dimension_matches() -> None:
    client = EmbedClient(FixedDimEmbedRouter(dim=1024))

    resp = await client.embed(task="test", text="hi", expected_dim=1024)

    assert len(resp.embeddings[0]) == 1024


@pytest.mark.asyncio
async def test_embed_skips_dimension_check_when_expected_dim_unresolved(monkeypatch) -> None:
    # When config is unbound, _resolved_expected_dim returns None and validation
    # is skipped so existing config-less call sites (and tests) keep working.
    monkeypatch.setattr("mindmemos.llm.embedding._resolved_expected_dim", lambda: None)
    client = EmbedClient(FixedDimEmbedRouter(dim=2))

    resp = await client.embed(task="test", text="hi")

    assert len(resp.embeddings[0]) == 2


def test_embedding_dimension_error_message_guides_operator() -> None:
    err = EmbeddingDimensionError(expected=1024, actual=2560, model="qwen", task="startup.probe")
    msg = str(err)

    assert "dimensions" in msg
    assert "drop_params" in msg
    assert "immutable" in msg
    assert "1024" in msg
    assert "2560" in msg
