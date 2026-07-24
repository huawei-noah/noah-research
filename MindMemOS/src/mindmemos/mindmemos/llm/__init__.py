"""LLM clients built on top of litellm.Router.

Public API:
- ``get_llm_client()`` -> ``LLMClient`` for chat calls.
- ``get_embed_client()`` -> ``EmbedClient`` for embedding calls.
- ``get_rerank_client()`` -> ``RerankClient`` for reranking calls.

Implementation lives in focused submodules instead of this package initializer:
``chat`` (chat client), ``embedding`` (embedding client), ``rerank`` (rerank
client), ``router`` (litellm Router construction), and ``registry`` (factory
helpers and LiteLLM lifecycle cleanup).
"""

from ..typing import ChatResponse, EmbeddingResponse, RerankHit, RerankResponse, Usage
from .chat import LLMClient
from .embedding import EmbedClient
from .registry import (
    close_llm_clients,
    get_embed_client,
    get_llm_client,
    get_rerank_client,
    init_embed_client,
    init_llm_client,
    init_rerank_client,
    reset_clients,
    validate_embedding_dimension,
)
from .rerank import RerankClient

__all__ = [
    "ChatResponse",
    "EmbedClient",
    "EmbeddingResponse",
    "LLMClient",
    "RerankClient",
    "RerankHit",
    "RerankResponse",
    "Usage",
    "close_llm_clients",
    "get_embed_client",
    "get_llm_client",
    "get_rerank_client",
    "init_embed_client",
    "init_llm_client",
    "init_rerank_client",
    "reset_clients",
    "validate_embedding_dimension",
]
