from typing import Any

from pydantic import BaseModel, Field


class Usage(BaseModel):
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    total_tokens: int | None = None


class ChatResponse(BaseModel):
    finish_reason: str
    content: str
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
    parsed: Any = None
    raw_response: dict[str, Any] = Field(default_factory=dict)


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
    raw_response: dict[str, Any] = Field(default_factory=dict)


class RerankHit(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: list[RerankHit]
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
