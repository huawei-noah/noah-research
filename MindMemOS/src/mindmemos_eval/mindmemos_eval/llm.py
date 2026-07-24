"""LLM provider configuration and a thin chat-completion client for evaluation.

Benchmark environments and scorers use this shared :class:`LLMConfig` /
:class:`LLMClient` pair to configure provider and parameters once.

The first supported provider is OpenAI (and any OpenAI-compatible endpoint via
``base_url``). ``provider`` is kept as an explicit field so more backends can be
added later without changing call sites.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal

from mindmemos_sdk.errors import MindMemOSSDKError
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai import AsyncOpenAI

LLMProvider = str  # currently only "openai"; kept open for future backends.


class LLMConfig(BaseModel):
    """LLM provider and call parameter configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: LLMProvider = "openai"
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float = 600.0
    max_retries: int = 5
    retry_backoff: float = 1.0
    extra: dict[str, Any] = Field(default_factory=dict)


class LLMCompletion(BaseModel):
    """Chat completion content and provider-reported token usage."""

    content: str
    message: dict[str, Any] | None = None
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMClient:
    """Small async chat wrapper shared by answering and scoring code."""

    def __init__(self, config: LLMConfig, *, client: AsyncOpenAI | Any | None = None) -> None:
        self.config = config
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self.config.provider != "openai":
            raise MindMemOSSDKError(
                f"Unsupported LLM provider {self.config.provider!r}; only 'openai' is supported for now."
            )
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise MindMemOSSDKError(
                "The 'openai' package is required for evaluation LLM calls. Install it with `uv add openai`."
            ) from exc
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        return self._client

    async def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Call chat completion and return the raw assistant message."""
        completion = await self.complete(messages, return_format="message", tools=tools)
        return completion.message or {"role": "assistant", "content": completion.content}

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        return_format: Literal["text", "message"] = "text",
        tools: list[dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> LLMCompletion:
        """Call chat completion and return content/message plus token usage."""
        if return_format not in {"text", "message"}:
            raise ValueError(f"Unsupported return_format: {return_format!r}")
        response = await self._create_completion(messages, tools=tools, **overrides)
        message = response.choices[0].message
        usage = getattr(response, "usage", None)
        return LLMCompletion(
            content=(self._message_content(message) or "").strip(),
            message=self._message_to_dict(message) if return_format == "message" else None,
            model=getattr(response, "model", None),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        )

    @staticmethod
    def _message_content(message: Any) -> str | None:
        return message.get("content") if isinstance(message, dict) else getattr(message, "content", None)

    @staticmethod
    def _message_to_dict(message: Any) -> dict[str, Any]:
        if hasattr(message, "model_dump"):
            return message.model_dump(exclude_none=True)
        if isinstance(message, dict):
            return {key: value for key, value in message.items() if value is not None}
        return {"role": "assistant", "content": getattr(message, "content", None)}

    async def _create_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> Any:
        client = self._ensure_client()
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            **self.config.extra,
        }
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            params["max_tokens"] = self.config.max_tokens
        if tools is not None:
            params["tools"] = tools or None
        params.update(overrides)

        attempts = self.config.max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return await client.chat.completions.create(**params)
            except Exception as exc:  # noqa: BLE001 - normalize all provider errors
                last_exc = exc
                if attempt + 1 < attempts:
                    await asyncio.sleep(self.config.retry_backoff * 2**attempt)

        raise MindMemOSSDKError(f"LLM completion failed after {attempts} attempts: {last_exc}") from last_exc
