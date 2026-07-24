from types import SimpleNamespace

import pytest
from mindmemos.llm.chat import LLMClient
from mindmemos.llm.embedding import EmbedClient


class RecordingLogger:
    def __init__(self) -> None:
        self.infos: list[dict] = []
        self.warnings: list[dict] = []

    def info(self, event: str, **kwargs) -> None:
        self.infos.append({"event": event, **kwargs})

    def warning(self, event: str, **kwargs) -> None:
        self.warnings.append({"event": event, **kwargs})


class FailingChatRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def acompletion(self, **kwargs):
        self.calls += 1
        raise RuntimeError("chat unavailable")


class SuccessfulChatRouter:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.calls = 0
        self.messages_by_call: list[list[dict]] = []

    async def acompletion(self, **kwargs):
        self.messages_by_call.append(list(kwargs["messages"]))
        content = self.contents[min(self.calls, len(self.contents) - 1)]
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model="chat",
        )


class FailingEmbedRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def aembedding(self, **kwargs):
        self.calls += 1
        raise RuntimeError("embedding unavailable")


class SuccessfulEmbedRouter:
    async def aembedding(self, **kwargs):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, 0.0])],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=None, total_tokens=1),
            model="embedding",
        )


@pytest.mark.asyncio
async def test_chat_client_delegates_provider_retry_to_router() -> None:
    router = FailingChatRouter()
    client = LLMClient(router, max_attempts=2)

    with pytest.raises(RuntimeError, match="chat unavailable"):
        await client.chat(task="test", messages=[{"role": "user", "content": "hello"}])

    # The client issues a single acompletion; provider retries are the router's job.
    assert router.calls == 1


@pytest.mark.asyncio
async def test_chat_format_parser_failure_uses_bounded_parse_retry() -> None:
    router = SuccessfulChatRouter(["bad"])
    client = LLMClient(router, max_attempts=2)

    def fail_parser(content: str):
        raise ValueError(f"cannot parse {content}")

    with pytest.raises(ValueError, match="cannot parse bad"):
        await client.chat(
            task="test",
            messages=[{"role": "user", "content": "hello"}],
            format_parser=fail_parser,
        )

    assert router.calls == 2


@pytest.mark.asyncio
async def test_chat_parse_feedback_is_appended_before_retry() -> None:
    router = SuccessfulChatRouter(["bad", "good"])
    client = LLMClient(router, max_attempts=2)

    def parser(content: str):
        if content == "bad":
            raise ValueError("missing field")
        return {"ok": True}

    response = await client.chat(
        task="test",
        messages=[{"role": "user", "content": "hello"}],
        format_parser=parser,
        feedback_on_parse_error=True,
    )

    assert response.parsed == {"ok": True}
    assert router.calls == 2
    assert router.messages_by_call[0] == [{"role": "user", "content": "hello"}]
    assert router.messages_by_call[1][1] == {"role": "assistant", "content": "bad"}
    assert router.messages_by_call[1][2]["role"] == "user"
    assert "missing field" in router.messages_by_call[1][2]["content"]


@pytest.mark.asyncio
async def test_chat_logs_one_litellm_call_with_latency_and_usage(monkeypatch) -> None:
    logs = RecordingLogger()
    monkeypatch.setattr("mindmemos.llm.chat.logger", logs)
    client = LLMClient(SuccessfulChatRouter(["ok"]))

    await client.chat(task="test", messages=[{"role": "user", "content": "hello"}])

    assert len(logs.infos) == 1
    assert logs.infos[0]["event"] == "litellm_call"
    assert logs.infos[0]["kind"] == "chat"
    assert logs.infos[0]["status"] == "ok"
    assert "latency_ms" in logs.infos[0]
    assert logs.infos[0]["usage"] == {"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2}
    assert logs.warnings == []


@pytest.mark.asyncio
async def test_chat_records_input_and_output_span_events(monkeypatch) -> None:
    events: list[dict] = []
    monkeypatch.setattr(
        "mindmemos.llm.chat.add_span_event",
        lambda name, attributes=None: events.append({"name": name, "attributes": attributes or {}}),
    )
    client = LLMClient(SuccessfulChatRouter(["ok"]))

    await client.chat(task="test", messages=[{"role": "user", "content": "hello"}], temperature=0)

    assert [event["name"] for event in events] == ["llm.chat.input", "llm.chat.output"]
    assert events[0]["attributes"]["messages"] == [{"role": "user", "content": "hello"}]
    assert events[0]["attributes"]["kwargs"] == {"temperature": 0}
    assert events[1]["attributes"]["content"] == "ok"
    assert events[1]["attributes"]["usage"] == {"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2}


@pytest.mark.asyncio
async def test_chat_parse_retry_logs_once_per_litellm_call(monkeypatch) -> None:
    logs = RecordingLogger()
    monkeypatch.setattr("mindmemos.llm.chat.logger", logs)
    router = SuccessfulChatRouter(["bad", "good"])
    client = LLMClient(router, max_attempts=2)

    def parser(content: str):
        if content == "bad":
            raise ValueError("missing field")
        return {"ok": True}

    await client.chat(
        task="test",
        messages=[{"role": "user", "content": "hello"}],
        format_parser=parser,
    )

    assert router.calls == 2
    assert [log["status"] for log in logs.infos] == ["ok", "ok"]
    assert logs.warnings == []


@pytest.mark.asyncio
async def test_embed_client_delegates_provider_retry_to_router() -> None:
    router = FailingEmbedRouter()
    client = EmbedClient(router)

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        await client.embed(task="test", text="hello")

    assert router.calls == 1


@pytest.mark.asyncio
async def test_embed_client_returns_success() -> None:
    client = EmbedClient(SuccessfulEmbedRouter())

    response = await client.embed(task="test", text="hello")

    assert response.embeddings == [[1.0, 0.0]]
    assert response.usage.total_tokens == 1


@pytest.mark.asyncio
async def test_embed_records_input_span_event(monkeypatch) -> None:
    events: list[dict] = []
    monkeypatch.setattr(
        "mindmemos.llm.embedding.add_span_event",
        lambda name, attributes=None: events.append({"name": name, "attributes": attributes or {}}),
    )
    client = EmbedClient(SuccessfulEmbedRouter())

    await client.embed(task="test", text=["hello", "world!"])

    assert [event["name"] for event in events] == ["llm.embed.input"]
    assert events[0]["attributes"]["text"] == ["hello", "world!"]
    assert events[0]["attributes"]["llm.embed.input.count"] == 2
    assert events[0]["attributes"]["llm.embed.input.chars"] == 11


@pytest.mark.asyncio
async def test_embed_provider_span_records_debug_attrs(monkeypatch) -> None:
    seen: dict = {}

    async def traced(name, awaitable, *, attributes=None, **kwargs):
        seen["name"] = name
        seen["attributes"] = attributes
        return await awaitable

    router = SuccessfulEmbedRouter()
    router.model_list = [
        {
            "model_name": "embedding",
            "litellm_params": {
                "model": "openai/embed",
                "api_base": "https://example.test/v1",
                "rpm": 60,
                "tpm": 6000,
                "timeout": 30,
                "num_retries": 2,
            },
        }
    ]
    router.routing_strategy = "least-busy"
    router.num_retries = 2
    router.allowed_fails = 1
    router.cooldown_time = 5
    monkeypatch.setattr("mindmemos.llm.embedding.traced_awaitable", traced)

    await EmbedClient(router).embed(task="bench", text=["hello", "world!"])

    assert seen["name"] == "llm.embed.provider"
    assert seen["attributes"]["llm.embed.input.count"] == 2
    assert seen["attributes"]["llm.embed.input.chars"] == 11
    assert seen["attributes"]["llm.endpoint.rpm.max"] == 60
    assert seen["attributes"]["llm.endpoint.api_hosts"] == "example.test"
