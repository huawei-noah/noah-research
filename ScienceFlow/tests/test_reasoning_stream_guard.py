"""Hidden reasoning chunks share visible-output guards without being rendered."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from deepcraft_core import Memory
from deepcraft_core.llm.base import GuardStreamChunk, StreamHandle

from scienceflow.core.agent import ScienceAgent


async def _sleep_noop(*_args: object, **_kwargs: object) -> None:
    return None


class _AsyncChunks:
    def __init__(self, chunks: list[object]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk

    async def close(self) -> None:
        return None


class _FakeCompletions:
    def __init__(self, response: _AsyncChunks) -> None:
        self._response = response

    async def create(self, **_kwargs: object) -> _AsyncChunks:
        return self._response


class _SSEState:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.requests = 0
        self.paths: list[str] = []


class _ReasoningSSEHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:
        state: _SSEState = self.server.state  # type: ignore[attr-defined]
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        state.requests += 1
        state.paths.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        if state.requests == 1 and state.mode == "repetition":
            deltas = [{"reasoning_content": "R" * 32}] * 3
        elif state.requests == 1:
            deltas = [
                {"reasoning_content": f"private-step-{idx:03d}: hypothesis {idx * 17 + 3}; "}
                for idx in range(12)
            ]
        else:
            deltas = [{"content": "transport recovered"}]

        try:
            for delta in deltas:
                self._send_chunk(delta=delta, finish_reason=None)
            self._send_chunk(delta={}, finish_reason="stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_chunk(self, *, delta: dict[str, str], finish_reason: str | None) -> None:
        payload = {
            "id": "chatcmpl-reasoning-guard",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        self.wfile.flush()


def _chat_chunk(
    *,
    reasoning: str | None = None,
    content: str | None = None,
    finish_reason: str | None = None,
) -> object:
    delta = SimpleNamespace(
        reasoning_content=reasoning,
        content=content,
        tool_calls=None,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


@pytest.mark.asyncio
async def test_online_llm_routes_reasoning_to_hidden_guard_channel() -> None:
    from deepcraft_core.llm.online import OnlineLLM

    async with httpx.AsyncClient(trust_env=False) as http_client:
        llm = OnlineLLM(
            model="unit-test-model",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            http_asyncclient=http_client,
        )
        response = _AsyncChunks(
            [
                _chat_chunk(reasoning="private reasoning"),
                _chat_chunk(content="visible answer", finish_reason="stop"),
            ],
        )
        llm.client = SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeCompletions(response)),
        )

        handle = StreamHandle()
        message = await llm.ask_tool_stream(
            messages=[{"role": "user", "content": "hi"}],
            handle=handle,
            collect_all_tool_calls=True,
        )

    items: list[object] = []
    while True:
        item = await handle.queue.get()
        if item is None:
            break
        items.append(item)

    hidden = [item for item in items if isinstance(item, GuardStreamChunk)]
    visible = [item for item in items if isinstance(item, str)]
    assert [(item.channel, item.text) for item in hidden] == [
        ("reasoning", "private reasoning"),
    ]
    assert "".join(visible) == "visible answer"
    assert message.reasoning_content == "private reasoning"
    assert message.content == "visible answer"


@pytest.mark.parametrize("mode", ["repetition", "soft_limit"])
@pytest.mark.asyncio
async def test_reasoning_guard_retries_real_sse_transport(
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from deepcraft_core.llm.online import OnlineLLM

    monkeypatch.setattr(asyncio, "sleep", _sleep_noop)
    state = _SSEState(mode)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ReasoningSSEHandler)
    server.state = state  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
        async with httpx.AsyncClient(trust_env=False) as http_client:
            llm = OnlineLLM(
                model="fake-model",
                api_key="fake-key",
                base_url=f"http://{host}:{port}/v1",
                max_tokens=128,
                http_asyncclient=http_client,
            )
            agent = ScienceAgent(
                llm=llm,
                memory=Memory(max_messages=20),
                workspace_dir=tmp_path,
                max_steps=1,
                llm_tool_stream_retry_base_delay_sec=0.0,
                llm_tool_stream_retry_max_delay_sec=0.0,
                stream_repetition_detection=mode == "repetition",
                stream_repetition_window_chars=256,
                stream_repetition_ngram_len=32,
                stream_repetition_max_repeats=3,
                stream_max_output_chars_soft=10_000 if mode == "repetition" else 256,
                stream_repetition_retry_max=1,
            )

            result = await agent.run("transport canary")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    output = capsys.readouterr().out
    assert result == "transport recovered"
    assert state.requests == 2
    assert state.paths == ["/v1/chat/completions", "/v1/chat/completions"]
    assert "private-step" not in output
    assert "R" * 16 not in output


@pytest.mark.asyncio
async def test_hidden_reasoning_repetition_uses_existing_retry_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(asyncio, "sleep", _sleep_noop)

    class _RepeatingReasoningLLM:
        model = "fake-reasoning-repeat"
        _last_call_input_tokens = 1
        _last_call_output_tokens = 2

        def __init__(self) -> None:
            self.calls = 0

        async def ask_tool_stream(self, **kwargs: object) -> object:
            self.calls += 1
            handle = kwargs["handle"]
            if self.calls == 1:
                for _ in range(3):
                    await handle.put_guard("R" * 32, channel="reasoning")
                    await asyncio.sleep(0)
                handle.finish()
                return SimpleNamespace(
                    tool_calls=[],
                    content="",
                    reasoning_content="R" * 96,
                )
            await handle.put("recovered")
            handle.finish()
            return SimpleNamespace(tool_calls=[], content="recovered", reasoning_content=None)

    llm = _RepeatingReasoningLLM()
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=2,
        llm_tool_stream_retry_base_delay_sec=0.0,
        llm_tool_stream_retry_max_delay_sec=0.0,
        stream_repetition_window_chars=256,
        stream_repetition_ngram_len=32,
        stream_repetition_max_repeats=3,
        stream_max_output_chars_soft=10_000,
        stream_repetition_retry_max=1,
    )

    result = await agent.run("hi")

    assert llm.calls == 2
    assert "recovered" in result
    assert "R" * 16 not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_non_main_streams_also_use_reasoning_guard_and_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", _sleep_noop)

    class _RouteLLM:
        model = "fake-route-repeat"

        def __init__(self) -> None:
            self.calls = 0

        async def ask_tool_stream(self, **kwargs: object) -> object:
            self.calls += 1
            handle = kwargs["handle"]
            if self.calls == 1:
                for _ in range(3):
                    await handle.put_guard("Q" * 32, channel="reasoning")
                    await asyncio.sleep(0)
                handle.finish()
                return SimpleNamespace(content="", reasoning_content="Q" * 96)
            await handle.put("route recovered")
            handle.finish()
            return SimpleNamespace(content="route recovered", reasoning_content=None)

    llm = _RouteLLM()
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=10),
        workspace_dir=tmp_path,
        stream_repetition_window_chars=256,
        stream_repetition_ngram_len=32,
        stream_repetition_max_repeats=3,
        stream_repetition_retry_max=1,
    )

    result = await agent._ask_tool_stream_guarded(
        messages=[],
        system_msgs=[],
        timeout=10,
        tools=[],
        tool_choice="none",
        parallel_tool_calls=False,
        collect_all_tool_calls=True,
    )

    assert llm.calls == 2
    assert result.content == "route recovered"


@pytest.mark.asyncio
async def test_hidden_reasoning_soft_limit_is_guarded_but_not_logged(tmp_path: Path) -> None:
    agent = ScienceAgent(
        llm=SimpleNamespace(model="unused"),
        memory=Memory(max_messages=10),
        workspace_dir=tmp_path,
        stream_repetition_detection=False,
        stream_max_output_chars_soft=256,
    )
    interaction_path = tmp_path / "interaction.log"
    logger = logging.getLogger(f"reasoning-guard-{id(agent)}")
    logger.propagate = False
    handler = logging.FileHandler(interaction_path)
    logger.addHandler(handler)
    private_reasoning = "private diverse reasoning " * 20
    handle = StreamHandle()
    await handle.put_guard(private_reasoning, channel="reasoning")
    handle.finish()

    try:
        await agent._consume_stream(handle, logger)
    finally:
        handler.close()
        logger.removeHandler(handler)

    log_text = interaction_path.read_text(encoding="utf-8")
    assert handle.interrupted is True
    assert agent._last_stream_guard_reason == "output_soft_limit"
    assert "channel=reasoning" in agent._last_stream_guard_detail
    assert private_reasoning not in log_text
    assert "channel=reasoning output_soft_limit" in log_text
