# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

import json
import re
import time
import logging
from types import SimpleNamespace
from typing import Any, List, Optional, Union, Tuple, Dict
from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
import asyncio
import math

from .base import BaseLLM, CallStats, StreamHandle
from ..message import Message
from ..tool import TOOL_CHOICE_VALUES, TOOL_CHOICE_TYPE
from .utils import get_text_from_content
logger = logging.getLogger(__name__)

# Strip literal think markers some models embed in ``content`` (not ``reasoning_content``).
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)

# Locate the opening of the ``content`` string value in streamed write-tool JSON.
_WRITE_TOOL_CONTENT_KEY_RE = re.compile(r'"content"\s*:\s*"')
# Extract ``path`` from partial JSON (before ``content``) for a one-line header.
_WRITE_TOOL_PATH_RE = re.compile(r'"path"\s*:\s*"((?:[^"\\]|\\.)*)"')

# Split large LLM / tool-arg chunks so TTY consumers can flush stdout between queue items.
_STREAM_PUT_CHUNK_CHARS = 128


def _extract_cached_input_tokens(usage_obj: Any) -> int:
    """Return prefix-cache hit input tokens reported by the API.

    Tries OpenAI-compatible ``usage.prompt_tokens_details.cached_tokens`` first, then
    DeepSeek's ``usage.prompt_cache_hit_tokens``. Returns 0 when the field is missing
    or the provider does not report cache stats. Tolerates both attribute-style and
    dict-style usage payloads.
    """
    if usage_obj is None:
        return 0
    try:
        details = getattr(usage_obj, "prompt_tokens_details", None)
        if details is None and isinstance(usage_obj, dict):
            details = usage_obj.get("prompt_tokens_details")
        if details is not None:
            v = getattr(details, "cached_tokens", None)
            if v is None and isinstance(details, dict):
                v = details.get("cached_tokens")
            if v is not None:
                return int(v)
        v = getattr(usage_obj, "prompt_cache_hit_tokens", None)
        if v is None and isinstance(usage_obj, dict):
            v = usage_obj.get("prompt_cache_hit_tokens")
        if v is not None:
            return int(v)
    except (TypeError, ValueError):
        return 0
    return 0


async def _chunked_put(handle: StreamHandle, text: str) -> None:
    """Enqueue *text* in small pieces so TTY consumers can render incrementally."""
    if not text:
        return
    n = _STREAM_PUT_CHUNK_CHARS
    if len(text) <= n:
        await handle.put(text)
        await asyncio.sleep(0)
        return
    for i in range(0, len(text), n):
        await handle.put(text[i : i + n])
        await asyncio.sleep(0)


async def _chunked_guard_put(handle: StreamHandle, text: str, *, channel: str) -> None:
    """Feed hidden provider output to stream guards without displaying or logging it."""
    if not text:
        return
    n = _STREAM_PUT_CHUNK_CHARS
    for i in range(0, len(text), n):
        if handle.interrupted:
            break
        await handle.put_guard(text[i : i + n], channel=channel)
        await asyncio.sleep(0)


class _WriteToolArgStreamDecoder:
    """Incrementally decode the ``content`` field of streamed ``write`` tool JSON for TTY.

    Avoids dumping raw JSON (with ``\\n`` escapes) and bypasses the small preview cap so
    users see readable code as it arrives. State is stored on the tool-call *slot* dict.
    """

    _PHASE_FIND = "find"
    _PHASE_BODY = "body"
    _PHASE_DONE = "done"

    @classmethod
    def ensure_state(cls, slot: dict[str, Any]) -> dict[str, Any]:
        st = slot.get("_write_stream_decode")
        if st is None:
            st = {
                "phase": cls._PHASE_FIND,
                "i": 0,
                "escape": False,
                "unicode_mode": False,
                "unicode_buf": "",  # hex digits after \\u (0..4 chars)
                "header": False,
            }
            slot["_write_stream_decode"] = st
        return st

    @staticmethod
    def _unescape_json_string_fragment(escaped_inner: str) -> str:
        """Decode a JSON string *body* (no surrounding quotes) for path header only."""
        if not escaped_inner:
            return ""
        try:
            return json.loads(f'"{escaped_inner}"')
        except json.JSONDecodeError:
            return escaped_inner.replace("\\/", "/")

    @classmethod
    async def feed(
        cls,
        slot: dict[str, Any],
        handle: "StreamHandle",
    ) -> None:
        """Consume new bytes in ``slot['arguments']`` and push decoded file text to *handle*."""
        args = slot.get("arguments") or ""
        st = cls.ensure_state(slot)

        if st["phase"] == cls._PHASE_DONE:
            return

        if st["phase"] == cls._PHASE_FIND:
            m = _WRITE_TOOL_CONTENT_KEY_RE.search(args)
            if not m:
                return
            st["i"] = m.end()
            st["phase"] = cls._PHASE_BODY
            if not st["header"]:
                st["header"] = True
                pm = _WRITE_TOOL_PATH_RE.search(args[: st["i"]])
                path = cls._unescape_json_string_fragment(pm.group(1)) if pm else "?"
                await handle.put(f"\n[write -> {path}]\n")
                await asyncio.sleep(0)

        if st["phase"] != cls._PHASE_BODY:
            return

        out: list[str] = []
        i = st["i"]
        escape = st["escape"]
        unicode_mode: bool = st["unicode_mode"]
        unicode_buf: str = st["unicode_buf"]
        out_len = 0
        _flush_n = _STREAM_PUT_CHUNK_CHARS

        while i < len(args):
            ch = args[i]

            if unicode_mode:
                if ch in "0123456789abcdefABCDEF" and len(unicode_buf) < 4:
                    unicode_buf += ch
                    i += 1
                    if len(unicode_buf) < 4:
                        if i >= len(args):
                            break
                        continue
                    try:
                        out.append(chr(int(unicode_buf, 16)))
                    except ValueError:
                        out.append("?")
                    out_len += 1
                    unicode_buf = ""
                    unicode_mode = False
                    escape = False
                    if out_len >= _flush_n:
                        await handle.put("".join(out))
                        await asyncio.sleep(0)
                        out.clear()
                        out_len = 0
                    continue
                out.append("u")
                for c in unicode_buf:
                    out.append(c)
                out_len += 1 + len(unicode_buf)
                unicode_buf = ""
                unicode_mode = False
                escape = False
                if out_len >= _flush_n:
                    await handle.put("".join(out))
                    await asyncio.sleep(0)
                    out.clear()
                    out_len = 0
                continue

            if escape:
                if ch == "u":
                    unicode_mode = True
                    unicode_buf = ""
                    escape = False
                    i += 1
                    continue
                esc_map = {
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                    '"': '"',
                    "\\": "\\",
                    "/": "/",
                    "b": "\b",
                    "f": "\f",
                }
                out.append(esc_map.get(ch, ch))
                out_len += 1
                escape = False
                i += 1
                if out_len >= _flush_n:
                    await handle.put("".join(out))
                    await asyncio.sleep(0)
                    out.clear()
                    out_len = 0
                continue

            if ch == "\\":
                escape = True
                i += 1
                continue

            if ch == '"':
                st["phase"] = cls._PHASE_DONE
                st["i"] = i + 1
                st["escape"] = False
                st["unicode_mode"] = False
                st["unicode_buf"] = ""
                if out:
                    await handle.put("".join(out))
                    await asyncio.sleep(0)
                return

            out.append(ch)
            out_len += 1
            i += 1
            if out_len >= _flush_n:
                await handle.put("".join(out))
                await asyncio.sleep(0)
                out.clear()
                out_len = 0

        st["i"] = i
        st["escape"] = escape
        st["unicode_mode"] = unicode_mode
        st["unicode_buf"] = unicode_buf
        if out:
            await handle.put("".join(out))
            await asyncio.sleep(0)


class OnlineLLM(BaseLLM):
    client: Any = Field(default=None, exclude=True)  #: :meta private:

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",  # Allow extra fields for flexibility in subclasses
    )

    @model_validator(mode="after")
    def init_onlinellm(self) -> "OnlineLLM":
        if self.api_type == "azure":
            self.client = AsyncAzureOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                default_headers=self.headers,
                api_version=self.api_version,
                http_client=self.http_asyncclient,
            )
        else:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=self.headers,
                http_client=self.http_asyncclient,
            )
        return self

    def _add_explicit_top_p(self, create_kwargs: dict[str, Any]) -> None:
        """Forward ``top_p`` only when the caller explicitly configured it.

        BaseLLM's historical default is 0.001. Some OpenAI-compatible
        providers reject that value, so default requests should omit
        ``top_p`` just like the tool-call paths do.
        """
        fields_set = getattr(self, "model_fields_set", set())
        if "top_p" not in fields_set or self.top_p is None:
            return
        create_kwargs["top_p"] = self.top_p

    @retry(
        wait=wait_random_exponential(min=2, max=120),
        stop=stop_after_attempt(10),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
        *,
        stream_handle: Optional[StreamHandle] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response
            stream_handle: Per-call streaming control (queue + interrupt).
                When provided, chunks are pushed to ``handle.queue`` and the
                caller can abort generation via ``handle.interrupt()``.
            timeout: Optional per-request timeout in seconds for the HTTP call.
                When omitted, relies on the SDK / httpx client defaults.

        Returns:
            str: The generated response

        Raises:
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            try:
                return await self._stream_request(
                    messages, stream, temperature,
                    handle=stream_handle, timeout=timeout,
                )
            except ValueError as ve:
                if "Empty response from streaming LLM" in str(ve) and stream:
                    logger.warning("[ask] Empty streaming response, retrying with non-streaming fallback...")
                    await asyncio.sleep(2)
                    return await self._stream_request(
                        messages, False, temperature,
                        handle=None, timeout=timeout,
                    )
                raise

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_logits(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Tuple[str, Dict[str, List[float]]]:
        """
        Send a prompt to the LLM and get the response with top 5 logprobs.
        Return top 5 logprobs, because api can return at most 5 logprobs.
            self,
            messages: List[Union[dict, Message]],
            system_msgs: Optional[List[Union[dict, Message]]] = None,
            stream: bool = True,
            temperature: Optional[float] = None,
            **kwargs,
    ) -> Tuple[str, Dict[str, List[float]]]:
        """
        try:
            # Format system and user messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            return await self._stream_request_logits(messages, temperature, stream)

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )

    async def ask_vllm_logits(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Tuple[str, Dict[str, List[float]]]:
        """
        Send a prompt to the LLM and get the response with top 5 logprobs.
        Return top 5 logprobs, because api can return at most 5 logprobs.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Format system and user messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            return await self._stream_request_vllm_logits(messages, temperature, stream)

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )

    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 60,
        tools: Optional[List[dict]] = None,
        tool_choice: Any = "auto",
        temperature: Optional[float] = None,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs,
    ):
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            parallel_tool_calls: Whether to call tools in parallel. ``None`` omits the field from
                the request (matches OpenAI ``NOT_GIVEN``); use for backends that reject unknown
                parameters (e.g. some GLM-compatible gateways).
            **kwargs: Additional completion arguments (e.g. ``top_p``). ``top_p`` is **not** sent
                by default here; pass ``top_p=...`` when you need BaseLLM sampling control.

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            t0 = time.time()
            # Omit ``top_p`` unless passed in ``kwargs``: some OpenAI-compatible gateways
            # (e.g. Zhipu GLM) return 400/1210 when ``top_p`` is sent with tools (BaseLLM
            # default 0.001 is brittle). Matches minimal clients that only set temperature.
            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": self.max_tokens,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
            }
            if self.frequency_penalty is not None:
                create_kwargs["frequency_penalty"] = float(self.frequency_penalty)
            if parallel_tool_calls is not None:
                create_kwargs["parallel_tool_calls"] = parallel_tool_calls
            create_kwargs.update(kwargs)
            response = await self.client.chat.completions.create(**create_kwargs)
            round_trip = time.time() - t0

            if not response.choices or not response.choices[0].message:
                print(response)
                raise ValueError("Invalid or empty response from LLM")

            local_input_tokens = 0
            local_output_tokens = 0
            local_tpot = 0.0
            local_cached_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                local_input_tokens = response.usage.prompt_tokens
                local_output_tokens = response.usage.completion_tokens
                local_cached_tokens = _extract_cached_input_tokens(response.usage)
                self.token_tracker(local_input_tokens, local_output_tokens)
                if local_output_tokens > 1:
                    local_tpot = round_trip / local_output_tokens * 1000
            self._save_call_stats(CallStats(
                input_tokens=local_input_tokens, output_tokens=local_output_tokens,
                ttft=round_trip, tpot=local_tpot, finish_reason="unknown",
                input_cached_tokens=local_cached_tokens,
            ))

            return response.choices[0].message

        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise

    @staticmethod
    def _tool_call_slot_ready(slot: dict[str, Any]) -> bool:
        """True when index-0 tool call has id, name, and parseable JSON arguments."""
        if not slot.get("name"):
            return False
        if not slot.get("id"):
            return False
        raw = (slot.get("arguments") or "").strip()
        if not raw:
            return False
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            return False
        return True

    @staticmethod
    def _tool_arg_stream_preview_cap(slot: dict[str, Any]) -> int:
        """Max chars of tool JSON to stream to the terminal (avoid huge ``write`` bodies)."""
        name = slot.get("name") or ""
        if name == "write":
            return 1200
        if name == "edit":
            return 2500
        return 4000

    @staticmethod
    def _tool_arg_stream_progress_chunks(
        slot: dict[str, Any],
        cap: int,
        total_len: int,
        step: int = 6144,
    ) -> list[str]:
        """After preview is capped, emit periodic one-line hints while JSON args still grow.

        Avoids a long silent terminal phase when the model streams a huge ``write``/``edit``
        payload. Mutates *slot* with ``_arg_stream_next_milestone``.
        """
        name = slot.get("name") or ""
        if name not in ("write", "edit") or total_len <= cap:
            return []
        out: list[str] = []
        next_milestone = slot.get("_arg_stream_next_milestone")
        if next_milestone is None:
            next_milestone = cap + step
        while total_len >= next_milestone:
            out.append(
                f"\n[streaming tool arguments: {total_len} chars received…]\n",
            )
            next_milestone += step
        slot["_arg_stream_next_milestone"] = next_milestone
        return out

    async def ask_tool_stream(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: Any = "auto",
        temperature: Optional[float] = None,
        parallel_tool_calls: Optional[bool] = None,
        *,
        handle: StreamHandle,
        collect_all_tool_calls: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Stream a tools completion: push *content* chunks to *handle*; return a synthetic message.

        *parallel_tool_calls*: ``None`` omits the field from the request (same as ``ask_tool``);
        set explicitly when you need OpenAI-style parallel tool calls. ``top_p`` is not sent by
        default; pass ``top_p=...`` in ``**kwargs`` when needed (some GLM gateways reject default
        ``top_p`` with tools).

        When *collect_all_tool_calls* is false (default): when the first tool call (index 0)
        has complete JSON arguments, calls ``handle.stop()`` and stops reading the stream
        (single-action mode).

        When *collect_all_tool_calls* is true: consumes the full stream and returns every
        tool call slot that has complete JSON (multiple indices), without setting API
        ``parallel_tool_calls`` — use this to support multi read/grep without changing model
        tool-choice behavior.
        """
        kwargs = dict(kwargs)
        kwargs.pop("stream", None)
        kwargs.pop("stream_handle", None)

        collect_stream_mode = collect_all_tool_calls

        try:
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            if tools:
                for t in tools:
                    if not isinstance(t, dict) or "type" not in t:
                        raise ValueError("Each tool must be a dict with 'type' field")

            t0 = time.time()
            # Omit ``top_p`` unless passed in ``kwargs``: align with ``ask_tool`` (GLM 400/1210).
            create_kwargs: dict[str, Any] = dict(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )
            if self.frequency_penalty is not None:
                create_kwargs["frequency_penalty"] = float(self.frequency_penalty)
            if parallel_tool_calls is not None:
                create_kwargs["parallel_tool_calls"] = parallel_tool_calls

            try:
                response = await self.client.chat.completions.create(**create_kwargs)
            except Exception:
                if not handle.interrupted:
                    handle.finish()
                raise

            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            by_index: dict[int, dict[str, Any]] = {}
            last_finish_reason = "unknown"
            api_usage = None
            local_input_tokens = 0
            local_output_tokens = 0
            for msg in messages:
                text = get_text_from_content(msg.get("content") or "")
                num_tokens_estimated = self.estimate_tokens(text)
                local_input_tokens += num_tokens_estimated
                self.token_tracker(num_tokens_estimated, 0)

            try:
                async for chunk in response:
                    if handle.interrupted:
                        logger.info("ask_tool_stream interrupted after first tool call")
                        last_finish_reason = "interrupted_by_client"
                        break

                    if hasattr(chunk, "usage") and chunk.usage:
                        api_usage = chunk.usage

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    if getattr(choice, "finish_reason", None):
                        last_finish_reason = choice.finish_reason or "unknown"

                    delta = choice.delta
                    if delta is None:
                        continue

                    stream_text = ""

                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        rchunk = delta.reasoning_content or ""
                        reasoning_parts.append(rchunk)
                        await _chunked_guard_put(handle, rchunk, channel="reasoning")
                        num_tokens_estimated = self.estimate_tokens(rchunk)
                        local_output_tokens += num_tokens_estimated
                        self.token_tracker(0, num_tokens_estimated)

                    if hasattr(delta, "content") and delta.content:
                        c = _THINK_TAG_RE.sub("", delta.content or "")
                        content_parts.append(c)
                        stream_text = c
                        num_tokens_estimated = self.estimate_tokens(c)
                        local_output_tokens += num_tokens_estimated
                        self.token_tracker(0, num_tokens_estimated)

                    if stream_text:
                        await _chunked_put(handle, stream_text)

                    tool_deltas = getattr(delta, "tool_calls", None) or []
                    for tc in tool_deltas:
                        idx = getattr(tc, "index", None)
                        if idx is None:
                            # Some OpenAI-compatible proxies omit ``index`` on extra tool slots; assign
                            # sequential indices so we do not drop the second+ tool call.
                            idx = max(by_index.keys(), default=-1) + 1
                        slot = by_index.setdefault(
                            idx,
                            {
                                "id": None,
                                "name": None,
                                "arguments": "",
                                "_name_streamed": False,
                                "_arg_streamed_len": 0,
                            },
                        )
                        if getattr(tc, "id", None):
                            slot["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                slot["name"] = fn.name
                                if (collect_stream_mode or idx == 0) and not slot.get(
                                    "_name_streamed",
                                ):
                                    slot["_name_streamed"] = True
                                    label = fn.name
                                    await handle.put(f"\n→ {label}\n")
                                    await asyncio.sleep(0)
                            if getattr(fn, "arguments", None):
                                frag = fn.arguments or ""
                                slot["arguments"] = slot["arguments"] + frag
                                if (collect_stream_mode or idx == 0) and frag:
                                    if (slot.get("name") or "") == "write":
                                        await _WriteToolArgStreamDecoder.feed(
                                            slot,
                                            handle,
                                        )
                                    else:
                                        cap = self._tool_arg_stream_preview_cap(slot)
                                        so_far = int(slot.get("_arg_streamed_len", 0))
                                        if so_far < cap:
                                            room = cap - so_far
                                            preview = (
                                                frag[:room]
                                                if len(frag) > room
                                                else frag
                                            )
                                            if preview:
                                                await _chunked_put(handle, preview)
                                        slot["_arg_streamed_len"] = so_far + len(frag)
                                        if (
                                            so_far + len(frag) > cap
                                            and not slot.get("_arg_trunc_notice")
                                        ):
                                            slot["_arg_trunc_notice"] = True
                                            await _chunked_put(
                                                handle,
                                                "\n… [tool args preview truncated here; "
                                                "full JSON in panel below]\n",
                                            )
                                        for prog in self._tool_arg_stream_progress_chunks(
                                            slot,
                                            cap,
                                            len(slot["arguments"]),
                                        ):
                                            await _chunked_put(handle, prog)

                        if (
                            not collect_stream_mode
                            and idx == 0
                            and self._tool_call_slot_ready(slot)
                        ):
                            handle.stop()
                            break

                    if handle.interrupted:
                        last_finish_reason = "interrupted_by_client"
                        break

            finally:
                if hasattr(response, "close"):
                    try:
                        await response.close()
                    except Exception:
                        pass
                if not handle.interrupted:
                    handle.finish()

            round_trip = time.time() - t0
            full_content = "".join(content_parts)
            joined_reasoning = "".join(reasoning_parts)
            reasoning_out = joined_reasoning if joined_reasoning.strip() else None

            local_cached_tokens = _extract_cached_input_tokens(api_usage)
            if api_usage:
                local_input_tokens = getattr(api_usage, "prompt_tokens", 0) or int(
                    local_input_tokens,
                )
                local_output_tokens = getattr(api_usage, "completion_tokens", 0) or int(
                    local_output_tokens,
                )
            else:
                local_input_tokens = int(local_input_tokens)
                local_output_tokens = int(local_output_tokens)

            if self.tracker and local_output_tokens > 1:
                tpot = round_trip / local_output_tokens * 1000
            else:
                tpot = 0.0

            self._save_call_stats(
                CallStats(
                    input_tokens=local_input_tokens,
                    output_tokens=local_output_tokens,
                    ttft=round_trip,
                    tpot=tpot,
                    finish_reason=last_finish_reason,
                    input_cached_tokens=local_cached_tokens,
                ),
            )

            if collect_stream_mode and by_index:
                incomplete_slots: list[tuple[int, dict[str, Any]]] = [
                    (idx, slot)
                    for idx, slot in sorted(by_index.items())
                    if not self._tool_call_slot_ready(slot)
                ]
                if incomplete_slots:
                    incomplete_names = [
                        str(slot.get("name") or "?")
                        for _, slot in incomplete_slots
                    ]
                    has_file_mutation = any(
                        name in ("write", "edit") for name in incomplete_names
                    )
                    if has_file_mutation or last_finish_reason == "length":
                        detail = ", ".join(
                            f"idx={idx} name={slot.get('name') or '?'} "
                            f"args_chars={len(slot.get('arguments') or '')}"
                            for idx, slot in incomplete_slots
                        )
                        raise ValueError(
                            "Incomplete streaming tool call(s) from LLM; "
                            f"finish_reason={last_finish_reason}; {detail}"
                        )

            tool_calls_out: list[Any] = []
            if collect_stream_mode:
                for idx in sorted(by_index.keys()):
                    slot = by_index[idx]
                    if not self._tool_call_slot_ready(slot):
                        continue
                    raw_args = (slot.get("arguments") or "").strip()
                    tool_calls_out.append(
                        SimpleNamespace(
                            id=slot["id"],
                            function=SimpleNamespace(
                                name=slot["name"],
                                arguments=raw_args if raw_args else "{}",
                            ),
                        ),
                    )
            elif 0 in by_index and self._tool_call_slot_ready(by_index[0]):
                s0 = by_index[0]
                raw_args = (s0.get("arguments") or "").strip()
                tool_calls_out.append(
                    SimpleNamespace(
                        id=s0["id"],
                        function=SimpleNamespace(
                            name=s0["name"],
                            arguments=raw_args if raw_args else "{}",
                        ),
                    ),
                )

            if (
                not tool_calls_out
                and not full_content.strip()
                and reasoning_out is None
                and not handle.interrupted
            ):
                raise ValueError("Empty response from streaming tool LLM")

            return SimpleNamespace(
                content=full_content,
                reasoning_content=reasoning_out,
                tool_calls=tool_calls_out if tool_calls_out else None,
            )

        except ValueError as ve:
            logger.error(f"Validation error in ask_tool_stream: {ve}")
            raise
        except OpenAIError as oe:
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(
                "Unexpected error in ask_tool_stream: %s: %r",
                type(e).__name__,
                e,
            )
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_embedding(
        self,
        input: str,
        encoding_format: str = "float",
    )-> list:
        """
        Ask Text Model return the embedding list.

        Args:
            input: List of conversation messages
            encoding_format: Optional data dtype, def. float

        Returns:
            list: The input's embedding list

        Raises:
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            response = await self.client.embeddings.create(
                input=input,
                model=self.model,
                encoding_format=encoding_format
            )

            embedding = response.data[0].embedding
            return embedding

        except OpenAIError as oe:
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
    async def _stream_request(
        self,
        messages,
        stream,
        temperature,
        handle: Optional[StreamHandle] = None,
        timeout: Optional[float] = None,
    ):
        """Send a request to the LLM.  Supports streaming and non-streaming.

        Args:
            handle: Per-call ``StreamHandle``.  When *None* a temporary handle
                is created internally (backward-compatible for callers that
                don't consume the chunk queue).
            timeout: Per-request timeout in seconds (passed to OpenAI ``create``).
                When *None*, the HTTP client default applies (may hang on stalled streams).
        """
        h = handle or StreamHandle()
        reasoning_flag = False
        local_input_tokens = 0
        local_output_tokens = 0
        t0 = time.time()

        create_kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=temperature or self.temperature,
            stream=stream and self.stream,
        )
        self._add_explicit_top_p(create_kwargs)
        if self.frequency_penalty is not None:
            create_kwargs["frequency_penalty"] = float(self.frequency_penalty)
        if self.reasoning_effort and "gemini" in (self.model or "").lower():
            create_kwargs["reasoning_effort"] = self.reasoning_effort
        if stream and self.stream:
            create_kwargs["stream_options"] = {"include_usage": True}
        if timeout is not None:
            create_kwargs["timeout"] = float(timeout)

        response = await self.client.chat.completions.create(**create_kwargs)

        if not (self.stream and stream):
            # Non-streaming request — handle not used
            local_ttft = time.time() - t0
            local_tpot = 0.0
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty or invalid response from LLM")
            local_cached_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                local_input_tokens = response.usage.prompt_tokens
                local_output_tokens = response.usage.completion_tokens
                local_cached_tokens = _extract_cached_input_tokens(response.usage)
                self.token_tracker(local_input_tokens, local_output_tokens)
                if local_output_tokens > 1:
                    local_tpot = local_ttft / local_output_tokens * 1000
            local_finish_reason = getattr(response.choices[0], "finish_reason", None) or "unknown"
            self._save_call_stats(CallStats(
                input_tokens=local_input_tokens, output_tokens=local_output_tokens,
                ttft=local_ttft, tpot=local_tpot, finish_reason=local_finish_reason,
                input_cached_tokens=local_cached_tokens,
            ))
            return response.choices[0].message.content

        else:
            # Streaming request
            collected_messages = []
            t_first = None
            t_last = t0
            chunk_count = 0
            local_ttft = 0.0
            last_finish_reason = "unknown"
            api_usage = None
            for msg in messages:
                text = get_text_from_content(msg.get("content") or "")
                num_tokens_estimated = self.estimate_tokens(text)
                local_input_tokens += num_tokens_estimated
                self.token_tracker(num_tokens_estimated, 0)
            try:
                async for chunk in response:
                    if h.interrupted:
                        logger.info("Generation interrupted via StreamHandle")
                        break

                    if hasattr(chunk, 'usage') and chunk.usage:
                        api_usage = chunk.usage

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]

                    if getattr(choice, "finish_reason", None):
                        last_finish_reason = choice.finish_reason

                    chunk_message = ""
                    if hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content:
                        chunk_message = choice.delta.reasoning_content or ""
                        if reasoning_flag == False:
                            chunk_message = "<think>" + chunk_message
                            reasoning_flag = True

                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        chunk_message = choice.delta.content or ""
                        if reasoning_flag == True:
                            chunk_message = "</think>" + chunk_message
                            reasoning_flag = False

                    if chunk_message:
                        t_now = time.time()
                        if t_first is None:
                            t_first = t_now
                            local_ttft = t_first - t0
                        t_last = t_now
                        chunk_count += 1

                    collected_messages.append(chunk_message)
                    num_tokens_estimated = self.estimate_tokens(chunk_message)
                    local_output_tokens += num_tokens_estimated
                    self.token_tracker(0, num_tokens_estimated)
                    await _chunked_put(h, chunk_message)
            finally:
                if hasattr(response, 'close'):
                    try:
                        await response.close()
                    except Exception:
                        pass
                h.finish()

            full_response = "".join(collected_messages)
            if not full_response and not h.interrupted:
                raise ValueError("Empty response from streaming LLM")
            local_tpot = ((t_last - t_first) / (chunk_count - 1) * 1000
                          if t_first is not None and chunk_count > 1 else 0.0)
            local_cached_tokens = _extract_cached_input_tokens(api_usage)
            if api_usage:
                local_input_tokens = getattr(api_usage, 'prompt_tokens', 0) or int(local_input_tokens)
                local_output_tokens = getattr(api_usage, 'completion_tokens', 0) or int(local_output_tokens)
            else:
                local_input_tokens = int(local_input_tokens)
                local_output_tokens = int(local_output_tokens)
            self._save_call_stats(CallStats(
                input_tokens=local_input_tokens, output_tokens=local_output_tokens,
                ttft=local_ttft, tpot=local_tpot, finish_reason=last_finish_reason,
                input_cached_tokens=local_cached_tokens,
            ))
            return full_response
    

    async def _stream_request_logits(self, messages, temperature, stream=False,
                                     handle: Optional[StreamHandle] = None):
        """Send a logprobs request.  Streaming path uses *handle* for chunks."""
        h = handle or StreamHandle()
        reasoning_flag = False
        t0 = time.time()
        req_kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            stream=stream and self.stream,
            temperature=temperature or self.temperature,
            logprobs=True,
            top_logprobs=5,
        )
        self._add_explicit_top_p(req_kwargs)
        if self.frequency_penalty is not None:
            req_kwargs["frequency_penalty"] = float(self.frequency_penalty)
        response = await self.client.chat.completions.create(**req_kwargs)
        top_probability_list = []
        if not (self.stream and stream):
            local_ttft = time.time() - t0
            local_tpot = 0.0
            local_input_tokens = 0
            local_output_tokens = 0
            local_cached_tokens = 0
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty or invalid response from LLM")
            if hasattr(response, 'usage') and response.usage:
                local_input_tokens = response.usage.prompt_tokens
                local_output_tokens = response.usage.completion_tokens
                local_cached_tokens = _extract_cached_input_tokens(response.usage)
                self.token_tracker(local_input_tokens, local_output_tokens)
                if local_output_tokens > 1:
                    local_tpot = local_ttft / local_output_tokens * 1000
            local_finish_reason = getattr(response.choices[0], "finish_reason", None) or "unknown"
            logprobs_list = response.choices[0].logprobs.content
            for i, token_data in enumerate(logprobs_list):
                token_i = token_data.token
                top_probability_i = []
                for rank, top_lp in enumerate(token_data.top_logprobs):
                    probability = top_lp.logprob
                    top_probability_i.append(probability)

                top_probability_list.append({
                    'token': token_i,
                    'top_logprobs': top_probability_i,
                })

            self._save_call_stats(CallStats(
                input_tokens=local_input_tokens, output_tokens=local_output_tokens,
                ttft=local_ttft, tpot=local_tpot, finish_reason=local_finish_reason,
                input_cached_tokens=local_cached_tokens,
            ))
            return response.choices[0].message.content, top_probability_list

        else:
            collected_messages = []
            t_first = None
            t_last = t0
            chunk_count = 0
            local_ttft = 0.0
            local_output_tokens = 0
            start_flag = False
            self.token_tracker(len(messages), 0)
            try:
                async for chunk in response:
                    if h.interrupted:
                        logger.info("Generation interrupted via StreamHandle (logits)")
                        break

                    choice = chunk.choices[0]
                    chunk_message = ""
                    if hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content:
                        chunk_message = choice.delta.reasoning_content or ""
                        if reasoning_flag == False:
                            chunk_message = "<think>" + chunk_message
                            reasoning_flag = True

                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        chunk_message = choice.delta.content or ""
                        if reasoning_flag == True:
                            chunk_message = "</think>" + chunk_message
                            reasoning_flag = False

                        if hasattr(choice.delta, 'logprobs') and choice.delta.logprobs and choice.delta.logprobs['content']:
                            for token_data in choice.delta.logprobs['content']:
                                token_i = token_data['token']
                                if token_i == "```":
                                    start_flag = True

                                if start_flag:
                                    top_logprobs_dict_i = token_data['top_logprobs']
                                    top_probability_i = []
                                    for top_lp in top_logprobs_dict_i:
                                        probability = top_lp['logprob']
                                        top_probability_i.append(probability)

                                    top_probability_list.append({
                                        'token': token_i,
                                        'top_logprobs': top_probability_i,
                                    })

                    if chunk_message:
                        t_now = time.time()
                        if t_first is None:
                            t_first = t_now
                            local_ttft = t_first - t0
                        t_last = t_now
                        chunk_count += 1

                    collected_messages.append(chunk_message)
                    num_tokens_estimated = self.estimate_tokens(chunk_message)
                    local_output_tokens += num_tokens_estimated
                    self.token_tracker(0, num_tokens_estimated)
                    await _chunked_put(h, chunk_message)

            finally:
                if hasattr(response, 'close'):
                    try:
                        await response.close()
                    except Exception:
                        pass
                h.finish()
            full_response = "".join(collected_messages).strip()
            if not full_response and not h.interrupted:
                raise ValueError("Empty response from streaming LLM")
            local_tpot = ((t_last - t_first) / (chunk_count - 1) * 1000
                          if t_first is not None and chunk_count > 1 else 0.0)
            self._save_call_stats(CallStats(
                output_tokens=local_output_tokens,
                ttft=local_ttft, tpot=local_tpot,
            ))

            return full_response, top_probability_list
    
    async def _stream_request_vllm_logits(self, messages, temperature=0, stream=False,
                                          handle: Optional[StreamHandle] = None):
        """vLLM logprobs request.  Streaming path uses *handle* for chunks."""
        h = handle or StreamHandle()
        reasoning_flag = False
        t0 = time.time()
        req_kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            stream=stream and self.stream,
            temperature=0,
            logprobs=True,
            top_logprobs=5,
        )
        if self.frequency_penalty is not None:
            req_kwargs["frequency_penalty"] = float(self.frequency_penalty)
        response = await self.client.chat.completions.create(**req_kwargs)
        if not (self.stream and stream):
            local_ttft = time.time() - t0
            local_tpot = 0.0
            local_input_tokens = 0
            local_output_tokens = 0
            local_cached_tokens = 0
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty or invalid response from LLM")
            if hasattr(response, 'usage') and response.usage:
                local_input_tokens = response.usage.prompt_tokens
                local_output_tokens = response.usage.completion_tokens
                local_cached_tokens = _extract_cached_input_tokens(response.usage)
                self.token_tracker(local_input_tokens, local_output_tokens)
                if local_output_tokens > 1:
                    local_tpot = local_ttft / local_output_tokens * 1000
            local_finish_reason = getattr(response.choices[0], "finish_reason", None) or "unknown"
            logprobs_list = response.choices[0].logprobs.content
            top_probability_list = []
            start_flag = False
            for i, token_data in enumerate(logprobs_list):
                token_i = token_data.token
                if token_i == "```":
                    start_flag = True
                if start_flag:
                    top_logprobs_dict_i = token_data.top_logprobs
                    top_probability_i = []
                    for top_lp in top_logprobs_dict_i:
                        probability = top_lp.logprob
                        top_probability_i.append(probability)

                    top_probability_list.append({
                        'token': token_i,
                        'top_logprobs': top_probability_i,
                    })

            self._save_call_stats(CallStats(
                input_tokens=local_input_tokens, output_tokens=local_output_tokens,
                ttft=local_ttft, tpot=local_tpot, finish_reason=local_finish_reason,
                input_cached_tokens=local_cached_tokens,
            ))
            return response.choices[0].message.content, top_probability_list

        else:
            collected_messages = []
            top_probability_list = []
            start_flag = False
            t_first = None
            t_last = t0
            chunk_count = 0
            local_output_tokens = 0
            self.token_tracker(len(messages), 0)
            try:
                async for chunk in response:
                    if h.interrupted:
                        logger.info("Generation interrupted via StreamHandle (vllm_logits)")
                        break

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    chunk_message = ""
                    if hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content:
                        chunk_message = choice.delta.reasoning_content or ""
                        if reasoning_flag == False:
                            chunk_message = "<think>" + chunk_message
                            reasoning_flag = True

                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        chunk_message = choice.delta.content or ""
                        if reasoning_flag == True:
                            chunk_message = "</think>" + chunk_message
                            reasoning_flag = False

                        if hasattr(choice, 'logprobs') and choice.logprobs and choice.logprobs.content:
                            for token_data in choice.logprobs.content:
                                token_i = token_data.token
                                if token_i == "```":
                                    start_flag = True

                                if start_flag:
                                    top_logprobs_dict_i = token_data.top_logprobs
                                    top_probability_i = []
                                    for top_lp in top_logprobs_dict_i:
                                        probability = top_lp.logprob
                                        top_probability_i.append(probability)

                                    top_probability_list.append({
                                        'token': token_i,
                                        'top_logprobs': top_probability_i,
                                    })

                    if chunk_message:
                        t_now = time.time()
                        if t_first is None:
                            t_first = t_now
                        t_last = t_now
                        chunk_count += 1

                    collected_messages.append(chunk_message)
                    num_tokens_estimated = self.estimate_tokens(chunk_message)
                    local_output_tokens += num_tokens_estimated
                    self.token_tracker(0, num_tokens_estimated)
                    await _chunked_put(h, chunk_message)

            finally:
                if hasattr(response, 'close'):
                    try:
                        await response.close()
                    except Exception:
                        pass
                h.finish()
            full_response = "".join(collected_messages).strip()
            if not full_response and not h.interrupted:
                raise ValueError("Empty response from streaming LLM")
            local_ttft = (t_first - t0) if t_first is not None else 0.0
            local_tpot = ((t_last - t_first) / (chunk_count - 1) * 1000
                          if t_first is not None and chunk_count > 1 else 0.0)
            self._save_call_stats(CallStats(
                output_tokens=local_output_tokens,
                ttft=local_ttft, tpot=local_tpot,
            ))

            return full_response, top_probability_list
