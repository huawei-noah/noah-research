"""Common tool-calling agent primitives for evaluations."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]
LLMCallable = Callable[[list[Message], list[dict[str, Any]]], Awaitable[Message]]


@dataclass
class Tool:
    """Callable tool exposed through an OpenAI-compatible JSON schema."""

    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    deliver_result_as_user: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)


@dataclass
class ParsedToolCall:
    """One tool call parsed from an assistant message."""

    name: str
    arguments: dict[str, Any]
    id: str = ""
    raw: Any = field(default=None, repr=False)


class OpenAIToolParser:
    """Parse native OpenAI ``tool_calls`` entries."""

    def parse(self, message: Message) -> list[ParsedToolCall]:
        calls: list[ParsedToolCall] = []
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            raw_args = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                arguments = {"__raw__": raw_args}
            calls.append(
                ParsedToolCall(
                    name=function.get("name", ""),
                    arguments=arguments or {},
                    id=call.get("id", ""),
                    raw=call,
                )
            )
        return calls


@dataclass
class RunResult:
    """Result of one agent rollout."""

    messages: list[Message]
    turns: int
    finished: bool

    @property
    def last_message(self) -> Message:
        return self.messages[-1]


class ReactAgent:
    """Small ReAct-style loop for OpenAI-compatible tool calling."""

    def __init__(
        self,
        llm: LLMCallable,
        tools: Iterable[Tool] | None = None,
        *,
        max_turns: int = 10,
        system_prompt: str | None = None,
        parser: OpenAIToolParser | None = None,
    ) -> None:
        self.llm = llm
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self.parser = parser or OpenAIToolParser()

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self.tools.values()]

    async def run(self, messages: list[Message]) -> RunResult:
        messages = list(messages)
        if self.system_prompt and not (messages and messages[0].get("role") == "system"):
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        finished = False
        turn = 0
        for turn in range(1, self.max_turns + 1):
            assistant = await self.llm(messages, self.tool_schemas)
            messages.append(assistant)

            calls = self.parser.parse(assistant)
            if not calls:
                finished = True
                break

            for call in calls:
                messages.extend(await self._execute(call))

        return RunResult(messages=messages, turns=turn, finished=finished)

    async def _execute(self, call: ParsedToolCall) -> list[Message]:
        tool = self.tools.get(call.name)
        if tool is None:
            content = f"Error: unknown tool '{call.name}'"
        else:
            try:
                if inspect.iscoroutinefunction(tool.func):
                    result = await tool(**call.arguments)
                else:
                    result = await asyncio.to_thread(tool, **call.arguments)
                content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            except Exception as exc:  # noqa: BLE001 - tool errors are surfaced to the model
                content = f"Error: {type(exc).__name__}: {exc}"

        if tool is not None and tool.deliver_result_as_user:
            return [
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": f"Result of '{call.name}' delivered in the following user message.",
                },
                {"role": "user", "content": content},
            ]
        return [
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": content,
            }
        ]


__all__ = [
    "LLMCallable",
    "Message",
    "OpenAIToolParser",
    "ParsedToolCall",
    "ReactAgent",
    "RunResult",
    "Tool",
]
