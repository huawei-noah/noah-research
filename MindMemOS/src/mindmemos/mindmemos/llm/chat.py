"""Chat client backed by litellm.Router."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from ..logging import add_span_event, get_logger, traced
from ..typing import ChatResponse, Usage
from .router import dump_response, get_response_value, litellm_response_headers, usage_tokens

if TYPE_CHECKING:
    from litellm import Router

logger = get_logger(__name__)

# Generic correction prompt appended when ``feedback_on_parse_error`` is on: the
# failed reply plus the parser's error are fed back so the model can self-correct
# on the next attempt instead of re-running the identical prompt.
_PARSE_FEEDBACK_TEMPLATE = (
    "Your previous reply could not be applied:\n{error}\n\n"
    "Fix exactly that problem and resend the COMPLETE corrected output in the same "
    "format as before. Do not apologize or add commentary."
)


class LLMClient:
    """Chat client that routes requests through litellm.Router."""

    ALIAS = "chat"

    def __init__(self, router: Router, *, default_model: str | None = ALIAS, max_attempts: int = 3) -> None:
        """Wrap a pre-built litellm Router for chat calls.

        Args:
            router: Shared litellm Router; built and cached by the registry layer.
            default_model: Router alias to target, or ``None`` when no endpoint is
                configured (chat then raises a clear error).
            max_attempts: Maximum chat generations when ``format_parser`` rejects output.
        """
        self._router = router
        self._default_model = default_model
        self._format_parser_max_attempts = max(1, max_attempts)

    @traced("llm.chat")
    async def chat(
        self,
        task: str,
        messages: list[dict[str, Any]],
        format_parser: Callable[[str], Any] | None = None,
        *,
        model: str | None = None,
        feedback_on_parse_error: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Call the chat model and optionally parse the response."""
        target = model or self._default_model
        if target is None:
            msg = "No chat model endpoint configured"
            raise RuntimeError(msg)

        # Work on a private copy so feedback turns never mutate the caller's list.
        convo: list[dict[str, Any]] = list(messages)
        last_parse_err: Exception | None = None
        attempt = 0
        max_parse_attempts = self._format_parser_max_attempts if format_parser is not None else 1

        while True:
            if attempt >= max_parse_attempts:
                break

            start = perf_counter()
            add_span_event(
                "llm.chat.input",
                {
                    "task": task,
                    "model": target,
                    "attempt": attempt,
                    "messages": convo,
                    "kwargs": kwargs,
                },
            )
            try:
                resp = await self._router.acompletion(model=target, messages=convo, **kwargs)
            except Exception as exc:
                logger.info(
                    "litellm_call",
                    kind="chat",
                    task=task,
                    model=target,
                    status="error",
                    latency_ms=round((perf_counter() - start) * 1000, 2),
                    error=str(exc),
                )
                raise
            usage = usage_tokens(getattr(resp, "usage", None))
            headers = litellm_response_headers(resp)
            model_name = get_response_value(resp, "model", target) or target
            logger.info(
                "litellm_call",
                kind="chat",
                task=task,
                model=model_name,
                status="ok",
                latency_ms=round((perf_counter() - start) * 1000, 2),
                usage=usage.model_dump(),
                litellm_attempted_retries=headers.get("x-litellm-attempted-retries"),
                litellm_max_retries=headers.get("x-litellm-max-retries"),
            )

            choice = resp.choices[0]
            content = getattr(choice.message, "content", "") or ""
            finish = getattr(choice, "finish_reason", "") or ""
            add_span_event(
                "llm.chat.output",
                {
                    "task": task,
                    "model": model_name,
                    "attempt": attempt,
                    "finish_reason": finish,
                    "content": content,
                    "usage": usage,
                },
            )

            parsed: Any = None
            if format_parser is not None:
                try:
                    parsed = format_parser(content)
                except Exception as parse_err:
                    last_parse_err = parse_err
                    # Span event lands in ClickHouse otel_traces (Events.*) so the
                    # parse/validation failure rate is queryable by task & error type.
                    add_span_event(
                        "llm.chat.parse_error",
                        {
                            "task": task,
                            "model": target,
                            "attempt": attempt,
                            "error.type": type(parse_err).__name__,
                            "error": str(parse_err),
                            "feedback": feedback_on_parse_error,
                        },
                    )
                    if feedback_on_parse_error and attempt < max_parse_attempts - 1:
                        # Feed the failed reply + error back so the next attempt sees
                        # what was wrong instead of re-running the identical prompt.
                        convo.append({"role": "assistant", "content": content})
                        convo.append(
                            {
                                "role": "user",
                                "content": _PARSE_FEEDBACK_TEMPLATE.format(error=str(parse_err)),
                            }
                        )
                    attempt += 1
                    continue

            _annotate_llm_usage_span(task=task, model=model_name, usage=usage)
            return ChatResponse(
                finish_reason=finish,
                content=content,
                model=model_name,
                usage=usage,
                parsed=parsed,
                raw_response=dump_response(resp),
            )

        assert last_parse_err is not None
        raise last_parse_err


def _annotate_llm_usage_span(*, task: str, model: str, usage: Usage) -> None:
    """Attach model identity and token usage to the active ``llm.chat`` span."""

    span = trace.get_current_span()
    if not span.get_span_context().is_valid:
        return

    span.set_attribute("llm.task", task)
    span.set_attribute("llm.model", model)
    span.set_attribute("llm.usage.prompt_tokens", int(usage.prompt_tokens or 0))
    span.set_attribute("llm.usage.completion_tokens", int(usage.completion_tokens or 0))
    span.set_attribute("llm.usage.total_tokens", int(usage.total_tokens or 0))
