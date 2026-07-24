"""LLM signal detection for implicit feedback."""

from __future__ import annotations

import json

from ...llm import LLMClient, get_llm_client
from ...prompts.EN.feedback import SIGNAL_DETECTION_PROMPT
from ...typing import ImplicitFeedbackSessionMaterial, ImplicitFeedbackSignalResult


class ImplicitFeedbackSignalDetector:
    """Detect negative feedback rounds from session conversation only."""

    def __init__(self, *, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    async def detect(self, material: ImplicitFeedbackSessionMaterial) -> ImplicitFeedbackSignalResult:
        payload = {
            "session_id": material.session_id,
            "rounds": [
                {"round_index": index, "messages": round_.messages} for index, round_ in enumerate(material.rounds)
            ],
        }
        response = await self._client.chat(
            task="feedback.implicit.detect_signals",
            messages=[
                {"role": "system", "content": SIGNAL_DETECTION_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            format_parser=_parse_signal_result,
            temperature=0,
        )
        result = response.parsed
        if not isinstance(result, ImplicitFeedbackSignalResult):
            msg = "implicit feedback detector expected parsed signal result"
            raise TypeError(msg)
        return result

    @property
    def _client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client


def _parse_signal_result(content: str) -> ImplicitFeedbackSignalResult:
    return ImplicitFeedbackSignalResult.model_validate_json(_json_object_text(content))


def _json_object_text(content: str) -> str:
    text = content.strip()
    try:
        json.loads(text)
        return text
    except ValueError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        return text[start : end + 1]
