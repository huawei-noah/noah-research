"""Turn grouping for vanilla add pipeline chunking.

Converts raw messages into ordered turns with boundary metadata.
A turn represents either one user intent with assistant response(s), or one
multi-speaker exchange when arbitrary speaker names are used as roles.
"""

from __future__ import annotations

from ....config import VanillaAddConfig
from ....typing import DialogueMessage, TextMessage, Turn, TurnBoundary, TurnMessageRef

_STANDARD_ROLES = {"user", "assistant", "system", "tool"}


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text.

    Uses whitespace-split word count as a rough proxy for token count.
    Chinese text (no spaces between words) is estimated at ~1.5 chars per token.
    This is intentionally simple and deterministic — no LLM or tiktoken dependency.
    """
    if not text:
        return 0
    # Heuristic: count CJK characters and latin words separately
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿")
    non_cjk_text = "".join(" " if ("一" <= ch <= "鿿" or "㐀" <= ch <= "䶿") else ch for ch in text)
    latin_words = len(non_cjk_text.split())
    # CJK: ~1.5 chars per token; Latin: ~1 word per token
    return int(cjk / 1.5) + latin_words


class TurnGrouper:
    """Group messages into turns based on role patterns and time gaps.

    A user message starts a new turn unless the current turn has only user
    messages and no assistant response yet (consecutive user elaboration).
    Assistant messages join the current turn. An assistant-first request
    creates an open-head turn. Arbitrary role names are treated as speaker
    identities and grouped into exchange turns.
    """

    def __init__(self, config: VanillaAddConfig | None = None) -> None:
        self._gap_threshold = (config or VanillaAddConfig()).time_gap_threshold_seconds

    def group(
        self,
        messages: list[tuple[int, DialogueMessage | TextMessage]],
    ) -> list[Turn]:
        """Convert ordered messages into ordered turns.

        Args:
            messages: Ordered input messages paired with their original index
                in AddPipelineInput.messages. Use ``enumerate(inp.messages)``
                when no filtering is applied, or compute indices when filtering
                non-dialogue messages.

        Returns:
            Ordered list of Turn objects with boundary metadata.
        """
        if not messages:
            return []

        refs = self._to_refs(messages)
        if not refs:
            return []

        if any(ref.role == "speaker" for ref in refs):
            return self._group_multi_speaker(refs)

        turns: list[Turn] = []
        current_refs: list[TurnMessageRef] = []
        current_has_assistant = False
        current_has_user = False

        for i, ref in enumerate(refs):
            is_user = ref.role == "user"
            is_assistant = ref.role == "assistant"
            is_system = ref.role == "system"

            # System messages attach to current turn, never start one
            if is_system:
                if current_refs:
                    current_refs.append(ref)
                else:
                    # System message before any user/assistant — buffer it
                    current_refs.append(ref)
                continue

            # Check time gap for consecutive same-role messages
            if self._should_split_by_time_gap(ref, current_refs):
                turns.append(self._finalize_turn(current_refs))
                current_refs = []
                current_has_assistant = False
                current_has_user = False

            if is_user:
                if current_has_assistant or (current_has_user and not current_refs):
                    # Previous turn is complete (has assistant response), start new turn
                    # OR empty state — start fresh
                    if current_refs:
                        turns.append(self._finalize_turn(current_refs))
                        current_refs = []
                        current_has_assistant = False
                        current_has_user = False
                elif current_has_user and not current_has_assistant:
                    # Consecutive user messages before assistant — same turn (elaboration)
                    pass

                current_refs.append(ref)
                current_has_user = True

            elif is_assistant:
                if not current_has_user and not current_refs:
                    # Assistant before any user — open head
                    current_refs.append(ref)
                    current_has_assistant = True
                else:
                    current_refs.append(ref)
                    current_has_assistant = True

        if current_refs:
            turns.append(self._finalize_turn(current_refs))

        # Post-process: distinguish open_head from orphan.
        # if user messages appear later. Only all-assistant requests produce orphan.
        request_has_user = any(any(r.role == "user" for r in t.messages) for t in turns)
        for i, turn in enumerate(turns):
            turn_has_user = any(r.role == "user" for r in turn.messages)
            turn_has_assistant = any(r.role == "assistant" for r in turn.messages)
            if not turn_has_user and turn_has_assistant:
                if request_has_user and i == 0:
                    # First turn with only assistant, but user msgs follow → open_head
                    turn.boundary = "open_head"
                else:
                    turn.boundary = "orphan"

        return turns

    def _group_multi_speaker(self, refs: list[TurnMessageRef]) -> list[Turn]:
        """Group arbitrary named-speaker dialogue into exchange turns.

        Unknown roles carry speaker identity rather than user/assistant
        semantics. A new turn starts when a large time gap appears, or when a
        speaker repeats in the current exchange after another speaker has
        spoken. Consecutive same-speaker messages stay in the same exchange
        unless the time gap threshold splits them.
        """
        turns: list[Turn] = []
        current_refs: list[TurnMessageRef] = []
        speakers_in_exchange: set[str] = set()

        for ref in refs:
            if ref.role == "system":
                current_refs.append(ref)
                continue

            if self._should_split_any_time_gap(ref, current_refs) or self._should_split_by_speaker_repeat(
                ref,
                current_refs,
                speakers_in_exchange,
            ):
                turns.append(self._finalize_turn(current_refs))
                current_refs = []
                speakers_in_exchange = set()

            current_refs.append(ref)
            if ref.is_extractable:
                speakers_in_exchange.add(self._speaker_key(ref))

        if current_refs:
            turns.append(self._finalize_turn(current_refs))

        return turns

    def _should_split_any_time_gap(self, ref: TurnMessageRef, current: list[TurnMessageRef]) -> bool:
        if not current:
            return False
        last = current[-1]
        if ref.timestamp is None or last.timestamp is None:
            return False
        gap_sec = abs(ref.timestamp - last.timestamp) / 1000.0
        return gap_sec > self._gap_threshold

    def _should_split_by_speaker_repeat(
        self,
        ref: TurnMessageRef,
        current: list[TurnMessageRef],
        speakers_in_exchange: set[str],
    ) -> bool:
        if not current or not ref.is_extractable:
            return False
        speaker = self._speaker_key(ref)
        last_extractable = next((msg for msg in reversed(current) if msg.is_extractable), None)
        if last_extractable is not None and self._speaker_key(last_extractable) == speaker:
            return False
        return speaker in speakers_in_exchange

    def _speaker_key(self, ref: TurnMessageRef) -> str:
        return (ref.speaker or ref.raw_role or ref.role).strip().lower()

    def _to_refs(self, messages: list[tuple[int, DialogueMessage | TextMessage]]) -> list[TurnMessageRef]:
        """Convert input messages to TurnMessageRef list.

        Uses the original index from each tuple so that message_index always
        reflects the position in AddPipelineInput.messages, not the position
        in any filtered sublist.
        """
        refs: list[TurnMessageRef] = []
        for original_index, msg in messages:
            if isinstance(msg, TextMessage):
                text = msg.text
                role = "user"  # TextMessage has no role; default to user
                raw_role = None
                speaker = None
                ts = None
            else:
                text = msg.content
                raw_role = msg.role
                role, speaker = self._normalize_role(msg.role)
                ts = msg.timestamp

            if not text.strip():
                continue

            refs.append(
                TurnMessageRef(
                    text=text,
                    role=role,
                    raw_role=raw_role,
                    speaker=speaker,
                    timestamp=ts,
                    message_index=original_index,
                    is_extractable=(role != "system"),
                )
            )
        return refs

    def _normalize_role(self, role: str) -> tuple[str, str | None]:
        value = (role or "").strip()
        normalized = value.lower().replace("-", "_").replace(" ", "_")
        if normalized in _STANDARD_ROLES:
            return normalized, None
        return "speaker", value or None

    def _should_split_by_time_gap(self, ref: TurnMessageRef, current: list[TurnMessageRef]) -> bool:
        """Check if a time gap warrants splitting the current group.

        Only applies between consecutive same-role messages. A large gap
        suggests a new task or session phase.
        """
        if not current:
            return False
        last = current[-1]
        # Only check gap between same-role non-system messages
        if ref.role != last.role or ref.role == "system":
            return False
        if ref.timestamp is None or last.timestamp is None:
            return False
        gap_sec = abs(ref.timestamp - last.timestamp) / 1000.0
        return gap_sec > self._gap_threshold

    def _finalize_turn(self, refs: list[TurnMessageRef]) -> Turn:
        """Compute boundary and build a Turn from accumulated refs."""
        boundary = self._compute_boundary(refs)
        token_count = sum(_estimate_tokens(r.text) for r in refs)
        return Turn(messages=refs, boundary=boundary, token_count=token_count)

    def _compute_boundary(self, refs: list[TurnMessageRef]) -> TurnBoundary:
        """Determine turn boundary from message roles.

        COMPLETE: starts with user and has at least one assistant response, or
        a multi-speaker exchange contains at least two distinct speakers.
        OPEN_HEAD: starts with assistant before any user message.
        OPEN_TAIL: ends with user message without assistant response.
        ORPHAN: only assistant messages, no user context at all.
        """
        non_system = [r for r in refs if r.role != "system"]
        if not non_system:
            return "complete"

        has_user = any(r.role == "user" for r in non_system)
        has_assistant = any(r.role == "assistant" for r in non_system)
        has_speaker = any(r.role == "speaker" for r in non_system)
        speaker_count = len({self._speaker_key(r) for r in non_system if r.role == "speaker"})
        first_non_system_role = non_system[0].role
        last_non_system_role = non_system[-1].role

        if has_speaker:
            return "complete" if speaker_count >= 2 else "open_tail"
        if not has_user and has_assistant:
            return "orphan"
        if first_non_system_role == "assistant" and has_user:
            # Has user messages but starts with assistant — open head
            return "open_head"
        if first_non_system_role == "assistant" and not has_user:
            return "orphan"
        if has_user and not has_assistant:
            return "open_tail"
        # Starts with user, has assistant
        return "complete"
