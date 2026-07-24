"""Official PersonaMem v1 protocol adapted to the MindMemOS memory API."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from mindmemos_sdk.memory import AsyncMemoryClient
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm

from mindmemos_eval.llm import LLMClient

PERSONAMEM_OFFICIAL_REPOSITORY = "https://github.com/bowen-upenn/PersonaMem"
PERSONAMEM_OFFICIAL_PROTOCOL_COMMIT = "caaae44b3f236b8751d499a770e94e5aecffcff1"
PERSONAMEM_OFFICIAL_INSTRUCTION = (
    "Find the most appropriate model response and give your final answer "
    "(a), (b), (c), or (d) after the special token <final_answer>."
)
PersonaMemEvaluationMode = Literal["memory_rag", "official_full_context"]

_REQUIRED_QUESTION_FIELDS = {
    "persona_id",
    "question_id",
    "question_type",
    "topic",
    "user_question_or_message",
    "correct_answer",
    "all_options",
    "shared_context_id",
    "end_index_in_shared_context",
}


class PersonaMemScope(BaseModel):
    """One immutable question-visible context boundary."""

    model_config = ConfigDict(extra="forbid")

    shared_context_id: str
    end_index: int
    scope_id: str
    user_id: str
    session_id: str


class PersonaMemItem(BaseModel):
    """One official PersonaMem v1 multiple-choice question."""

    model_config = ConfigDict(extra="allow")

    index: int
    persona_id: str
    question_id: str
    question_type: str
    topic: str
    question: str
    correct_answer: str
    all_options: str
    scope: PersonaMemScope
    metadata: dict[str, Any] = Field(default_factory=dict)


class PersonaMemBuildSummary(BaseModel):
    """Build outcome for one unique visible-context scope."""

    scope: PersonaMemScope
    total_messages: int = 0
    added_messages: int = 0
    add_calls: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


class PersonaMemAnswer(BaseModel):
    """Answer output and evaluation-client usage for one question."""

    response: str
    extracted_answer: str
    is_correct: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0


class PersonaMemQAResult(BaseModel):
    """End-to-end result for one PersonaMem question."""

    item: PersonaMemItem
    retrieved_memories: list[str] = Field(default_factory=list)
    prompt: list[dict[str, Any]] = Field(default_factory=list)
    search_elapsed_seconds: float = 0.0
    answer: PersonaMemAnswer | None = None
    error: str | None = None


class PersonaMemRunResult(BaseModel):
    """Serializable result for one PersonaMem benchmark run."""

    protocol: str
    official_protocol_commit: str = PERSONAMEM_OFFICIAL_PROTOCOL_COMMIT
    benchmark_version: str = "v1"
    context_size: str = "32k"
    evaluation_mode: PersonaMemEvaluationMode
    items: list[PersonaMemItem] = Field(default_factory=list)
    build_summaries: list[PersonaMemBuildSummary] = Field(default_factory=list)
    qa_results: list[PersonaMemQAResult] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    def format_report(self) -> str:
        """Render the primary official score and failure counts."""
        return (
            "PersonaMem evaluation report\n"
            f"  protocol={self.protocol}\n"
            f"  accuracy={self.metrics.get('overall_accuracy', 0.0):.4f} "
            f"({self.metrics.get('correct', 0)}/{self.metrics.get('total', 0)})\n"
            f"  scopes={self.metrics.get('scope_build_success', 0)}/"
            f"{self.metrics.get('scope_total', 0)} built\n"
            f"  search_failures={self.metrics.get('search_failure_count', 0)} "
            f"answer_failures={self.metrics.get('answer_failure_count', 0)}"
        )


class PersonaMemContextStore:
    """Random-access reader for official shared-context JSONL files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._offsets = self._build_index()

    def _build_index(self) -> dict[str, int]:
        offsets: dict[str, int] = {}
        with self.path.open("rb") as fh:
            while True:
                offset = fh.tell()
                line = fh.readline()
                if not line:
                    break
                payload = json.loads(line)
                if not isinstance(payload, Mapping) or len(payload) != 1:
                    raise ValueError(f"invalid PersonaMem context line at byte {offset}")
                offsets[str(next(iter(payload)))] = offset
        return offsets

    def load(self, shared_context_id: str) -> list[dict[str, Any]]:
        """Load one context by the official shared-context identifier."""
        try:
            offset = self._offsets[shared_context_id]
        except KeyError as exc:
            raise KeyError(f"unknown PersonaMem shared_context_id: {shared_context_id}") from exc
        with self.path.open("rb") as fh:
            fh.seek(offset)
            payload = json.loads(fh.readline())
        context = next(iter(payload.values()))
        if not isinstance(context, list):
            raise ValueError(f"PersonaMem context {shared_context_id!r} must be a list")
        return [dict(message) for message in context if isinstance(message, Mapping)]

    def visible(self, scope: PersonaMemScope) -> list[dict[str, Any]]:
        """Apply the official exclusive end-index slice for one question scope."""
        context = self.load(scope.shared_context_id)
        if scope.end_index < 0 or scope.end_index > len(context):
            raise ValueError(
                f"invalid end_index {scope.end_index} for context "
                f"{scope.shared_context_id!r} with {len(context)} messages"
            )
        return context[: scope.end_index]


def load_personamem_questions(path: str | Path) -> list[dict[str, str]]:
    """Load and validate the official PersonaMem questions CSV."""
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        missing = sorted(_REQUIRED_QUESTION_FIELDS - fields)
        if missing:
            raise ValueError(f"PersonaMem questions CSV is missing fields: {', '.join(missing)}")
        return [dict(row) for row in reader]


def build_personamem_scope(shared_context_id: str, end_index: int) -> PersonaMemScope:
    """Create a stable, isolated memory scope from the official visibility boundary."""
    raw = f"{shared_context_id}\0{end_index}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    scope_id = f"{shared_context_id}:{end_index}"
    user_id = f"personamem-{digest}"
    return PersonaMemScope(
        shared_context_id=shared_context_id,
        end_index=end_index,
        scope_id=scope_id,
        user_id=user_id,
        session_id=user_id,
    )


def build_personamem_items(rows: Sequence[Mapping[str, Any]]) -> list[PersonaMemItem]:
    """Normalize official CSV rows while retaining analysis metadata."""
    items: list[PersonaMemItem] = []
    for index, row in enumerate(rows):
        shared_context_id = str(row["shared_context_id"])
        end_index = int(row["end_index_in_shared_context"])
        scope = build_personamem_scope(shared_context_id, end_index)
        metadata = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "persona_id",
                "question_id",
                "question_type",
                "topic",
                "user_question_or_message",
                "correct_answer",
                "all_options",
                "shared_context_id",
                "end_index_in_shared_context",
            }
        }
        items.append(
            PersonaMemItem(
                index=index,
                persona_id=str(row["persona_id"]),
                question_id=str(row["question_id"]),
                question_type=str(row["question_type"]),
                topic=str(row["topic"]),
                question=str(row["user_question_or_message"]),
                correct_answer=str(row["correct_answer"]),
                all_options=str(row["all_options"]),
                scope=scope,
                metadata=metadata,
            )
        )
    return items


def build_personamem_prompt(
    item: PersonaMemItem,
    *,
    retrieved_memories: Sequence[str] | None = None,
    visible_context: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build either the official full-context prompt or the memory-RAG adaptation."""
    query = f"{item.question}\n\n{PERSONAMEM_OFFICIAL_INSTRUCTION}\n\n{item.all_options}"
    if visible_context is not None:
        return [dict(message) for message in visible_context] + [{"role": "user", "content": query}]

    memories = list(retrieved_memories or [])
    memory_text = "\n".join(f"[{index}] {text}" for index, text in enumerate(memories, start=1))
    context = (
        "Use the retrieved user memories below to select the most appropriate response.\n\n"
        f"Retrieved memories:\n{memory_text or '(none)'}"
    )
    return [
        {"role": "system", "content": context},
        {"role": "user", "content": query},
    ]


def convert_personamem_system_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mirror the official system-to-user conversion used for OpenAI o-series names."""
    converted: list[dict[str, Any]] = []
    system_buffer = ""
    for message in messages:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "system":
            system_buffer += f"[System]: {content}\n"
            continue
        if system_buffer:
            content = system_buffer + content
            system_buffer = ""
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"] += "\n" + content
        else:
            converted.append({"role": role, "content": content})
    return converted


def extract_personamem_answer(response: str, correct_answer: str) -> tuple[bool, str]:
    """Apply the official PersonaMem v1 option extraction and correctness rule."""

    def _extract_only_options(text: str) -> set[str]:
        lowered = text.lower()
        in_parens = re.findall(r"\(([a-d])\)", lowered)
        if in_parens:
            return set(in_parens)
        return set(re.findall(r"\b([a-d])\b", lowered))

    correct = correct_answer.lower().strip("() ")
    full_response = response
    predicted = response.strip()
    if "<final_answer>" in predicted:
        predicted = predicted.split("<final_answer>")[-1].strip()
    if predicted.endswith("</final_answer>"):
        predicted = predicted[: -len("</final_answer>")].strip()

    predicted_options = _extract_only_options(predicted)
    if predicted_options == {correct}:
        return True, predicted
    if _extract_only_options(full_response) == {correct}:
        return True, predicted
    return False, predicted


def calculate_personamem_metrics(
    results: Sequence[PersonaMemQAResult],
    build_summaries: Sequence[PersonaMemBuildSummary],
    *,
    total_elapsed_seconds: float,
) -> dict[str, Any]:
    """Aggregate the official accuracy plus operational diagnostics."""
    total = len(results)
    correct = sum(1 for result in results if result.answer and result.answer.is_correct)
    by_question_type: dict[str, list[bool]] = defaultdict(list)
    by_topic: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        is_correct = bool(result.answer and result.answer.is_correct)
        by_question_type[result.item.question_type].append(is_correct)
        by_topic[result.item.topic].append(is_correct)

    def _breakdown(groups: Mapping[str, Sequence[bool]]) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "correct": sum(values),
                "total": len(values),
                "accuracy": sum(values) / len(values) if values else 0.0,
            }
            for name, values in sorted(groups.items())
        }

    answers = [result.answer for result in results if result.answer is not None]
    search_failure_count = sum(1 for result in results if result.error and result.error.startswith("search failed:"))
    answer_failure_count = sum(1 for result in results if result.error and result.error.startswith("answer failed:"))
    build_elapsed = sum(summary.elapsed_seconds for summary in build_summaries)
    search_elapsed = sum(result.search_elapsed_seconds for result in results)
    answer_elapsed = sum(answer.elapsed_seconds for answer in answers)
    return {
        "overall_accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "by_question_type": _breakdown(by_question_type),
        "by_topic": _breakdown(by_topic),
        "scope_total": len(build_summaries),
        "scope_build_success": sum(1 for summary in build_summaries if summary.error is None),
        "scope_build_failure": sum(1 for summary in build_summaries if summary.error is not None),
        "scope_violation_count": 0,
        "search_failure_count": search_failure_count,
        "answer_failure_count": answer_failure_count,
        "answer_llm_calls": len(answers),
        "answer_prompt_tokens": sum(answer.prompt_tokens for answer in answers),
        "answer_completion_tokens": sum(answer.completion_tokens for answer in answers),
        "answer_total_tokens": sum(answer.total_tokens for answer in answers),
        "build_elapsed_seconds": build_elapsed,
        "search_elapsed_seconds": search_elapsed,
        "answer_elapsed_seconds": answer_elapsed,
        "total_elapsed_seconds": total_elapsed_seconds,
    }


class PersonaMemEnv:
    """Run official PersonaMem v1 questions against MindMemOS or full context."""

    def __init__(
        self,
        memory: AsyncMemoryClient,
        *,
        answer_llm: LLMClient,
        context_store: PersonaMemContextStore,
        evaluation_mode: PersonaMemEvaluationMode = "memory_rag",
        context_size: str = "32k",
        top_k: int = 50,
        search_strategy: str = "fast",
        rerank: bool = False,
        add_batch_size: int = 50,
    ) -> None:
        self._memory = memory
        self._answer_llm = answer_llm
        self._context_store = context_store
        self._evaluation_mode = evaluation_mode
        self._context_size = context_size
        self._top_k = top_k
        self._search_strategy = search_strategy
        self._rerank = rerank
        self._add_batch_size = add_batch_size

    @staticmethod
    def load_items(path: str | Path) -> list[PersonaMemItem]:
        """Load official question rows as normalized benchmark items."""
        return build_personamem_items(load_personamem_questions(path))

    async def _build_scope(self, scope: PersonaMemScope) -> PersonaMemBuildSummary:
        started = time.monotonic()
        visible = self._context_store.visible(scope)
        messages = [
            {
                "role": str(message.get("role") or "user"),
                "content": str(message.get("content") or ""),
                "timestamp": 1767225600000 + index * 60000,
            }
            for index, message in enumerate(visible)
            if str(message.get("content") or "").strip()
        ]
        added_messages = 0
        add_calls = 0
        try:
            for start in range(0, len(messages), self._add_batch_size):
                batch = messages[start : start + self._add_batch_size]
                await self._memory.add(
                    batch,
                    user_id=scope.user_id,
                    session_id=scope.session_id,
                    mode="sync",
                    metadata={
                        "benchmark": "personamem",
                        "shared_context_id": scope.shared_context_id,
                        "end_index_in_shared_context": scope.end_index,
                    },
                )
                add_calls += 1
                added_messages += len(batch)
            return PersonaMemBuildSummary(
                scope=scope,
                total_messages=len(messages),
                added_messages=added_messages,
                add_calls=add_calls,
                elapsed_seconds=time.monotonic() - started,
            )
        except Exception as exc:  # noqa: BLE001 - one bad scope must not discard the full run
            return PersonaMemBuildSummary(
                scope=scope,
                total_messages=len(messages),
                added_messages=added_messages,
                add_calls=add_calls,
                elapsed_seconds=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _answer_item(
        self,
        item: PersonaMemItem,
        *,
        build_error: str | None,
    ) -> PersonaMemQAResult:
        if build_error and self._evaluation_mode == "memory_rag":
            return PersonaMemQAResult(item=item, error=f"build failed: {build_error}")

        memories: list[str] = []
        search_elapsed = 0.0
        if self._evaluation_mode == "memory_rag":
            search_started = time.monotonic()
            try:
                search = await self._memory.search(
                    item.question,
                    user_id=item.scope.user_id,
                    session_id=item.scope.session_id,
                    top_k=self._top_k,
                    search_strategy=self._search_strategy,
                    rerank=self._rerank,
                )
                memories = [hit.memory for hit in search.memories if hit.memory]
            except Exception as exc:  # noqa: BLE001 - failures remain in the official denominator
                return PersonaMemQAResult(
                    item=item,
                    search_elapsed_seconds=time.monotonic() - search_started,
                    error=f"search failed: {type(exc).__name__}: {exc}",
                )
            search_elapsed = time.monotonic() - search_started
            prompt = build_personamem_prompt(item, retrieved_memories=memories)
        else:
            prompt = build_personamem_prompt(item, visible_context=self._context_store.visible(item.scope))

        if "o" in self._answer_llm.config.model:
            prompt = convert_personamem_system_messages(prompt)

        answer_started = time.monotonic()
        try:
            completion = await self._answer_llm.complete(prompt)
        except Exception as exc:  # noqa: BLE001 - failures remain in the official denominator
            return PersonaMemQAResult(
                item=item,
                retrieved_memories=memories,
                prompt=prompt,
                search_elapsed_seconds=search_elapsed,
                error=f"answer failed: {type(exc).__name__}: {exc}",
            )
        answer_elapsed = time.monotonic() - answer_started
        is_correct, extracted = extract_personamem_answer(completion.content, item.correct_answer)
        return PersonaMemQAResult(
            item=item,
            retrieved_memories=memories,
            prompt=prompt,
            search_elapsed_seconds=search_elapsed,
            answer=PersonaMemAnswer(
                response=completion.content,
                extracted_answer=extracted,
                is_correct=is_correct,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                total_tokens=completion.total_tokens,
                elapsed_seconds=answer_elapsed,
            ),
        )

    async def run_dataset(
        self,
        items: Sequence[PersonaMemItem],
        *,
        max_build_concurrency: int = 2,
        max_qa_concurrency: int = 4,
        add: bool = True,
        score: bool = True,
        show_progress: bool = True,
    ) -> PersonaMemRunResult:
        """Build unique official scopes, answer questions, and score deterministically."""
        del score  # PersonaMem scoring is deterministic and always accompanies an answer.
        started = time.monotonic()
        scopes = {item.scope.scope_id: item.scope for item in items}
        build_summaries: list[PersonaMemBuildSummary] = []

        if self._evaluation_mode == "memory_rag":
            build_sem = asyncio.Semaphore(max_build_concurrency)
            build_pbar = tqdm(
                total=len(scopes), disable=not show_progress, desc="Building PersonaMem scopes", unit="scope"
            )

            async def _build(scope: PersonaMemScope) -> PersonaMemBuildSummary:
                async with build_sem:
                    if add:
                        summary = await self._build_scope(scope)
                    else:
                        summary = PersonaMemBuildSummary(scope=scope)
                    build_pbar.update()
                    return summary

            build_summaries = list(await asyncio.gather(*(_build(scope) for scope in scopes.values())))
            build_pbar.close()

        build_errors = {summary.scope.scope_id: summary.error for summary in build_summaries}
        qa_sem = asyncio.Semaphore(max_qa_concurrency)
        qa_pbar = tqdm(total=len(items), disable=not show_progress, desc="Evaluating PersonaMem", unit="question")

        async def _answer(item: PersonaMemItem) -> PersonaMemQAResult:
            async with qa_sem:
                result = await self._answer_item(item, build_error=build_errors.get(item.scope.scope_id))
                qa_pbar.update()
                return result

        results = list(await asyncio.gather(*(_answer(item) for item in items)))
        qa_pbar.close()
        total_elapsed = time.monotonic() - started
        protocol = f"personamem-v1-{'memory-rag' if self._evaluation_mode == 'memory_rag' else 'official-full-context'}"
        metrics = calculate_personamem_metrics(results, build_summaries, total_elapsed_seconds=total_elapsed)
        return PersonaMemRunResult(
            protocol=protocol,
            context_size=self._context_size,
            evaluation_mode=self._evaluation_mode,
            items=list(items),
            build_summaries=build_summaries,
            qa_results=results,
            metrics=metrics,
        )
