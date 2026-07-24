"""LongMemEval evaluation environment.

This module implements a benchmark-specific runner for LongMemEval with two
separate phases:

- ``answer``: retrieve memories and produce a hypothesis for one question.
- ``judge``: score the hypothesis against the gold answer using an independent
  LLM judge.

The two phases intentionally use separate prompts and entry points so the
evaluation flow stays aligned with the official benchmark semantics.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mindmemos_sdk.memory import AsyncMemoryClient, MemorySearchHit
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm

from mindmemos_eval.llm import LLMClient
from mindmemos_eval.memory.scorer import ScoreResult

LONGMEMEVAL_ANSWER_PROMPT = """
You answer LongMemEval questions using only the retrieved memories.

The benchmark covers questions about long-term user memory, temporal reasoning,
preference recall, knowledge updates, and multi-session facts.

Rules:
- Use only the retrieved memories.
- If the memories do not contain enough information, answer with "I don't know".
- If the question is abstention-oriented, do not guess.
- Keep the answer brief and factual.

Retrieved memories:
{context}

Question type:
{question_type}

Question date:
{question_date}

Question:
{question}

Return only the final answer inside <answer> and </answer>.
""".strip()

LONGMEMEVAL_JUDGE_DEFAULT_TEMPLATE = """I will give you a question, a correct answer, and a response from a model.
Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {gold_answer}

Model Response: {generated_answer}

Is the model response correct? Answer yes or no only."""

LONGMEMEVAL_JUDGE_TEMPORAL_TEMPLATE = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no.
If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Question: {question}

Correct Answer: {gold_answer}

Model Response: {generated_answer}

Is the model response correct? Answer yes or no only."""

LONGMEMEVAL_JUDGE_KNOWLEDGE_UPDATE_TEMPLATE = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no.
If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {question}

Correct Answer: {gold_answer}

Model Response: {generated_answer}

Is the model response correct? Answer yes or no only."""

LONGMEMEVAL_JUDGE_PREFERENCE_TEMPLATE = """I will give you a question, a rubric for desired personalized response, and a response from a model.
Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric.
The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: {question}

Rubric: {gold_answer}

Model Response: {generated_answer}

Is the model response correct? Answer yes or no only."""

LONGMEMEVAL_JUDGE_ABSTENTION_TEMPLATE = """I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable.
The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {question}

Explanation: {gold_answer}

Model Response: {generated_answer}

Does the model correctly identify the question as unanswerable? Answer yes or no only."""


def strip_speaker_prefix(content: str) -> str:
    """Remove simple speaker prefixes from dialogue text."""
    text = content or ""
    for prefix in ("User: ", "Assistant: ", "user: ", "assistant: "):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def session_timestamp_millis(raw_timestamp: str) -> int:
    """Parse a session timestamp into milliseconds since epoch."""
    from dateutil import parser as dateutil_parser

    normalized = re.sub(r"\s*\([A-Za-z]{3}\)\s*", " ", raw_timestamp).strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(normalized, fmt).replace(tzinfo=UTC)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    dt = dateutil_parser.parse(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.astimezone(UTC).timestamp() * 1000)


def _message_text(msg: Mapping[str, Any]) -> str:
    """Build one text payload for a dialogue turn."""
    text = str(msg.get("content") or msg.get("text") or "")
    return strip_speaker_prefix(text)


def _extract_answer(full_response: str) -> tuple[str, str]:
    """Extract the final answer and any pre-answer reasoning text."""
    answer = full_response
    if "<answer>" in answer:
        answer = answer.split("<answer>", 1)[1]
    if "</answer>" in answer:
        answer = answer.split("</answer>", 1)[0]
    chain_of_thought = full_response.split("<answer>", 1)[0].strip() if "<answer>" in full_response else ""
    return answer.strip(), chain_of_thought


def _format_memory_for_answering(hit: MemorySearchHit) -> str:
    """Format one memory hit for answer generation."""
    event_time = hit.event_time
    source_timestamp = hit.source_timestamp
    if not event_time and not source_timestamp:
        return hit.memory
    return (
        f"[event_time: {event_time or 'unknown time'}; "
        f"source_timestamp: {source_timestamp or 'unknown time'}] {hit.memory}"
    )


def _is_abstention_question(question_id: str) -> bool:
    """Return whether a question uses the official abstention suffix."""
    return question_id.endswith("_abs")


def _question_text(sample: Mapping[str, Any], question: Mapping[str, Any]) -> str:
    """Resolve a question string from a sample or nested QA row."""
    value = question.get("question")
    if value is None:
        value = sample.get("question", "")
    return str(value or "")


def _question_answer(question: Mapping[str, Any], sample: Mapping[str, Any]) -> str:
    """Resolve the gold answer from a question or sample."""
    value = question.get("answer")
    if value is None:
        value = sample.get("answer", "")
    return str(value or "")


def _question_type(question: Mapping[str, Any], sample: Mapping[str, Any]) -> str:
    """Resolve the benchmark question type."""
    value = question.get("question_type")
    if value is None:
        value = question.get("type")
    if value is None:
        value = sample.get("question_type")
    if value is None:
        value = sample.get("type")
    return str(value or "unknown")


def _question_date(question: Mapping[str, Any], sample: Mapping[str, Any]) -> str:
    """Resolve the question date used for prompt context."""
    value = question.get("question_date")
    if value is None:
        value = sample.get("question_date")
    return str(value or "")


def _question_id(sample: Mapping[str, Any], question: Mapping[str, Any], sample_index: int, question_index: int) -> str:
    """Resolve a stable question id for reporting."""
    value = question.get("question_id")
    if value is None:
        value = sample.get("question_id")
    if value is None:
        value = f"sample_{sample_index}_q_{question_index}"
    return str(value)


def _session_key(session: Mapping[str, Any], session_index: int, session_ids: Sequence[Any]) -> str:
    """Resolve a stable session key."""
    if session_index < len(session_ids):
        return str(session_ids[session_index])
    for key in ("session_id", "id", "haystack_session_id", "key"):
        value = session.get(key)
        if value is not None:
            return str(value)
    return f"session_{session_index}"


def _session_date(session: Mapping[str, Any], session_index: int, session_dates: Sequence[Any]) -> str:
    """Resolve the timestamp attached to one session."""
    if session_index < len(session_dates):
        return str(session_dates[session_index])
    for key in ("date_time", "session_date_time", "session_date", "date", "timestamp"):
        value = session.get(key)
        if value is not None:
            return str(value)
    return ""


def _iter_turns(session: Any) -> list[Mapping[str, Any]]:
    """Normalize a session into a list of turn mappings."""
    if isinstance(session, list):
        return [turn for turn in session if isinstance(turn, Mapping)]
    if isinstance(session, Mapping):
        for key in ("turns", "messages", "conversation", "dialogue"):
            turns = session.get(key)
            if isinstance(turns, list):
                return [turn for turn in turns if isinstance(turn, Mapping)]
        if "role" in session or "content" in session:
            return [session]
    return []


def _normalize_sessions(sample: Mapping[str, Any]) -> list[tuple[str, str, list[Mapping[str, Any]]]]:
    """Normalize the haystack session payload into addable sessions."""
    sessions_raw = sample.get("haystack_sessions") or sample.get("sessions") or []
    if isinstance(sessions_raw, Mapping):
        sessions_iterable: list[Any] = [sessions_raw]
    elif isinstance(sessions_raw, list):
        sessions_iterable = list(sessions_raw)
    else:
        sessions_iterable = []

    session_ids = sample.get("haystack_session_ids") or sample.get("session_ids") or []
    session_dates = sample.get("haystack_dates") or sample.get("session_dates") or []
    normalized: list[tuple[str, str, list[Mapping[str, Any]]]] = []
    for index, raw_session in enumerate(sessions_iterable):
        if isinstance(raw_session, Mapping):
            session_key = _session_key(raw_session, index, session_ids)
            session_date = _session_date(raw_session, index, session_dates)
            turns = _iter_turns(raw_session)
        else:
            session_key = _session_key({}, index, session_ids)
            session_date = _session_date({}, index, session_dates)
            turns = _iter_turns(raw_session)
        normalized.append((session_key, session_date, turns))
    return normalized


def _normalize_questions(sample: Mapping[str, Any], sample_index: int) -> list[Mapping[str, Any]]:
    """Normalize the sample into one or more question records."""
    qa_rows = sample.get("qa")
    if isinstance(qa_rows, list) and qa_rows:
        out: list[Mapping[str, Any]] = []
        for q_index, row in enumerate(qa_rows):
            if isinstance(row, Mapping):
                normalized = dict(row)
                normalized.setdefault(
                    "question_id", normalized.get("question_id") or f"sample_{sample_index}_q_{q_index}"
                )
                out.append(normalized)
            else:
                out.append({"question": str(row), "question_id": f"sample_{sample_index}_q_{q_index}"})
        return out

    question = sample.get("question")
    if question is None:
        return []
    return [
        {
            "question_id": sample.get("question_id") or f"sample_{sample_index}_q_0",
            "question": question,
            "answer": sample.get("answer", ""),
            "question_type": sample.get("question_type") or sample.get("type") or "unknown",
            "question_date": sample.get("question_date", ""),
        }
    ]


def build_answer_context(memories: list[str], sample: Mapping[str, Any], question: Mapping[str, Any]) -> str:
    """Format retrieved memories for the LongMemEval answer prompt."""
    lines: list[str] = []
    lines.append(f"Question id: {_question_id(sample, question, 0, 0)}")
    lines.append(f"Question type: {_question_type(question, sample)}")
    question_date = _question_date(question, sample)
    if question_date:
        lines.append(f"Question date: {question_date}")
    lines.append("")
    lines.append("Reference memories:")
    if not memories:
        lines.append("No relevant memories.")
        return "\n".join(lines)
    for index, memory in enumerate(memories, start=1):
        lines.append(f"{index}. {memory}")
    return "\n".join(lines)


def build_answer_prompt(memories: list[str], sample: Mapping[str, Any], question: Mapping[str, Any]) -> str:
    """Build the answer-generation prompt."""
    context = build_answer_context(memories, sample, question)
    return (
        LONGMEMEVAL_ANSWER_PROMPT.replace("{context}", context)
        .replace("{question}", _question_text(sample, question))
        .replace("{question_type}", _question_type(question, sample))
        .replace("{question_date}", _question_date(question, sample))
    )


def build_judge_prompt(sample: Mapping[str, Any], question: Mapping[str, Any], hypothesis: str) -> str:
    """Build the judge prompt used for independent scoring."""
    question_type = _question_type(question, sample)
    question_id = _question_id(sample, question, 0, 0)
    if _is_abstention_question(question_id):
        template = LONGMEMEVAL_JUDGE_ABSTENTION_TEMPLATE
    elif question_type in {"single-session-user", "single-session-assistant", "multi-session"}:
        template = LONGMEMEVAL_JUDGE_DEFAULT_TEMPLATE
    elif question_type == "temporal-reasoning":
        template = LONGMEMEVAL_JUDGE_TEMPORAL_TEMPLATE
    elif question_type == "knowledge-update":
        template = LONGMEMEVAL_JUDGE_KNOWLEDGE_UPDATE_TEMPLATE
    elif question_type == "single-session-preference":
        template = LONGMEMEVAL_JUDGE_PREFERENCE_TEMPLATE
    else:
        raise NotImplementedError(f"unsupported LongMemEval question_type: {question_type}")
    return (
        template.replace("{question}", _question_text(sample, question))
        .replace("{gold_answer}", _question_answer(question, sample))
        .replace("{generated_answer}", hypothesis)
    )


class LongMemEvalJudgeScorer:
    """LLM judge used by LongMemEval scoring."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def score(self, sample: Mapping[str, Any], question: Mapping[str, Any], hypothesis: str) -> ScoreResult:
        """Judge one hypothesis against the gold answer."""
        content = build_judge_prompt(sample, question, hypothesis)
        raw = (await self._llm.complete([{"role": "user", "content": content}], max_tokens=10)).content
        lowered = raw.strip().lower()
        passed = "yes" in lowered
        return ScoreResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=raw.strip() or ("yes" if passed else "no"),
            raw={"label": raw.strip()},
        )


class LongMemEvalAnswer(BaseModel):
    """Result for the answer-generation stage."""

    model_config = ConfigDict(extra="ignore")

    question_id: str
    question_type: str
    question: str
    question_date: str = ""
    hypothesis: str
    response_raw: str
    chain_of_thought: str = ""
    memories: list[str] = Field(default_factory=list)
    search_time: float = 0.0
    prompt: str = ""


class LongMemEvalQAResult(BaseModel):
    """Result for one question."""

    model_config = ConfigDict(extra="ignore")

    question_id: str
    question_type: str
    question_date: str = ""
    question: str
    gold_answer: str
    hypothesis: str
    response: str
    chain_of_thought: str = ""
    memory: list[str] = Field(default_factory=list)
    search_time: float = 0.0
    score: ScoreResult | None = None
    abstention: bool = False


class LongMemEvalAddSummary(BaseModel):
    """Summary for the add phase of one sample."""

    model_config = ConfigDict(extra="ignore")

    sample_idx: int
    sample_id: str
    user_id: str
    total_sessions: int
    added_sessions: int
    failed_sessions: list[tuple[str, str]] = Field(default_factory=list)


class LongMemEvalSampleResult(BaseModel):
    """Full result for one LongMemEval sample."""

    model_config = ConfigDict(extra="ignore")

    sample_idx: int
    sample_id: str
    user_id: str
    num_questions: int
    qa_results: list[LongMemEvalQAResult] = Field(default_factory=list)
    add_summary: LongMemEvalAddSummary | None = None


class LongMemEvalMetricBucket(BaseModel):
    """Aggregate counts for one question type."""

    model_config = ConfigDict(extra="ignore")

    count: int = 0
    correct: int = 0
    abstention_count: int = 0
    abstention_correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.count if self.count else 0.0

    @property
    def abstention_accuracy(self) -> float:
        return self.abstention_correct / self.abstention_count if self.abstention_count else 0.0


class LongMemEvalRunResult(BaseModel):
    """Full dataset result for LongMemEval."""

    model_config = ConfigDict(extra="ignore")

    samples: list[LongMemEvalSampleResult] = Field(default_factory=list)
    total_questions: int = 0
    correct: int = 0
    abstention_total: int = 0
    abstention_correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_questions if self.total_questions else 0.0

    @property
    def abstention_accuracy(self) -> float:
        return self.abstention_correct / self.abstention_total if self.abstention_total else 0.0

    def is_scored(self) -> bool:
        """Return whether any question has a score."""
        return any(qa.score is not None for sample in self.samples for qa in sample.qa_results)

    def by_type(self) -> dict[str, LongMemEvalMetricBucket]:
        """Aggregate counts and accuracies by question type."""
        buckets: dict[str, LongMemEvalMetricBucket] = {}
        for sample in self.samples:
            for qa in sample.qa_results:
                if qa.score is None:
                    continue
                bucket = buckets.setdefault(qa.question_type, LongMemEvalMetricBucket())
                bucket.count += 1
                bucket.correct += int(qa.score.passed)
                if qa.abstention:
                    bucket.abstention_count += 1
                    bucket.abstention_correct += int(qa.score.passed)
        return buckets

    def task_averaged_accuracy(self) -> float:
        """Return the mean accuracy across question types."""
        buckets = self.by_type()
        if not buckets:
            return 0.0
        return sum(bucket.accuracy for bucket in buckets.values()) / len(buckets)

    def official_metrics(self) -> dict[str, Any]:
        """Return the benchmark-facing summary metrics."""
        buckets = self.by_type()
        return {
            "by_type": {
                question_type: {
                    "count": bucket.count,
                    "correct": bucket.correct,
                    "accuracy": round(bucket.accuracy, 6),
                    "abstention_count": bucket.abstention_count,
                    "abstention_correct": bucket.abstention_correct,
                    "abstention_accuracy": round(bucket.abstention_accuracy, 6),
                }
                for question_type, bucket in sorted(buckets.items())
            },
            "task_averaged_accuracy": round(self.task_averaged_accuracy(), 6),
            "overall_accuracy": round(self.accuracy, 6),
            "abstention_accuracy": round(self.abstention_accuracy, 6),
            "total_questions": self.total_questions,
            "correct": self.correct,
            "abstention_total": self.abstention_total,
            "abstention_correct": self.abstention_correct,
        }

    @property
    def metrics(self) -> dict[str, Any]:
        """Expose the public metric summary."""
        return self.official_metrics()

    def format_metrics(self) -> str:
        """Format the per-type and official summary metrics."""
        lines: list[str] = []
        lines.append("By question type (count, accuracy):")
        buckets = self.by_type()
        if not buckets:
            lines.append("  (no scored questions)")
        for question_type in sorted(buckets):
            bucket = buckets[question_type]
            lines.append(
                f"  {question_type}: n={bucket.count} acc={bucket.accuracy:.4f} ({bucket.correct}/{bucket.count})"
            )
            if bucket.abstention_count:
                lines.append(
                    f"    abstention: n={bucket.abstention_count} acc={bucket.abstention_accuracy:.4f} "
                    f"({bucket.abstention_correct}/{bucket.abstention_count})"
                )
        lines.append("Official summary:")
        lines.append(f"  task_avg_acc={self.task_averaged_accuracy():.4f}")
        lines.append(f"  overall_acc={self.accuracy:.4f} ({self.correct}/{self.total_questions})")
        if self.abstention_total:
            lines.append(
                f"  abstention_acc={self.abstention_accuracy:.4f} ({self.abstention_correct}/{self.abstention_total})"
            )
        return "\n".join(lines)

    def format_report(self) -> str:
        """Format a human-readable run report."""
        lines: list[str] = ["=" * 60, "LongMemEval evaluation report", "=" * 60]
        for sample in self.samples:
            summary = sample.add_summary
            added = f"{summary.added_sessions}/{summary.total_sessions} sessions" if summary else "add skipped"
            lines.append(f"  {sample.sample_id}: {sample.num_questions} questions, {added}")
            if summary and summary.failed_sessions:
                for session, reason in summary.failed_sessions:
                    lines.append(f"    - add failed: {session}: {reason}")
        lines.append("-" * 60)
        if not self.is_scored():
            answered = sum(sample.num_questions for sample in self.samples)
            lines.append(f"Answered {answered} questions (scoring skipped).")
        else:
            lines.append(self.format_metrics())
        lines.append("=" * 60)
        return "\n".join(lines)

    def official_answers(self) -> list[dict[str, str]]:
        """Return the official submission rows."""
        rows: list[dict[str, str]] = []
        for sample in self.samples:
            for qa in sample.qa_results:
                rows.append({"question_id": qa.question_id, "hypothesis": qa.hypothesis})
        return rows


class LongMemEvalEnv:
    """LongMemEval memory evaluation environment."""

    def __init__(
        self,
        memory: AsyncMemoryClient,
        *,
        answer_llm: LLMClient,
        judge_llm: LLMClient | None = None,
        judge_scorer: LongMemEvalJudgeScorer | None = None,
        top_k: int | None = 50,
        search_strategy: str = "fast",
        rerank: bool = False,
    ) -> None:
        """Create a new LongMemEval environment."""
        self._memory = memory
        self._answer_llm = answer_llm
        self._judge_scorer = judge_scorer or LongMemEvalJudgeScorer(judge_llm or answer_llm)
        self._top_k = top_k
        self._search_strategy = search_strategy
        self._rerank = rerank

    async def add_session(
        self,
        user_id: str,
        messages: list[Mapping[str, Any]],
        raw_timestamp: str,
        *,
        session_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add one session to the memory store."""
        ts_millis = session_timestamp_millis(raw_timestamp)
        dict_messages = [
            {
                "role": str(msg.get("role") or msg.get("speaker") or "user"),
                "content": _message_text(msg),
                "timestamp": ts_millis,
            }
            for msg in messages
            if _message_text(msg)
        ]
        if not dict_messages:
            return
        payload = dict(metadata or {})
        if session_key is not None:
            payload.setdefault("longmemeval_session_key", session_key)
        await self._memory.add(
            dict_messages,
            user_id=user_id,
            mode="sync",
            session_id=session_key or user_id,
            metadata=payload or None,
        )

    async def add_sample(
        self,
        sample: Mapping[str, Any],
        sample_index: int,
        *,
        session_limit: int | None = None,
        on_session_done: Callable[[], None] | None = None,
    ) -> LongMemEvalAddSummary:
        """Add all sessions for one benchmark sample."""
        sample_id = str(sample.get("question_id") or f"sample_{sample_index}")
        user_id = f"lme_{sample_id}"
        sessions = _normalize_sessions(sample)
        if session_limit is not None:
            sessions = sessions[:session_limit]

        added = 0
        failed: list[tuple[str, str]] = []
        for session_key, session_date, turns in sessions:
            try:
                if not session_date:
                    failed.append((session_key, "missing_timestamp"))
                    continue
                if not turns:
                    continue
                metadata = {
                    "longmemeval_sample_id": sample_id,
                    "longmemeval_question_type": str(sample.get("question_type") or sample.get("type") or "unknown"),
                }
                await self.add_session(
                    user_id,
                    turns,
                    session_date,
                    session_key=session_key,
                    metadata=metadata,
                )
                added += 1
            except Exception as exc:  # noqa: BLE001 - benchmark runners should keep going on one bad session
                failed.append((session_key, f"{type(exc).__name__}: {exc}"))
            finally:
                if on_session_done is not None:
                    on_session_done()

        return LongMemEvalAddSummary(
            sample_idx=sample_index,
            sample_id=sample_id,
            user_id=user_id,
            total_sessions=len(sessions),
            added_sessions=added,
            failed_sessions=failed,
        )

    async def answer(self, user_id: str, sample: Mapping[str, Any], question: Mapping[str, Any]) -> LongMemEvalAnswer:
        """Generate a hypothesis for one LongMemEval question."""
        start = time.time()
        question_text = _question_text(sample, question)
        search = await self._memory.search(
            question_text,
            user_id=user_id,
            top_k=self._top_k,
            search_strategy=self._search_strategy,
            rerank=self._rerank,
            filters={"user_id": user_id},
            session_id=user_id,
        )
        memories = [_format_memory_for_answering(hit) for hit in search.memories]
        search_time = time.time() - start
        prompt = build_answer_prompt(memories, sample, question)
        full_response = (await self._answer_llm.complete([{"role": "user", "content": prompt}])).content
        hypothesis, chain_of_thought = _extract_answer(full_response)
        return LongMemEvalAnswer(
            question_id=_question_id(sample, question, 0, 0),
            question_type=_question_type(question, sample),
            question=question_text,
            question_date=_question_date(question, sample),
            hypothesis=hypothesis,
            response_raw=full_response,
            chain_of_thought=chain_of_thought,
            memories=memories,
            search_time=search_time,
            prompt=prompt,
        )

    async def judge(self, sample: Mapping[str, Any], question: Mapping[str, Any], hypothesis: str) -> ScoreResult:
        """Judge a hypothesis with an independent LLM call."""
        return await self._judge_scorer.score(sample, question, hypothesis)

    async def evaluate_question(
        self,
        user_id: str,
        sample: Mapping[str, Any],
        question: Mapping[str, Any],
        *,
        score: bool = True,
    ) -> LongMemEvalQAResult:
        """Run answer generation and optional judging for one question."""
        answer = await self.answer(user_id, sample, question)
        score_result = await self.judge(sample, question, answer.hypothesis) if score else None
        return LongMemEvalQAResult(
            question_id=answer.question_id,
            question_type=answer.question_type,
            question_date=answer.question_date,
            question=answer.question,
            gold_answer=_question_answer(question, sample),
            hypothesis=answer.hypothesis,
            response=answer.hypothesis,
            chain_of_thought=answer.chain_of_thought,
            memory=answer.memories,
            search_time=answer.search_time,
            score=score_result,
            abstention=_is_abstention_question(answer.question_id),
        )

    @staticmethod
    def _sample_id(sample: Mapping[str, Any], sample_index: int) -> str:
        """Return the public sample identifier."""
        value = sample.get("question_id")
        if value is None:
            value = f"sample_{sample_index}"
        return str(value)

    async def run_dataset(
        self,
        data: Sequence[Mapping[str, Any]],
        *,
        max_sample_concurrency: int = 4,
        max_qa_concurrency: int = 20,
        max_search_concurrency: int | None = None,
        max_score_concurrency: int | None = None,
        session_limit: int | None = None,
        add: bool = True,
        score: bool = True,
        print_report: bool = True,
        show_progress: bool = True,
    ) -> LongMemEvalRunResult:
        """Run a full LongMemEval dataset."""
        if session_limit is not None and session_limit < 1:
            raise ValueError("session_limit must be at least 1")

        search_sem = asyncio.Semaphore(max_search_concurrency or max_qa_concurrency)
        score_sem = asyncio.Semaphore(max_score_concurrency or max_qa_concurrency)
        sample_sem = asyncio.Semaphore(max_sample_concurrency)

        total_sessions = (
            sum(min(len(_normalize_sessions(sample)), session_limit) for sample in data)
            if add and session_limit is not None
            else sum(len(_normalize_sessions(sample)) for sample in data)
            if add
            else 0
        )
        total_questions = sum(len(_normalize_questions(sample, idx)) for idx, sample in enumerate(data))

        add_pbar = (
            tqdm(total=total_sessions, desc="Adding memories (session)", unit="session", position=0)
            if show_progress and add
            else None
        )
        sample_pbar = (
            tqdm(total=len(data), desc="Evaluating samples", unit="sample", position=1) if show_progress else None
        )
        qa_pbar = (
            tqdm(total=total_questions, desc="Answering questions", unit="q", position=2) if show_progress else None
        )

        async def run_sample(sample_index: int, sample: Mapping[str, Any]) -> LongMemEvalSampleResult:
            async with sample_sem:
                sample_id = self._sample_id(sample, sample_index)
                user_id = f"lme_{sample_id}"
                on_session_done = add_pbar.update if add_pbar is not None else None
                add_summary = (
                    await self.add_sample(
                        sample,
                        sample_index,
                        session_limit=session_limit,
                        on_session_done=on_session_done,
                    )
                    if add
                    else None
                )
                qa_rows = _normalize_questions(sample, sample_index)

                async def run_question(question: Mapping[str, Any]) -> LongMemEvalQAResult:
                    async with search_sem:
                        answer = await self.answer(user_id, sample, question)
                    if score:
                        async with score_sem:
                            score_result = await self.judge(sample, question, answer.hypothesis)
                    else:
                        score_result = None
                    if qa_pbar is not None:
                        qa_pbar.update()
                    return LongMemEvalQAResult(
                        question_id=answer.question_id,
                        question_type=answer.question_type,
                        question_date=answer.question_date,
                        question=answer.question,
                        gold_answer=_question_answer(question, sample),
                        hypothesis=answer.hypothesis,
                        response=answer.hypothesis,
                        chain_of_thought=answer.chain_of_thought,
                        memory=answer.memories,
                        search_time=answer.search_time,
                        score=score_result,
                        abstention=_is_abstention_question(answer.question_id),
                    )

                qa_results = await asyncio.gather(*(run_question(question) for question in qa_rows)) if qa_rows else []
                if sample_pbar is not None:
                    sample_pbar.update()
                return LongMemEvalSampleResult(
                    sample_idx=sample_index,
                    sample_id=sample_id,
                    user_id=user_id,
                    num_questions=len(qa_rows),
                    qa_results=list(qa_results),
                    add_summary=add_summary,
                )

        try:
            samples = await asyncio.gather(*(run_sample(index, sample) for index, sample in enumerate(data)))
        finally:
            for pbar in (qa_pbar, sample_pbar, add_pbar):
                if pbar is not None:
                    pbar.close()

        samples = list(samples)
        # total_questions 来自第 799 行，是全部问题数（不过滤）
        # 现在计算有评分和正确的数量
        scored_correct = 0
        abstention_total = 0
        abstention_correct = 0
        for sample in samples:
            for qa in sample.qa_results:
                if qa.score is None:
                    continue
                scored_correct += int(qa.score.passed)
                if qa.abstention:
                    abstention_total += 1
                    abstention_correct += int(qa.score.passed)

        run = LongMemEvalRunResult(
            samples=samples,
            total_questions=total_questions,  # ✅ 使用第 799 行的全部问题数
            correct=scored_correct,
            abstention_total=abstention_total,
            abstention_correct=abstention_correct,
        )
        if print_report:
            print(run.format_report(), flush=True)
        return run

    @staticmethod
    def load_dataset(path: str | Path) -> list[dict[str, Any]]:
        """Load a LongMemEval dataset from JSON or JSONL."""
        text = Path(path).read_text(encoding="utf-8")
        stripped = text.lstrip()
        if not stripped:
            return []
        if stripped.startswith("["):
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("LongMemEval dataset JSON must be a list")
            return [dict(item) for item in data if isinstance(item, Mapping)]
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(dict(json.loads(line)))
        return rows


def save_longmemeval_results(
    output_path: str | Path,
    run: LongMemEvalRunResult,
    *,
    official_answers_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> None:
    """Save the full run plus benchmark-facing output files."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(run.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    if official_answers_path is not None:
        answers_file = Path(official_answers_path)
        answers_file.parent.mkdir(parents=True, exist_ok=True)
        with answers_file.open("w", encoding="utf-8") as fh:
            for row in run.official_answers():
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if metrics_path is not None:
        metrics_file = Path(metrics_path)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(json.dumps(run.metrics, ensure_ascii=False, indent=2), encoding="utf-8")
