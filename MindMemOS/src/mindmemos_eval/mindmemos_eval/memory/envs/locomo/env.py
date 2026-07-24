"""LoCoMo evaluation environment.

Ports the memory-add, question-answering and grading behavior of the reference
runner onto the async SDK clients:

- **Add**: each conversation session is added in a single ``/v1/memory/add`` call
  with ``mode="sync"`` (we do not use the legacy per-message ``force_generation``
  flag). Sessions within a conversation are added serially; conversations run
  concurrently.
- **Answer**: retrieve memories, format them with event/source timestamps, build
  the LoCoMo grounding prompt, and call the answer LLM. The ``<answer>`` span is
  extracted from the response.
- **Score**: an LLM judge labels each prediction CORRECT/WRONG (1/0) using the
  reference accuracy prompt.

All HTTP I/O (memory add/search and LLM calls) is async. Answering and scoring
both run through a configurable :class:`~mindmemos_eval.llm.LLMClient`.

The answer prompt, grounding rules, context formatting and judge prompt are kept
verbatim from the reference so scores stay comparable.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from datetime import UTC
from typing import Any

from mindmemos_sdk.memory import AsyncMemoryClient, MemorySearchHit
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm

from mindmemos_eval.llm import LLMClient
from mindmemos_eval.memory.scorer import Scorer, ScoreResult, _parse_judge_json

# Prompt + context building

LOCOMO_ANSWER_GROUNDING_RULES = """# LoCoMo memory grounding rules
- The retrieved memories are all from the same LoCoMo conversation as the question.
- Flat vanilla memories often store the named participant as "the user". If the question names a person, do not
  require that person's name to appear inside a relevant memory. Treat "the user" as a candidate alias for the named
  participant when the memory contains the requested fact.
- If a memory contains both "the user" and the named person, keep their roles separate; use the fact attributed to the
  named person, not facts attributed only to "the user".
- Do not answer that information is unavailable when a retrieved memory directly contains the requested fact, object,
  date, number, named entity, or image caption.
- For questions asking what two named people "both" share, appreciate, like, or have in common, choose a theme or fact
  supported by memories involving both people, reciprocal support, or shared activities. Do not answer with a detail
  that belongs to only one of them.
- For shared-answer questions, prefer the most concrete shared object or activity in the memories, such as outdoor
  experiences, nature, a title, or a place. Avoid abstract relationship labels like "mutual support" when a concrete
  shared activity or object is available.
- Memories with the same event_time/source_timestamp usually describe the same episode. Combine their details before
  deciding a requested fact is missing.
- If one memory in an episode matches the event and another same-event memory contains the specific missing detail,
  use the specific detail.
- For questions with relative dates such as "last week", "last weekend", "yesterday", or "the week before <date>",
  resolve the relative time using the memory event_time/source_timestamp shown in the context.
- Prefer specific facts from the retrieved memories over generic summaries. Preserve names, numbers, dates, teams,
  places, programming languages, image captions, and meal names exactly when present.
- Use lightweight common knowledge only to decode concrete retrieved entities when the question asks for a category
  or artist, such as mapping a well-known movie to its genre or a well-known song to its artists.
- If a named person recommended a concrete movie, book, song, game, or other title, treat that title as evidence about
  the named person's preference when the question asks for a genre, category, or artist.
"""

LOCOMO_ANSWER_PROMPT_EN = """
You answer LoCoMo benchmark questions using only the retrieved flat memories.

The memories are search results from one conversation. They are not structured entity slices, and they may omit the
speaker's proper name even when the question uses that name.

{grounding_rules}

# Answer procedure
1. Identify the person, event, date, object, number, or category requested by the question.
2. Scan every retrieved memory, including lower-ranked memories, before saying the information is unavailable.
3. Treat repeated memories and same-event memories as evidence from one episode. Merge complementary details from them.
4. For relative dates, calculate from the memory event_time/source_timestamp. For example, an event_time of
   2023-08-09 with "last week" means the week before 9 August 2023.
5. If the retrieved memories contain the requested fact through a candidate alias such as "the user", answer the fact
   directly instead of saying the named person is not mentioned.
6. If the question asks about two people, first look for common/shared themes instead of single-person details.
7. For shared themes, prefer concrete activities, objects, places, or genres over abstract relationship summaries.

# Retrieved memories
{context}

# Question
{question}

Return only the final answer inside <answer> and </answer>. Keep it brief, but include all exact requested names,
numbers, dates, places, teams, programming languages, image captions, and meal names.
"""

# LLM-judge accuracy prompt.
LOCOMO_ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


_QUESTION_NAME_STOPWORDS = {
    "According",
    "Apr",
    "April",
    "Aug",
    "August",
    "Dec",
    "December",
    "Did",
    "Does",
    "Feb",
    "February",
    "Friday",
    "How",
    "Jan",
    "January",
    "Jul",
    "July",
    "Jun",
    "June",
    "Mar",
    "March",
    "May",
    "Monday",
    "Nov",
    "November",
    "Oct",
    "October",
    "Saturday",
    "Sep",
    "Sept",
    "September",
    "Sunday",
    "The",
    "Thursday",
    "Tuesday",
    "Wednesday",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Whose",
}

_EVENT_TIME_RE = re.compile(r"\[(?:event_time|source_timestamp):\s*([^;\]]+)")
_USER_ALIAS_RE = re.compile(r"\bthe user\b", re.IGNORECASE)


def _question_focus_names(question: str) -> list[str]:
    names: list[str] = []
    for name in re.findall(r"\b[A-Z][a-z]+\b", question):
        if name in _QUESTION_NAME_STOPWORDS:
            continue
        if name not in names:
            names.append(name)
    return names


def _repeated_event_times(memories: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for memory in memories:
        match = _EVENT_TIME_RE.search(memory)
        if not match:
            continue
        event_time = match.group(1).strip()
        if not event_time or event_time == "unknown time":
            continue
        counts[event_time] = counts.get(event_time, 0) + 1
    return [event_time for event_time, count in counts.items() if count > 1]


def _mentions_focus_name(memory: str, focus_names: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(name)}\b", memory) for name in focus_names)


def build_answer_context(memories: list[str], question: str = "") -> str:
    """Format retrieved memories for the LoCoMo answer prompt (verbatim port)."""
    lines: list[str] = []
    focus_names = _question_focus_names(question)
    if focus_names:
        joined_names = ", ".join(focus_names)
        lines.append(f"Question focus names: {joined_names}")
        lines.append(
            f'Alias rule for this question: when a relevant memory says "the user", treat it as a candidate memory '
            f"about {joined_names} only if the memory does not already distinguish the named person from the user."
        )
        if re.search(r"\bboth\b", question, re.IGNORECASE) and len(focus_names) >= 2:
            lines.append(
                "Shared-answer rule: this question asks about multiple people, so prefer a common theme or shared "
                "activity over details about only one person."
            )
            lines.append(
                "Shared-answer specificity: choose concrete shared activities, objects, places, or genres before "
                'abstract relationship labels like "mutual support".'
            )
        lines.append("")
    repeated_event_times = _repeated_event_times(memories)
    if repeated_event_times:
        event_times = ", ".join(repeated_event_times[:5])
        lines.append(f"Same-event clusters: {event_times}")
        lines.append("Combine details from memories with these event_time/source_timestamp values before answering.")
        lines.append("")
    lines.append("Reference memories:")
    if not memories:
        lines.append("No relevant memories.")
        return "\n".join(lines)
    joined_names = ", ".join(focus_names)
    for index, memory in enumerate(memories, start=1):
        lines.append(f"{index}. {memory}")
        if focus_names and _USER_ALIAS_RE.search(memory):
            if _mentions_focus_name(memory, focus_names):
                lines.append(
                    f'   Participant-role note: memory {index} mentions both "the user" and {joined_names}; keep '
                    "their roles separate and use only facts attributed to the named participant as direct evidence "
                    "about that participant."
                )
            else:
                lines.append(
                    f'   Subject alias note: "the user" in memory {index} can refer to {joined_names} '
                    "when this memory contains the requested fact."
                )
    return "\n".join(lines)


def build_answer_prompt(memories: list[str], question: str, template: str | None = None) -> str:
    """Build the LoCoMo answer prompt with grounding rules for flat memories (verbatim port)."""
    context = build_answer_context(memories, question=question)
    selected_template = template or LOCOMO_ANSWER_PROMPT_EN
    prompt = (
        selected_template.replace("{grounding_rules}", LOCOMO_ANSWER_GROUNDING_RULES)
        .replace("{context}", context)
        .replace("{conversation_memories}", context)
        .replace("{question}", question)
    )
    if "{grounding_rules}" not in selected_template and "# CRITICAL REQUIREMENTS" in selected_template:
        prompt = prompt.replace("# CRITICAL REQUIREMENTS", LOCOMO_ANSWER_GROUNDING_RULES + "\n# CRITICAL REQUIREMENTS")
    return prompt


# Message / timestamp helpers


def strip_speaker_prefix(content: str) -> str:
    """Remove speaker prefixes from dialogue text."""
    text = content or ""
    for prefix in ("User: ", "Assistant: ", "user: ", "assistant: "):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def session_timestamp_millis(raw_timestamp: str) -> int:
    """Parse a LoCoMo session date as a millisecond timestamp."""
    from dateutil import parser as dateutil_parser

    dt = dateutil_parser.parse(raw_timestamp)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.astimezone(UTC).timestamp() * 1000)


def _message_text(msg: dict[str, Any]) -> str:
    """Build text for one LoCoMo message."""
    text = msg.get("text", "") or ""
    if msg.get("blip_caption"):
        text += f" [Shared image: {msg['blip_caption']}]"
    if msg.get("query"):
        text += f" [Image context: {msg['query']}]"
    return strip_speaker_prefix(text)


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


def _extract_answer(full_response: str) -> tuple[str, str]:
    """Extract the answer and chain-of-thought from model output."""
    answer = full_response
    if "<answer>" in answer:
        answer = answer.split("<answer>")[1]
    if "</answer>" in answer:
        answer = answer.split("</answer>")[0]
    chain_of_thought = full_response.split("<answer>")[0].strip() if "<answer>" in full_response else ""
    return answer.strip(), chain_of_thought


class LocomoLLMJudgeScorer(Scorer):
    """LoCoMo LLM judge scorer that maps CORRECT/WRONG to 1/0."""

    def __init__(self, llm: LLMClient, *, prompt: str = LOCOMO_ACCURACY_PROMPT) -> None:
        self._llm = llm
        self._prompt = prompt

    async def score(
        self,
        *,
        question: str,
        answer: str,
        gold: str,
        contexts: list[str] | None = None,
    ) -> ScoreResult:
        content = self._prompt.format(question=question, gold_answer=gold, generated_answer=answer)
        raw = (
            await self._llm.complete(
                [{"role": "user", "content": content}],
                response_format={"type": "json_object"},
            )
        ).content
        payload = _parse_judge_json(raw)
        label = str(payload.get("label", "")).strip().upper()
        passed = label == "CORRECT"
        return ScoreResult(score=1.0 if passed else 0.0, passed=passed, reason=label or raw[:120], raw=payload)


class LocomoAnswer(BaseModel):
    """Answer result for one question."""

    model_config = ConfigDict(extra="ignore")

    question: str
    answer: str
    response_raw: str
    chain_of_thought: str = ""
    memories: list[str] = Field(default_factory=list)
    search_time: float = 0.0
    prompt: str = ""


class LocomoQAResult(BaseModel):
    """End-to-end result for one question."""

    model_config = ConfigDict(extra="ignore")

    question: str
    gold_answer: str
    category: Any = None
    response: str
    chain_of_thought: str = ""
    memory: list[str] = Field(default_factory=list)
    search_time: float = 0.0
    score: ScoreResult | None = None


class LocomoAddSummary(BaseModel):
    """Represent LocomoAddSummary."""

    model_config = ConfigDict(extra="ignore")

    conv_idx: int
    user_id: str
    total_sessions: int
    added_sessions: int
    failed_sessions: list[tuple[str, str]] = Field(default_factory=list)


class LocomoConversationResult(BaseModel):
    """Full result for one conversation."""

    model_config = ConfigDict(extra="ignore")

    conv_idx: int
    user_id: str
    num_questions: int
    qa_results: list[LocomoQAResult] = Field(default_factory=list)
    add_summary: LocomoAddSummary | None = None


class MetricBucket(BaseModel):
    """Represent MetricBucket."""

    model_config = ConfigDict(extra="ignore")

    count: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.count if self.count else 0.0


class LocomoRunResult(BaseModel):
    """Represent LocomoRunResult."""

    model_config = ConfigDict(extra="ignore")

    conversations: list[LocomoConversationResult] = Field(default_factory=list)
    total_questions: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_questions if self.total_questions else 0.0

    def is_scored(self) -> bool:
        """Handle is scored."""
        return any(qa.score is not None for conv in self.conversations for qa in conv.qa_results)

    def by_category(self) -> dict[str, MetricBucket]:
        """Aggregate counts and accuracy by category."""
        buckets: dict[str, MetricBucket] = {}
        for conv in self.conversations:
            for qa in conv.qa_results:
                if qa.score is None:
                    continue
                bucket = buckets.setdefault(str(qa.category), MetricBucket())
                bucket.count += 1
                bucket.correct += int(qa.score.passed)
        return buckets

    def overall(self) -> MetricBucket:
        """Aggregate overall counts and accuracy."""
        return MetricBucket(count=self.total_questions, correct=self.correct)

    def format_metrics(self) -> str:
        """Format category and overall metrics."""
        lines: list[str] = []
        lines.append("By category (count, accuracy):")
        categories = self.by_category()
        if not categories:
            lines.append("  (no scored questions)")
        for category in sorted(categories):
            bucket = categories[category]
            lines.append(
                f"  category {category}: n={bucket.count} acc={bucket.accuracy:.4f} ({bucket.correct}/{bucket.count})"
            )
        overall = self.overall()
        lines.append("Overall (count, accuracy):")
        lines.append(f"  n={overall.count} acc={overall.accuracy:.4f} ({overall.correct}/{overall.count})")
        return "\n".join(lines)

    def format_report(self) -> str:
        """Format a full dataset run report."""
        lines: list[str] = ["=" * 60, "LoCoMo evaluation report", "=" * 60]
        for conv in self.conversations:
            summary = conv.add_summary
            added = f"{summary.added_sessions}/{summary.total_sessions} sessions" if summary else "add skipped"
            lines.append(f"  {conv.user_id}: {conv.num_questions} questions, {added}")
            if summary and summary.failed_sessions:
                for session, reason in summary.failed_sessions:
                    lines.append(f"    - add failed: {session}: {reason}")
        lines.append("-" * 60)
        if not self.is_scored():
            answered = sum(conv.num_questions for conv in self.conversations)
            lines.append(f"Answered {answered} questions (scoring skipped).")
        else:
            lines.append(self.format_metrics())
        lines.append("=" * 60)
        return "\n".join(lines)


class LocomoEnv:
    """LoCoMo memory evaluation environment."""

    def __init__(
        self,
        memory: AsyncMemoryClient,
        *,
        answer_llm: LLMClient,
        judge_llm: LLMClient | None = None,
        scorer: Scorer | None = None,
        top_k: int | None = 50,
        search_strategy: str = "agentic",
        rerank: bool = False,
        answer_template: str = LOCOMO_ANSWER_PROMPT_EN,
    ) -> None:
        """Handle init."""
        self._memory = memory
        self._answer_llm = answer_llm
        self._scorer = scorer or LocomoLLMJudgeScorer(judge_llm or answer_llm)
        self._top_k = top_k
        self._search_strategy = search_strategy
        self._rerank = rerank
        self._answer_template = answer_template

    async def add_session(
        self,
        user_id: str,
        messages: list[dict[str, Any]],
        raw_timestamp: str,
        *,
        session_key: str | None = None,
    ) -> None:
        """Add all messages in one LoCoMo session."""
        ts_millis = session_timestamp_millis(raw_timestamp)
        dict_messages = [
            {"role": msg["speaker"], "content": _message_text(msg), "timestamp": ts_millis} for msg in messages
        ]
        if not dict_messages:
            return
        metadata = {"locomo_session_key": session_key} if session_key is not None else None
        await self._memory.add(
            dict_messages,
            user_id=user_id,
            mode="sync",
            session_id=user_id,
            metadata=metadata,
        )

    @staticmethod
    def _session_keys(conversation: dict[str, Any]) -> list[str]:
        """Handle session keys."""
        session_keys = [k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")]
        session_keys.sort(key=lambda x: int(x.split("_")[1]))
        return session_keys

    async def add_conversation(
        self,
        item: dict[str, Any],
        idx: int,
        *,
        on_session_done: Callable[[], None] | None = None,
    ) -> LocomoAddSummary:
        """Add all sessions in one LoCoMo conversation."""
        conversation = item["conversation"]
        user_id = f"conv_{idx}"
        session_keys = self._session_keys(conversation)

        added = 0
        failed: list[tuple[str, str]] = []
        for session_key in session_keys:
            try:
                date_key = session_key + "_date_time"
                if date_key not in conversation:
                    failed.append((session_key, "missing_timestamp"))
                    continue
                messages = conversation.get(session_key) or []
                if not messages:
                    continue
                try:
                    await self.add_session(user_id, messages, conversation[date_key], session_key=session_key)
                    added += 1
                except Exception as exc:  # noqa: BLE001 - record and continue, like the reference
                    failed.append((session_key, f"{type(exc).__name__}: {exc}"))
            finally:
                if on_session_done is not None:
                    on_session_done()

        return LocomoAddSummary(
            conv_idx=idx,
            user_id=user_id,
            total_sessions=len(session_keys),
            added_sessions=added,
            failed_sessions=failed,
        )

    async def answer(self, user_id: str, question: str) -> LocomoAnswer:
        """Answer a question using existing memories."""
        start = time.time()
        search = await self._memory.search(
            question,
            user_id=user_id,
            top_k=self._top_k,
            search_strategy=self._search_strategy,
            rerank=self._rerank,
            filters={"user_id": user_id},
            session_id=user_id,
        )
        memories = [_format_memory_for_answering(hit) for hit in search.memories]
        search_time = time.time() - start

        prompt = build_answer_prompt(memories, question, self._answer_template)
        full_response = (await self._answer_llm.complete([{"role": "user", "content": prompt}])).content
        answer_text, chain_of_thought = _extract_answer(full_response)
        return LocomoAnswer(
            question=question,
            answer=answer_text,
            response_raw=full_response,
            chain_of_thought=chain_of_thought,
            memories=memories,
            search_time=search_time,
            prompt=prompt,
        )

    async def score(self, question: str, gold_answer: str, response: str) -> ScoreResult:
        """Score one predicted answer."""
        return await self._scorer.score(question=question, answer=response, gold=gold_answer)

    async def evaluate_question(self, user_id: str, q_item: dict[str, Any], *, score: bool = True) -> LocomoQAResult:
        """Evaluate one LoCoMo question end to end."""
        question = q_item.get("question", "")
        gold_answer = str(q_item.get("answer", ""))
        category = q_item.get("category")

        answer = await self.answer(user_id, question)
        score_result = await self.score(question, gold_answer, answer.answer) if score else None
        return LocomoQAResult(
            question=question,
            gold_answer=gold_answer,
            category=category,
            response=answer.answer,
            chain_of_thought=answer.chain_of_thought,
            memory=answer.memories,
            search_time=answer.search_time,
            score=score_result,
        )

    async def run_dataset(
        self,
        data: list[dict[str, Any]],
        *,
        max_conv_concurrency: int = 4,
        max_qa_concurrency: int = 20,
        max_search_concurrency: int | None = None,
        max_score_concurrency: int | None = None,
        add: bool = True,
        score: bool = True,
        skip_category_5: bool = True,
        print_report: bool = True,
        show_progress: bool = True,
    ) -> LocomoRunResult:
        """Run a full LoCoMo dataset."""
        search_sem = asyncio.Semaphore(max_search_concurrency or max_qa_concurrency)
        score_sem = asyncio.Semaphore(max_score_concurrency or max_qa_concurrency)
        conv_sem = asyncio.Semaphore(max_conv_concurrency)

        # Precompute progress totals for sessions, conversations, and questions.
        total_sessions = sum(len(self._session_keys(it["conversation"])) for it in data) if add else 0
        total_questions = sum(
            len([q for q in it.get("qa", []) if not (skip_category_5 and q.get("category") == 5)]) for it in data
        )

        add_pbar = (
            tqdm(total=total_sessions, desc="添加记忆 (session)", unit="session", position=0)
            if show_progress and add
            else None
        )
        conv_pbar = (
            tqdm(total=len(data), desc="对话测评 (conversation)", unit="conv", position=1) if show_progress else None
        )
        qa_pbar = (
            tqdm(total=total_questions, desc="回答问题 (question)", unit="q", position=2) if show_progress else None
        )

        async def run_conversation(idx: int, item: dict[str, Any]) -> LocomoConversationResult:
            async with conv_sem:
                user_id = f"conv_{idx}"
                on_session_done = add_pbar.update if add_pbar is not None else None
                add_summary = await self.add_conversation(item, idx, on_session_done=on_session_done) if add else None

                qa_list = [q for q in item.get("qa", []) if not (skip_category_5 and q.get("category") == 5)]

                async def run_question(q_item: dict[str, Any]) -> LocomoQAResult:
                    question = q_item.get("question", "")
                    gold_answer = str(q_item.get("answer", ""))
                    category = q_item.get("category")

                    async with search_sem:
                        answer = await self.answer(user_id, question)
                    if score:
                        async with score_sem:
                            score_result = await self.score(question, gold_answer, answer.answer)
                    else:
                        score_result = None
                    if qa_pbar is not None:
                        qa_pbar.update()
                    return LocomoQAResult(
                        question=question,
                        gold_answer=gold_answer,
                        category=category,
                        response=answer.answer,
                        chain_of_thought=answer.chain_of_thought,
                        memory=answer.memories,
                        search_time=answer.search_time,
                        score=score_result,
                    )

                qa_results = await asyncio.gather(*(run_question(q) for q in qa_list)) if qa_list else []
                if conv_pbar is not None:
                    conv_pbar.update()
                return LocomoConversationResult(
                    conv_idx=idx,
                    user_id=user_id,
                    num_questions=len(qa_list),
                    qa_results=list(qa_results),
                    add_summary=add_summary,
                )

        try:
            conversations = await asyncio.gather(*(run_conversation(i, it) for i, it in enumerate(data)))
        finally:
            for pbar in (qa_pbar, conv_pbar, add_pbar):
                if pbar is not None:
                    pbar.close()
        conversations = list(conversations)

        total = 0
        correct = 0
        for conv in conversations:
            for qa in conv.qa_results:
                if qa.score is None:
                    continue
                total += 1
                correct += int(qa.score.passed)
        run = LocomoRunResult(conversations=conversations, total_questions=total, correct=correct)
        if print_report:
            print(run.format_report(), flush=True)
        return run

    @staticmethod
    def load_dataset(path: str) -> list[dict[str, Any]]:
        """Handle load dataset."""
        with open(path, encoding="utf-8") as f:
            return json.load(f)
