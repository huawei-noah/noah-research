"""MemoryAgentBench evaluation environment.

This module adapts the official MemoryAgentBench workflow to the MindMemOS SDK:

- split each benchmark context into chunks;
- add every chunk to memory as one synchronous user/assistant dialogue;
- answer each formatted benchmark query from retrieved memories;
- score outputs with the same text metrics used by the reference runner.

The implementation intentionally keeps the heavy official agent stack out of the
SDK and ports only the dataset shaping, prompt templates, and metrics needed to
evaluate a MindMemOS-backed memory agent.
"""

from __future__ import annotations

import asyncio
import json
import re
import string
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mindmemos_sdk.memory import AsyncMemoryClient, MemorySearchHit
from pydantic import BaseModel, ConfigDict, Field
from rouge_score import rouge_scorer
from tqdm.auto import tqdm

from mindmemos_eval.llm import LLMClient
from mindmemos_eval.memory.scorer import ScoreResult

MEMORYAGENTBENCH_HF_DATASET = "ai-hyz/MemoryAgentBench"
MEMORYAGENTBENCH_SYSTEM_PROMPT = (
    "You are a helpful assistant that can read the context and memorize it for future retrieval."
)

_BASE_TEMPLATES: dict[str, dict[str, Any]] = {
    "ruler_qa": {
        "memorize": "Dialogue between User and Assistant {time_stamp}\n<User> The following context is the documents I have read: \n{context}\n <Assistant> I have learned the documents and I will answer the question you ask.",
        "query": "Search Archival Memory and answer my question. Only give me the answer and do not output any other words. \n\nQuestion: {question} \n\n Answer:",
    },
    "longmemeval": {
        "memorize": "Dialogue between User and Assistant \n<User> The following context is the conversation between the user and the assistant: \n{context}\n <Assistant> I have memorized the conversation and I will answer the question you ask.",
        "query": "Search Archival Memory and answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
    },
    "eventqa": {
        "memorize": "Dialogue between User and Assistant {time_stamp}\n<User> The following context is the book excerpt: \n{context}\n <Assistant> I have read the book excerpt and I will answer the question you ask.",
        "query": "Search Archival Memory, complete the task below:\n\n{question}\n\n The event that happens next is:",
    },
    "in_context_learning": {
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the examples I have learned: \n{context}\n <Assistant> I have learned the examples and I will answer the question you ask.",
        "query": 'Search Archival Memory and use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output "label: {label}" and nothing else. \n\n{question} \n\n label:',
    },
    "recsys_redial": {
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the dialogues between a user and recommender system: \n{context}\n <Assistant> I have memorized the dialogues and I will answer the question you ask.",
        "query": "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Search Archival Memory, you reply me with 20 recommendations without extra sentences. \n\nFor Example:\n\n[Conversation]\n\nThe recommendations are: \n1.movie1\n2.movie2\n...\n\n Here is the conversation: {question} \n\n The recommendations are: \n",
    },
    "infbench_sum": {
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the book I have read: \n{context}\n <Assistant> I have read the book and I will answer the question you ask.",
        "query": "You are given a book above and you are tasked to summarize it. \n\n{question} \n\n Now summarize the book.",
    },
    "detective_qa": {
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the book I have read: \n{context}\n <Assistant> I have read the book and I will answer the question you ask.",
        "query": "Search Archival Memory and answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
    },
    "factconsolidation": {
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the facts I have learned: \n{context}\n <Assistant> I have learned the facts and I will answer the question you ask.",
        "query": "Pretend you are a knowledge management system. Each fact in the  Archival Memory is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the Archival Memory by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Archival Memory] \n\n Question: Based on the Archival Memory, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the  Archival Memory, {question} \nAnswer:",
    },
}

_DATASET_MAPPING: tuple[tuple[tuple[str, ...], str], ...] = (
    (("ruler_", "qa"), "ruler_qa"),
    (("icl_",), "in_context_learning"),
    (("infbench_", "sum"), "infbench_sum"),
    (("eventqa_",), "eventqa"),
    (("recsys_", "redial"), "recsys_redial"),
    (("longmemeval_",), "longmemeval"),
    (("factconsolidation_",), "factconsolidation"),
    (("detective_", "qa"), "detective_qa"),
)

_SUBSTRING_METRIC_DATASETS = ("eventqa", "ruler_", "factconsolidation")
_EXACT_METRIC_DATASETS = ("detective_", "icl_")
_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL", "rougeLsum"], use_stemmer=True)


def normalize_memoryagentbench_dataset_name(sub_dataset: str) -> str:
    """Normalize a MemoryAgentBench sub-dataset name to a template family."""
    normalized = sub_dataset.strip().lower()
    for patterns, name in _DATASET_MAPPING:
        if all(pattern in normalized for pattern in patterns):
            return name
    raise ValueError(f"Unknown MemoryAgentBench sub_dataset: {sub_dataset!r}")


def get_memoryagentbench_template(sub_dataset: str, template_name: str) -> str:
    """Return the official MemoryAgentBench template for MindMemOS agentic memory."""
    dataset_name = normalize_memoryagentbench_dataset_name(sub_dataset)
    return str(_BASE_TEMPLATES[dataset_name][template_name])


def normalize_answer(text: str) -> str:
    """Normalize text for the official DRQA-style metrics."""
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def parse_output(output_text: str, answer_prefix: str = "Answer:") -> str:
    """Extract the first answer line, matching the reference runner's parser."""
    patterns = [
        re.compile(f"(?:{re.escape(answer_prefix)})(.*)(?:\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ]
    for pattern in patterns:
        match = pattern.search(output_text or "")
        if match:
            extracted = match[1].strip()
            return re.sub(f"^{re.escape(answer_prefix)}", "", extracted, flags=re.IGNORECASE).strip()
    return ""


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _flat_ground_truths(ground_truths: Any) -> list[str]:
    values = _ensure_list(ground_truths)
    flat: list[str] = []
    for value in values:
        if isinstance(value, list):
            flat.extend(str(item) for item in value)
        else:
            flat.append(str(value))
    return flat


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Official normalized exact match."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def substring_exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Official normalized substring exact match."""
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 used by the official helper."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _max_over_ground_truths(metric: Callable[[str, str], float | bool], prediction: str, ground_truths: Any) -> float:
    answers = _flat_ground_truths(ground_truths)
    if not answers:
        return 0.0
    return max(float(metric(prediction, answer)) for answer in answers)


def _calculate_reference_metrics(prediction: str, answers: Any) -> dict[str, float]:
    """Port of official ``calculate_metrics`` for one prediction string."""
    answer_list = _flat_ground_truths(answers)
    metrics = {
        "exact_match": _max_over_ground_truths(exact_match_score, prediction, answer_list),
        "f1": _max_over_ground_truths(f1_score, prediction, answer_list),
        "substring_exact_match": _max_over_ground_truths(substring_exact_match_score, prediction, answer_list),
    }
    if answer_list:
        rouge_scores = [_ROUGE_SCORER.score(target=answer, prediction=prediction) for answer in answer_list]
        for rouge_type in _ROUGE_SCORER.rouge_types:
            metrics[f"{rouge_type}_f1"] = max(score[rouge_type].fmeasure for score in rouge_scores)
            metrics[f"{rouge_type}_recall"] = max(score[rouge_type].recall for score in rouge_scores)
    return metrics


def calculate_memoryagentbench_metrics(prediction: str, answers: Any, sub_dataset: str) -> dict[str, float]:
    """Calculate MemoryAgentBench metrics using the official post-process branches."""
    lowered = sub_dataset.strip().lower()
    if "eventqa" in lowered:
        answer_list = _flat_ground_truths(answers)
        parsed = parse_output(prediction)
        metrics = _calculate_reference_metrics(parsed, answers)
        if answer_list:
            recall = sum(answer.lower() in prediction.lower() for answer in answer_list) / len(answer_list)
            metrics["eventqa_recall"] = float(recall == 1.0)
        return metrics
    if "icl" in lowered:
        parsed = parse_output(prediction)
        return _calculate_reference_metrics(parsed, answers)

    metrics = _calculate_reference_metrics(prediction, answers)
    parsed = parse_output(prediction)
    if parsed is not None:
        parsed_metrics = _calculate_reference_metrics(parsed, answers)
        metrics = {name: max(value, parsed_metrics[name]) for name, value in metrics.items()}
    return metrics


def primary_metric_for_sub_dataset(sub_dataset: str) -> str:
    """Return the README-documented primary metric field for a sub-dataset."""
    lowered = sub_dataset.strip().lower()
    if any(pattern in lowered for pattern in _SUBSTRING_METRIC_DATASETS):
        return "substring_exact_match"
    if any(pattern in lowered for pattern in _EXACT_METRIC_DATASETS):
        return "exact_match"
    if "recsys" in lowered:
        return "recsys_recall@5"
    if "longmemeval" in lowered or "infbench" in lowered:
        return "f1"
    return "substring_exact_match"


def chunk_text_into_sentences(text: str, model_name: str = "gpt-4o-mini", chunk_size: int = 4096) -> list[str]:
    """Port of official NLTK sentence chunker with tiktoken token limits."""
    import nltk
    import tiktoken

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    sentences = nltk.sent_tokenize(text)
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence, allowed_special={"<|endoftext|>"}))
        if current_tokens + sentence_tokens > chunk_size:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_tokens = sentence_tokens
        else:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


def _message_token_count(text: str) -> int:
    return len((text or "").split())


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _coerce_dataset_item(raw: dict[str, Any], *, default_source: str = "") -> "MemoryAgentBenchItem":
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    questions = _ensure_list(_first_present(raw, "questions", "question"))
    answers = _ensure_list(_first_present(raw, "answers", "answer"))
    source = str(raw.get("source") or metadata.get("source") or default_source)
    qa_pair_ids = _ensure_list(raw.get("qa_pair_ids") or metadata.get("qa_pair_ids"))
    question_ids = _ensure_list(raw.get("question_ids") or metadata.get("question_ids"))
    question_dates = _ensure_list(raw.get("question_dates") or metadata.get("question_dates"))
    question_types = _ensure_list(raw.get("question_types") or metadata.get("question_types"))
    previous_events = _ensure_list(raw.get("previous_events") or metadata.get("previous_events"))
    context = str(raw.get("context", ""))
    if not context:
        raise ValueError("MemoryAgentBench item is missing a non-empty `context` field.")
    return MemoryAgentBenchItem(
        context=context,
        questions=[str(q) for q in questions],
        answers=answers,
        source=source,
        qa_pair_ids=[str(x) for x in qa_pair_ids],
        question_ids=[str(x) for x in question_ids],
        question_dates=[str(x) for x in question_dates],
        question_types=[str(x) for x in question_types],
        previous_events=[str(x) for x in previous_events],
        raw={k: v for k, v in raw.items() if k != "context"},
    )


class MemoryAgentBenchQuestion(BaseModel):
    """One formatted MemoryAgentBench query and its gold answer."""

    model_config = ConfigDict(extra="ignore")

    question: str
    query: str
    answer: Any
    source: str
    primary_metric: str
    qa_pair_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryAgentBenchItem(BaseModel):
    """One MemoryAgentBench context with one or more QA pairs."""

    model_config = ConfigDict(extra="ignore")

    context: str
    questions: list[str]
    answers: list[Any]
    source: str = ""
    qa_pair_ids: list[str] = Field(default_factory=list)
    question_ids: list[str] = Field(default_factory=list)
    question_dates: list[str] = Field(default_factory=list)
    question_types: list[str] = Field(default_factory=list)
    previous_events: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    def build_questions(self, sub_dataset: str) -> list[MemoryAgentBenchQuestion]:
        """Format all QA pairs with the official query template."""
        query_template = get_memoryagentbench_template(sub_dataset, "query")
        if len(self.questions) > 1 and len(self.answers) > 1:
            pairs = list(zip(self.questions, self.answers, strict=False))
        else:
            pairs = [
                (
                    self.questions[0] if self.questions else "",
                    self.answers,
                )
            ]

        result: list[MemoryAgentBenchQuestion] = []
        for idx, pair in enumerate(pairs):
            question = str(pair[0])
            answer = pair[1]
            metadata = {
                **self.raw,
                "question": question,
                "answer": answer,
                "source": self.source,
                "question_ids": self.question_ids[idx] if idx < len(self.question_ids) else self.question_ids,
                "question_dates": self.question_dates[idx] if idx < len(self.question_dates) else self.question_dates,
                "question_types": self.question_types[idx] if idx < len(self.question_types) else self.question_types,
                "previous_events": self.previous_events[idx]
                if idx < len(self.previous_events)
                else self.previous_events,
                "qa_pair_ids": self.qa_pair_ids[idx] if idx < len(self.qa_pair_ids) else None,
            }
            query = query_template.format(**metadata)
            result.append(
                MemoryAgentBenchQuestion(
                    question=question,
                    query=query,
                    answer=answer,
                    source=self.source,
                    primary_metric=primary_metric_for_sub_dataset(self.source),
                    qa_pair_id=metadata.get("qa_pair_ids"),
                    metadata=metadata,
                )
            )
        return result


class MemoryAgentBenchAnswer(BaseModel):
    """Answer result for one MemoryAgentBench query."""

    model_config = ConfigDict(extra="ignore")

    question: str
    query: str
    answer: str
    memories: list[str] = Field(default_factory=list)
    prompt: list[dict[str, Any]] = Field(default_factory=list)
    input_len: int = 0
    output_len: int = 0
    search_time: float = 0.0
    query_time_len: float = 0.0


class MemoryAgentBenchQAResult(BaseModel):
    """One MemoryAgentBench QA result with metric fields."""

    model_config = ConfigDict(extra="ignore")

    query_id: int
    context_id: int
    qa_pair_id: str | None = None
    query: str
    question: str
    answer: Any
    source: str
    primary_metric: str
    output: str
    parsed_output: str = ""
    memory: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    score: ScoreResult | None = None
    input_len: int = 0
    output_len: int = 0
    memory_construction_time: float = 0.0
    query_time_len: float = 0.0


class MemoryAgentBenchAddSummary(BaseModel):
    """Add-stage summary for one context."""

    model_config = ConfigDict(extra="ignore")

    context_id: int
    user_id: str
    total_chunks: int
    added_chunks: int
    failed_chunks: list[tuple[int, str]] = Field(default_factory=list)
    memory_construction_time: float = 0.0


class MemoryAgentBenchContextResult(BaseModel):
    """Full result for one MemoryAgentBench context."""

    model_config = ConfigDict(extra="ignore")

    context_id: int
    user_id: str
    num_questions: int
    qa_results: list[MemoryAgentBenchQAResult] = Field(default_factory=list)
    add_summary: MemoryAgentBenchAddSummary | None = None


class MemoryAgentBenchRunResult(BaseModel):
    """Whole-run MemoryAgentBench results and aggregate metrics."""

    model_config = ConfigDict(extra="ignore")

    sub_dataset: str = ""
    primary_metric: str = ""
    contexts: list[MemoryAgentBenchContextResult] = Field(default_factory=list)
    metrics: dict[str, list[float]] = Field(default_factory=dict)
    averaged_metrics: dict[str, float] = Field(default_factory=dict)

    def format_report(self) -> str:
        """Format an official-style metric report."""
        lines = ["=" * 60, "MemoryAgentBench evaluation report", "=" * 60]
        for ctx in self.contexts:
            summary = ctx.add_summary
            added = f"{summary.added_chunks}/{summary.total_chunks} chunks" if summary else "add skipped"
            lines.append(f"  {ctx.user_id}: {ctx.num_questions} questions, {added}")
            if summary and summary.failed_chunks:
                for chunk_idx, reason in summary.failed_chunks:
                    lines.append(f"    - add failed: chunk {chunk_idx}: {reason}")
        lines.append("-" * 60)
        if not self.averaged_metrics:
            lines.append("No scored questions.")
        else:
            for name in sorted(self.averaged_metrics):
                marker = " (primary)" if self.primary_metric and name == self.primary_metric else ""
                value = self.averaged_metrics[name]
                lines.append(f"  {name}: {value:.4f}{marker}")
        lines.append("=" * 60)
        return "\n".join(lines)


class MemoryAgentBenchEnv:
    """MemoryAgentBench environment backed by AsyncMemoryClient and LLMClient."""

    def __init__(
        self,
        memory: AsyncMemoryClient,
        *,
        answer_llm: LLMClient,
        sub_dataset: str = "",
        top_k: int = 50,
        search_strategy: str = "agentic",
        chunk_size: int = 4096,
        answer_system_prompt: str = MEMORYAGENTBENCH_SYSTEM_PROMPT,
    ) -> None:
        self._memory = memory
        self._answer_llm = answer_llm
        self._sub_dataset = sub_dataset.strip()
        self._top_k = top_k
        self._search_strategy = search_strategy
        self._chunk_size = chunk_size
        self._answer_system_prompt = answer_system_prompt
        self.primary_metric = primary_metric_for_sub_dataset(self._sub_dataset) if self._sub_dataset else ""

    def chunk_context(self, context: str) -> list[str]:
        """Split context into chunks for incremental memory injection."""
        return chunk_text_into_sentences(context, chunk_size=self._chunk_size)

    def format_memorize_message(self, chunk: str, source: str) -> str:
        """Format one context chunk with the official memorize template."""
        memorize_template = get_memoryagentbench_template(source, "memorize")
        kwargs = {"context": chunk}
        if "{time_stamp}" in memorize_template:
            kwargs["time_stamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return memorize_template.format(**kwargs)

    def context_user_id(self, context_id: int, source: str) -> str:
        """Return the isolated user/session id for a context."""
        return f"context_{context_id}_{source}"

    async def add_context(
        self,
        item: MemoryAgentBenchItem,
        context_id: int,
        *,
        on_chunk_done: Callable[[], None] | None = None,
    ) -> MemoryAgentBenchAddSummary:
        """Add every context chunk to memory in order."""
        user_id = self.context_user_id(context_id, item.source)
        chunks = self.chunk_context(item.context)
        added = 0
        failed: list[tuple[int, str]] = []
        start = time.time()
        for chunk_idx, chunk in enumerate(chunks):
            try:
                content = self.format_memorize_message(chunk, item.source)
                timestamp = int(time.time() * 1000)
                # Match the official mem0-style agentic memory path: system, user
                # memorize prompt, then a fixed assistant acknowledgement.
                messages = [
                    {"role": "system", "content": self._answer_system_prompt, "timestamp": timestamp},
                    {"role": "user", "content": content, "timestamp": timestamp},
                    {
                        "role": "assistant",
                        "content": "I'll make sure to add the content into the memory.",
                        "timestamp": timestamp,
                    },
                ]
                await self._memory.add(messages, user_id=user_id, mode="sync", session_id=user_id)
                added += 1
            except Exception as exc:  # noqa: BLE001 - keep benchmark running and report failed chunks
                failed.append((chunk_idx, f"{type(exc).__name__}: {exc}"))
            finally:
                if on_chunk_done is not None:
                    on_chunk_done()
        return MemoryAgentBenchAddSummary(
            context_id=context_id,
            user_id=user_id,
            total_chunks=len(chunks),
            added_chunks=added,
            failed_chunks=failed,
            memory_construction_time=time.time() - start,
        )

    @staticmethod
    def _format_hit(hit: MemorySearchHit) -> str:
        if hit.event_time or hit.source_timestamp:
            return (
                f"[event_time: {hit.event_time or 'unknown time'}; "
                f"source_timestamp: {hit.source_timestamp or 'unknown time'}] {hit.memory}"
            )
        return hit.memory

    def build_answer_messages(self, query: str, memories: list[str]) -> list[dict[str, Any]]:
        """Build the answer prompt from retrieved memory text and formatted query."""
        memories_str = "\n".join(f"- {memory}" for memory in memories)
        system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\n{memories_str}\n"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query + "\n\nCurrent Time: " + time.strftime("%Y-%m-%d %H:%M:%S")},
        ]

    async def answer(self, user_id: str, question: MemoryAgentBenchQuestion) -> MemoryAgentBenchAnswer:
        """Search memories and answer one formatted MemoryAgentBench query."""
        start_search = time.time()
        search = await self._memory.search(
            question.query,
            user_id=user_id,
            top_k=self._top_k,
            search_strategy=self._search_strategy,
            session_id=user_id,
        )
        memories = [self._format_hit(hit) for hit in search.memories]
        search_time = time.time() - start_search
        messages = self.build_answer_messages(question.query, memories)
        start_query = time.time()
        answer = (await self._answer_llm.complete(messages)).content
        query_time = time.time() - start_query
        return MemoryAgentBenchAnswer(
            question=question.question,
            query=question.query,
            answer=answer,
            memories=memories,
            prompt=messages,
            input_len=sum(_message_token_count(message.get("content", "")) for message in messages),
            output_len=_message_token_count(answer),
            search_time=search_time,
            query_time_len=query_time,
        )

    async def evaluate_question(
        self,
        user_id: str,
        question: MemoryAgentBenchQuestion,
        *,
        query_id: int,
        context_id: int,
        score: bool = True,
        memory_construction_time: float = 0.0,
    ) -> MemoryAgentBenchQAResult:
        """Answer and score one MemoryAgentBench question."""
        answer = await self.answer(user_id, question)
        metrics = calculate_memoryagentbench_metrics(answer.answer, question.answer, question.source) if score else {}
        score_result = None
        if score and question.primary_metric in metrics:
            value = metrics[question.primary_metric]
            score_result = ScoreResult(
                score=value,
                passed=value >= 1.0 if question.primary_metric != "f1" else value > 0.0,
                reason=question.primary_metric,
                raw=metrics,
            )
        return MemoryAgentBenchQAResult(
            query_id=query_id,
            context_id=context_id,
            qa_pair_id=question.qa_pair_id,
            query=question.query,
            question=question.question,
            answer=question.answer,
            source=question.source,
            primary_metric=question.primary_metric,
            output=answer.answer,
            parsed_output=parse_output(answer.answer),
            memory=answer.memories,
            metrics=metrics,
            score=score_result,
            input_len=answer.input_len,
            output_len=answer.output_len,
            memory_construction_time=memory_construction_time,
            query_time_len=answer.query_time_len,
        )

    async def run_dataset(
        self,
        data: list[MemoryAgentBenchItem],
        *,
        max_context_concurrency: int = 2,
        max_qa_concurrency: int = 10,
        add: bool = True,
        score: bool = True,
        max_queries: int = 0,
        print_report: bool = True,
        show_progress: bool = True,
    ) -> MemoryAgentBenchRunResult:
        """Run the benchmark over a list of MemoryAgentBench items."""
        ctx_sem = asyncio.Semaphore(max_context_concurrency)
        qa_sem = asyncio.Semaphore(max_qa_concurrency)
        total_chunks = sum(len(self.chunk_context(item.context)) for item in data) if add else 0
        all_questions = [item.build_questions(item.source) for item in data]
        total_questions = sum(len(questions) for questions in all_questions)
        if max_queries > 0:
            total_questions = min(total_questions, max_queries)

        add_pbar = (
            tqdm(total=total_chunks, desc="添加记忆 (chunk)", unit="chunk", position=0)
            if show_progress and add
            else None
        )
        ctx_pbar = tqdm(total=len(data), desc="上下文测评 (context)", unit="ctx", position=1) if show_progress else None
        qa_pbar = (
            tqdm(total=total_questions, desc="回答问题 (question)", unit="q", position=2) if show_progress else None
        )

        query_counter = 0
        query_counter_lock = asyncio.Lock()

        async def next_query_id() -> int | None:
            nonlocal query_counter
            async with query_counter_lock:
                if max_queries > 0 and query_counter >= max_queries:
                    return None
                current = query_counter
                query_counter += 1
                return current

        async def run_context(context_id: int, item: MemoryAgentBenchItem) -> MemoryAgentBenchContextResult:
            async with ctx_sem:
                user_id = self.context_user_id(context_id, item.source)
                add_summary = (
                    await self.add_context(item, context_id, on_chunk_done=add_pbar.update if add_pbar else None)
                    if add
                    else None
                )
                questions = item.build_questions(item.source)

                async def run_question(question: MemoryAgentBenchQuestion) -> MemoryAgentBenchQAResult | None:
                    query_id = await next_query_id()
                    if query_id is None:
                        return None
                    async with qa_sem:
                        result = await self.evaluate_question(
                            user_id,
                            question,
                            query_id=query_id,
                            context_id=context_id,
                            score=score,
                            memory_construction_time=add_summary.memory_construction_time if add_summary else 0.0,
                        )
                        if qa_pbar is not None:
                            qa_pbar.update()
                        return result

                qa_raw = await asyncio.gather(*(run_question(question) for question in questions)) if questions else []
                qa_results = [result for result in qa_raw if result is not None]
                if ctx_pbar is not None:
                    ctx_pbar.update()
                return MemoryAgentBenchContextResult(
                    context_id=context_id,
                    user_id=user_id,
                    num_questions=len(qa_results),
                    qa_results=qa_results,
                    add_summary=add_summary,
                )

        try:
            contexts = await asyncio.gather(*(run_context(idx, item) for idx, item in enumerate(data)))
        finally:
            for pbar in (qa_pbar, ctx_pbar, add_pbar):
                if pbar is not None:
                    pbar.close()

        metric_lists: dict[str, list[float]] = {}
        for ctx in contexts:
            for qa in ctx.qa_results:
                for name, value in qa.metrics.items():
                    metric_lists.setdefault(name, []).append(float(value))
                if qa.primary_metric in qa.metrics:
                    metric_lists.setdefault(f"primary/{qa.source}/{qa.primary_metric}", []).append(
                        float(qa.metrics[qa.primary_metric])
                    )
                metric_lists.setdefault("input_len", []).append(float(qa.input_len))
                metric_lists.setdefault("output_len", []).append(float(qa.output_len))
                metric_lists.setdefault("memory_construction_time", []).append(float(qa.memory_construction_time))
                metric_lists.setdefault("query_time_len", []).append(float(qa.query_time_len))
        averaged = {name: sum(values) / len(values) for name, values in metric_lists.items() if values}
        run = MemoryAgentBenchRunResult(
            sub_dataset=self._sub_dataset,
            primary_metric=self.primary_metric,
            contexts=list(contexts),
            metrics=metric_lists,
            averaged_metrics=averaged,
        )
        if print_report:
            print(run.format_report(), flush=True)
        return run

    @staticmethod
    def load_dataset(
        path: str | Path | None = None,
        *,
        hf_dataset: str = MEMORYAGENTBENCH_HF_DATASET,
        split: str | None = None,
        sub_dataset: str = "",
        limit: int = 0,
        seed: int = 42,
    ) -> list[MemoryAgentBenchItem]:
        """Load MemoryAgentBench data from local JSON/JSONL or HuggingFace.

        Local files may be a list of objects, JSONL records, or an object with a
        top-level ``data`` list. HuggingFace loading is used when ``path`` is not
        supplied.
        """
        del seed  # reserved for future random sampling; official configs use head sampling
        items: list[MemoryAgentBenchItem]
        if path:
            raw = _read_local_dataset(Path(path))
            items = [_coerce_dataset_item(obj, default_source=sub_dataset) for obj in raw]
            if sub_dataset:
                items = [item for item in items if not item.source or item.source == sub_dataset]
        else:
            if not split:
                raise ValueError("`split` is required when loading MemoryAgentBench from HuggingFace.")
            try:
                from datasets import load_dataset
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Install `datasets` to load MemoryAgentBench from HuggingFace.") from exc
            dataset = load_dataset(hf_dataset, split=split, revision="main")
            available_sources = sorted({(sample.get("metadata") or {}).get("source", "") for sample in dataset})
            raw_items = []
            for sample in dataset:
                metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
                if sub_dataset and metadata.get("source", "") != sub_dataset:
                    continue
                raw_items.append(sample)
                if limit > 0 and len(raw_items) >= limit:
                    break
            items = [_coerce_dataset_item(obj, default_source=sub_dataset) for obj in raw_items]
        items = [item for item in items if len(item.context) > 2000]
        if not items and sub_dataset and not path:
            raise ValueError(
                f"No MemoryAgentBench samples matched sub_dataset={sub_dataset!r} in split={split!r}. "
                f"Available sources: {available_sources}"
            )
        return items[:limit] if limit > 0 else items


def _read_local_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        data = payload.get("data", payload.get("items", payload.get("examples")))
        if isinstance(data, list):
            return data
        if "context" in payload:
            return [payload]
    raise ValueError(f"Unsupported MemoryAgentBench dataset file format: {path}")


def save_memoryagentbench_results(path: str | Path, run: MemoryAgentBenchRunResult) -> None:
    """Save results in a shape close to the official runner output."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for ctx in run.contexts:
        for qa in ctx.qa_results:
            record = qa.model_dump()
            record.update(qa.metrics)
            data.append(record)
    payload = {
        "sub_dataset": run.sub_dataset,
        "data": data,
        "metrics": run.metrics,
        "averaged_metrics": run.averaged_metrics,
        "primary_metric": run.primary_metric,
        "contexts": [ctx.model_dump() for ctx in run.contexts],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
