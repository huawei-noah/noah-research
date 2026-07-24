"""PersonaMem-Evo evaluation environment.

This module adapts the PersonaMem-Evo benchmark to the MindMemOS SDK:
- build benchmark items from CSV rows and persona chat histories
- run the add / feedback / search / answer loop for each persona
- score answers via MCQ correctness and chain-level metrics
- save results as JSON + CSV to the runs/ directory
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mindmemos_sdk.memory import AsyncMemoryClient, Message
from tqdm.auto import tqdm

from ....llm import LLMClient

PERSONAMEM_EVO_HF_DATASET = "Aiden0526/PersonaMem-Evo"
DEFAULT_ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the user memories. "
    "Use ONLY the provided memories. If the memories do not contain enough "
    "information, say you don't know. Answer concisely."
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CompletionResult:
    """Result of an LLM completion call with token usage."""
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

@dataclass
class PersonaMemEvoAnswer:
    """One MCQ answer produced by the model."""
    response: str
    predicted_answer: str
    is_correct: bool
    correct_letter: str
    option_mapping: dict[str, str]
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class PersonaMemEvoBuildSummary:
    """Per-persona build phase summary."""
    persona_id: str
    user_id: str
    session_id: str
    total_messages: int = 0
    added_batches: int = 0
    added_messages: int = 0
    returned_memories: int = 0
    add_calls: int = 0
    feedback_calls: int = 0
    feedback_time: float = 0.0
    feedback_request_id: str | None = None
    failed_batches: list[list] = field(default_factory=list)
    feedback_status: str | None = None
    feedback_message: str = ""
    elapsed_seconds: float = 0.0


@dataclass
class PersonaMemEvoItem:
    """One PersonaMem-Evo benchmark item (one question for one persona)."""
    row: dict[str, str]
    index: int
    persona_id: str
    user_id: str
    user_query: str
    correct_answer: str
    incorrect_answers: list[str]
    chain_key: tuple[str, str]
    chat_history_path: Path
    options: dict[str, str]
    correct_letter: str
    ood_type: str | None = None
    ood_difficulty: str | None = None


@dataclass
class PersonaMemEvoQAResult:
    """Result for one QA step."""
    index: int
    persona_id: str
    user_id: str
    question: str
    correct_answer: str
    chain_key: tuple[str, str]
    answer: PersonaMemEvoAnswer | None = None
    retrieved_memories: list[str] = field(default_factory=list)
    prompt: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class PersonaMemEvoRunResult:
    """Result for a full PersonaMem-Evo dataset run."""
    items: list[PersonaMemEvoItem] = field(default_factory=list)
    results: list[PersonaMemEvoQAResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    build_summaries: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_user_query(query_raw: str) -> dict[str, Any]:
    """Parse a raw query and wrap it as a user instruction with recall directive."""
    try:
        parsed = json.loads(query_raw)
    except (json.JSONDecodeError, TypeError):
        parsed = {"role": "user", "content": str(query_raw)}
    content = parsed.get("content", "")
    instruction = (
        "Please recall the relevant information from the memories "
        f"and answer the question.\nQuestion: {content}"
    )
    return {"role": "user", "content": instruction}


def parse_incorrect_answers(raw: str) -> list[str]:
    """Parse a JSON-serialised list of incorrect answers."""
    return json.loads(raw)


def create_mcq_options(
    correct_answer: str,
    distractors: list[str],
    seed: int = 42,
) -> tuple[dict[str, str], str]:
    """Create MCQ options from a correct answer and distractors.

    Returns (options_map, correct_letter) where options_map[letter] = text.
    """
    import random as _random
    rng = _random.Random(seed)
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    candidates = [(correct_answer, True)] + [(d, False) for d in distractors]
    rng.shuffle(candidates)
    options: dict[str, str] = {}
    correct_letter: str | None = None
    for i, (opt_text, is_correct) in enumerate(candidates):
        letter = letters[i]
        options[letter] = opt_text
        if is_correct:
            correct_letter = letter
    assert correct_letter is not None, "at least correct_answer must survive shuffle"
    return options, correct_letter


def extract_final_answer(response: str) -> str:
    """Extract the final answer letter from a model response."""
    match = re.search(r"Final Answer:\s*([A-Za-z])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def check_mcq_correctness(
    correct_letter: str,
    predicted_answer: str,
    options: dict[str, str],
) -> bool:
    """Check if a predicted answer matches the correct option."""
    return correct_letter == predicted_answer


def calculate_personamem_evo_metrics(
    results: list[PersonaMemEvoQAResult],
) -> dict[str, float]:
    """Calculate step-level and chain-level metrics."""
    if not results:
        return {"step_accuracy": 0.0, "chain_total": 0.0, "chain_accuracy": 0.0}
    correct = sum(1 for r in results if r.answer and r.answer.is_correct)
    step_accuracy = correct / len(results)

    chain_groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in results:
        if r.answer is not None:
            chain_groups[r.chain_key].append(r.answer.is_correct)
        else:
            chain_groups[r.chain_key].append(False)
    chain_total = len(chain_groups)
    chain_correct = sum(
        1 for chain_results in chain_groups.values() if all(chain_results)
    )
    chain_accuracy = chain_correct / chain_total if chain_total > 0 else 0.0

    return {
        "step_accuracy": step_accuracy,
        "chain_total": float(chain_total),
        "chain_accuracy": chain_accuracy,
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_benchmark_rows(csv_path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Load PersonaMem-Evo CSV rows.

    Returns (fieldnames, rows).
    """
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def load_chat_history_messages(
    history_path: str | Path,
) -> list[dict[str, Any]]:
    """Load a PersonaMem-Evo chat history JSON file.

    Returns a list of message dicts.
    """
    history_path = Path(history_path)
    with history_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("messages", data.get("chat_history", [data]))
    return []


def save_personamem_evo_results(
    result: PersonaMemEvoRunResult,
    output_path: str | Path,
    *,
    mode: str | None = None,
    size: str | None = None,
) -> None:
    """Save PersonaMem-Evo run results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if mode is not None:
        data["mode"] = mode
    if size is not None:
        data["size"] = size
    data["metrics"] = result.metrics
    data["build_summaries"] = result.build_summaries
    data["items"] = [
        {
            "index": item.index,
            "persona_id": item.persona_id,
            "user_id": item.user_id,
            "chain_key": list(item.chain_key),
            "question": item.user_query,
            "correct_answer": item.correct_answer,
            "options": item.options,
            "correct_letter": item.correct_letter,
            "ood_type": item.ood_type,
            "ood_difficulty": item.ood_difficulty,
        }
        for item in result.items
    ]
    data["results"] = [
        {
            "index": r.index,
            "persona_id": r.persona_id,
            "user_id": r.user_id,
            "chain_key": list(r.chain_key),
            "question": r.question,
            "correct_answer": r.correct_answer,
            "answer": {
                "response": r.answer.response if r.answer else None,
                "predicted_answer": r.answer.predicted_answer if r.answer else None,
                "is_correct": r.answer.is_correct if r.answer else None,
                "correct_letter": r.answer.correct_letter if r.answer else None,
                "option_mapping": r.answer.option_mapping if r.answer else None,
                "prompt_tokens": r.answer.prompt_tokens if r.answer else None,
                "completion_tokens": r.answer.completion_tokens if r.answer else None,
            }
            if r.answer
            else None,
            "retrieved_memories": r.retrieved_memories,
            "error": r.error,
        }
        for r in result.results
    ]
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_personamem_evo_csv(
    result: PersonaMemEvoRunResult,
    output_path: str | Path,
) -> None:
    """Save detailed per-item PersonaMem-Evo results as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "persona_id",
        "user_id",
        "ood_type",
        "ood_difficulty",
        "chain_key",
        "question",
        "correct_answer",
        "correct_letter",
        "predicted_answer",
        "is_correct",
        "model_response",
        "retrieved_memories",
        "prompt",
        "prompt_tokens",
        "completion_tokens",
        "error",
    ]
    items_by_index = {it.index: it for it in result.items}
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in result.results:
            item = items_by_index.get(r.index)
            writer.writerow(
                {
                    "index": r.index,
                    "persona_id": r.persona_id,
                    "user_id": r.user_id,
                    "ood_type": item.ood_type if item else "",
                    "ood_difficulty": item.ood_difficulty if item else "",
                    "chain_key": "||".join(r.chain_key),
                    "question": r.question,
                    "correct_answer": r.correct_answer,
                    "correct_letter": r.answer.correct_letter if r.answer else "",
                    "predicted_answer": r.answer.predicted_answer if r.answer else "",
                    "is_correct": str(r.answer.is_correct).lower() if r.answer else "",
                    "model_response": r.answer.response if r.answer else "",
                    "retrieved_memories": json.dumps(r.retrieved_memories, ensure_ascii=False),
                    "prompt": json.dumps(r.prompt, ensure_ascii=False),
                    "prompt_tokens": str(r.answer.prompt_tokens) if r.answer else "",
                    "completion_tokens": str(r.answer.completion_tokens) if r.answer else "",
                    "error": r.error or "",
                }
            )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PersonaMemEvoEnv:
    """PersonaMem-Evo benchmark environment.

    Builds benchmark items from CSV rows and chat history files, then runs the
    add / feedback / search / answer loop for each persona.
    """

    answer_system_prompt: str = DEFAULT_ANSWER_SYSTEM_PROMPT

    def __init__(
        self,
        memory: AsyncMemoryClient,
        *,
        answer_llm: LLMClient | None = None,
        add_batch_size: int = 10,
        top_k: int = 10,
        user_id_prefix: str = "pm",
        qa_concurrency: int = 8,
        build_concurrency: int = 4,
    ) -> None:
        self._memory = memory
        self._answer_llm = answer_llm
        self._user_id = None
        self._top_k = top_k
        self._search_strategy = "fast"
        self._rerank = False
        self._add_batch_size = add_batch_size
        self._user_id_prefix = user_id_prefix
        self._qa_concurrency = qa_concurrency
        self._build_concurrency = build_concurrency
    # ------------------------------------------------------------------
    # Async overrides (PersonaMemEvoEnv uses AsyncMemoryClient)
    # ------------------------------------------------------------------

    async def add(self, messages: list[Message | dict[str, Any]], **kwargs: Any) -> Any:
        """Add content to the memory store (async override for AsyncMemoryClient)."""
        kwargs.setdefault("user_id", self._user_id)
        return await self._memory.add(messages, **kwargs)

    async def retrieve(self, query: str, **kwargs: Any) -> Any:
        """Retrieve memories for a query (async override for AsyncMemoryClient)."""
        kwargs.setdefault("user_id", self._user_id)
        kwargs.setdefault("top_k", self._top_k)
        kwargs.setdefault("search_strategy", self._search_strategy)
        kwargs.setdefault("rerank", self._rerank)
        return await self._memory.search(query, **kwargs)

    def format_context(self, hits: list[Any]) -> list[str]:
        """Format retrieved memories as context strings."""
        result: list[str] = []
        for hit in hits:
            memory_text = getattr(hit, "memory", "")
            if not memory_text:
                continue
            lineage = getattr(hit, "lineage", None)
            if getattr(lineage, "role", "current") == "archived":
                result.append(f"[历史版本] {memory_text}")
            else:
                result.append(memory_text)
        return result

    def build_messages(self, question: str, contexts: list[str]) -> list[dict[str, Any]]:
        """Build chat messages from a question and retrieved context."""
        if contexts:
            joined = "\n".join(f"[{i}] {context}" for i, context in enumerate(contexts, 1))
            context_block = f"Memories:\n{joined}"
        else:
            context_block = "Memories:\n(none)"
        return [
            {"role": "system", "content": self.answer_system_prompt},
            {"role": "user", "content": f"{context_block}\n\nQuestion: {question}"},
        ]

    # ------------------------------------------------------------------
    # Build items
    # ------------------------------------------------------------------

    def build_items(
        self,
        rows: list[dict[str, str]],
        size: str = "32k",
        persona_root: Path | None = None,
    ) -> list[PersonaMemEvoItem]:
        """Build benchmark items from CSV rows."""
        items: list[PersonaMemEvoItem] = []
        for idx, row in enumerate(rows):
            persona_id = row.get("persona_id", str(idx))
            user_id = f"{self._user_id_prefix}-{persona_id}"
            query_raw = row.get("user_query", "")
            user_query = parse_user_query(query_raw).get("content", "")
            correct_answer = row.get("correct_answer", "")
            incorrect_raw = row.get("incorrect_answers", "[]")
            incorrect_answers = parse_incorrect_answers(incorrect_raw)
            chain_key = (row.get("chain_id", "unknown"), persona_id)
            history_rel = row.get(f"chat_history_{size}_link", "")
            history_path = (
                persona_root / history_rel
                if persona_root and history_rel
                else Path(history_rel)
            )
            options, correct_letter = create_mcq_options(
                correct_answer, incorrect_answers, seed=idx
            )
            item = PersonaMemEvoItem(
                row=row, index=idx, persona_id=persona_id, user_id=user_id,
                user_query=str(query_raw), correct_answer=correct_answer,
                incorrect_answers=incorrect_answers, chain_key=chain_key,
                chat_history_path=history_path, options=options,
                correct_letter=correct_letter,
                ood_type=row.get("ood_type"),
                ood_difficulty=row.get("ood_difficulty"),
            )
            items.append(item)
        return items

    # ------------------------------------------------------------------
    # Run full dataset
    # ------------------------------------------------------------------

    async def run_dataset(
        self,
        rows: list[dict[str, str]],
        size: str = "32k",
        persona_root: Path | None = None,
        mode: str = "add_feedback",
        show_progress: bool = True,
    ) -> PersonaMemEvoRunResult:
        """Run a full PersonaMem-Evo dataset evaluation.

        Phase 1 - Build: add chat histories + feedback once per persona.
        Phase 2 - QA:  search + LLM answer + grade per item.
        """
        items = self.build_items(rows, size=size, persona_root=persona_root)

        # Group items by persona
        persona_items: dict[str, list[PersonaMemEvoItem]] = defaultdict(list)
        for item in items:
            persona_items[item.persona_id].append(item)

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Phase 1: Build (add + feedback) per persona
        # ------------------------------------------------------------------
        build_summaries: list[dict[str, Any]] = []
        _build_sem = asyncio.Semaphore(self._build_concurrency)
        build_pbar = tqdm(
            total=len(persona_items),
            disable=not show_progress,
            desc=f"Build PersonaMem-Evo ({mode})",
            unit="persona",
        )

        async def _build_one(
            persona_id: str,
            p_items: list[PersonaMemEvoItem],
        ) -> dict[str, Any]:
            async with _build_sem:
                user_id = f"{self._user_id_prefix}-{mode}-{persona_id}"
                session_id = f"personamem-evo-{mode}-{persona_id}"
                metadata = {"benchmark": "personamem-evo"}

                chat_history_path = p_items[0].chat_history_path
                messages = (
                    load_chat_history_messages(chat_history_path)
                    if chat_history_path.exists()
                    else []
                )

                build_start = time.monotonic()
                tqdm.write(
                    f"  [Build] persona {persona_id}: "
                    f"{len(messages)} messages, mode={mode}"
                )
                add_calls = 0
                added = 0
                returned = 0
                failed_batches: list[list] = []

                if mode in ("add_feedback", "add_only"):
                    n_batches = (len(messages) + self._add_batch_size - 1) // self._add_batch_size
                    if n_batches > 1:
                        batch_pbar = tqdm(
                            total=n_batches,
                            desc=f"  [{persona_id}] add",
                            unit="batch",
                            leave=False,
                        )
                    for i in range(0, len(messages), self._add_batch_size):
                        batch = messages[i : i + self._add_batch_size]
                        try:
                            add_result = await self.add(
                                batch,
                                user_id=user_id,
                                session_id=session_id,
                                metadata=metadata,
                            )
                            add_calls += 1
                            added += len(batch)
                            returned += len(getattr(add_result, "memories", []) or [])
                        except Exception as exc:
                            failed_batches.append([i // self._add_batch_size, str(exc)])
                            add_calls += 1
                        if n_batches > 1:
                            batch_pbar.update()
                    if n_batches > 1:
                        batch_pbar.close()

                feedback_calls = 0
                fb_time = 0.0
                fb_status: str | None = None
                fb_request_id: str | None = None
                fb_msg = ""
                fb_start = time.monotonic()

                if mode in ("add_feedback",):
                    tqdm.write(
                        f"  [Feedback] persona {persona_id}: starting implicit feedback..."
                    )
                    try:
                        fb_result = await self._memory.feedback(
                            mode="sync",
                            user_id=user_id,
                        )
                        feedback_calls = 1
                        fb_status = "ok"
                        fb_request_id = getattr(fb_result, "request_id", None)
                        fb_msg = getattr(fb_result, "message", "")
                    except Exception as exc:
                        fb_status = "error"
                        fb_msg = f"{type(exc).__name__}: {exc}"

                fb_time = time.monotonic() - fb_start
                build_elapsed = time.monotonic() - build_start
                fb_detail = f" ({fb_msg})" if fb_msg else ""
                tqdm.write(
                    f"  [Build] persona {persona_id} done: "
                    f"{added}/{len(messages)} msgs added, "
                    f"{feedback_calls} feedback calls ({fb_status}){fb_detail}, "
                    f"{build_elapsed:.1f}s"
                )

                summary = {
                    "persona_id": persona_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "total_messages": len(messages),
                    "added_batches": add_calls,
                    "added_messages": added,
                    "returned_memories": returned,
                    "add_calls": add_calls,
                    "feedback_calls": feedback_calls,
                    "feedback_time": fb_time,
                    "feedback_request_id": fb_request_id,
                    "failed_batches": failed_batches,
                    "feedback_status": fb_status,
                    "feedback_message": fb_msg,
                    "elapsed_seconds": build_elapsed,
                }
                build_pbar.update()
                return summary

        persona_ids_sorted = sorted(
            persona_items.keys(),
            key=lambda x: int(x) if x.isdigit() else x,
        )
        build_tasks = [
            _build_one(pid, persona_items[pid])
            for pid in persona_ids_sorted
        ]
        build_summaries_results = await asyncio.gather(*build_tasks, return_exceptions=True)
        build_pbar.close()

        # Aggregate build results
        total_build_elapsed = 0.0
        total_feedback_elapsed = 0.0
        total_add_calls = 0
        total_feedback_calls = 0
        total_added_messages = 0
        total_returned_from_add = 0
        build_summaries = []
        for s in build_summaries_results:
            if isinstance(s, dict):
                build_summaries.append(s)
                total_add_calls += s["add_calls"]
                total_feedback_calls += s["feedback_calls"]
                total_feedback_elapsed += s["feedback_time"]
                total_added_messages += s["added_messages"]
                total_returned_from_add += s["returned_memories"]
                total_build_elapsed += s["elapsed_seconds"]
            else:
                tqdm.write(f"Build exception for persona: {s}")
        # Phase 2: QA (search + LLM answer + grade) per item
        # ------------------------------------------------------------------
        results: list[PersonaMemEvoQAResult] = []
        results_lock = asyncio.Lock()
        _search_elapsed = 0.0
        _llm_elapsed = 0.0
        _api_search_calls = 0
        _answer_llm_calls = 0
        _total_prompt_tokens = 0
        _total_completion_tokens = 0
        _metrics_lock = asyncio.Lock()

        sem = asyncio.Semaphore(self._qa_concurrency)
        qa_pbar = tqdm(
            total=len(items),
            disable=not show_progress,
            desc="PersonaMem-Evo QA",
            unit="item",
        )

        async def _process_one(item: PersonaMemEvoItem) -> None:
            nonlocal _search_elapsed, _llm_elapsed, _api_search_calls, _answer_llm_calls, _total_prompt_tokens, _total_completion_tokens
            async with sem:
                user_id = f"{self._user_id_prefix}-{mode}-{item.persona_id}"

                # --- search ---
                search_start = time.monotonic()
                try:
                    search = await self.retrieve(
                        item.user_query, user_id=user_id, top_k=self._top_k
                    )
                except Exception as exc:
                    async with _metrics_lock:
                        _search_elapsed += time.monotonic() - search_start
                        _api_search_calls += 1
                    async with results_lock:
                        results.append(PersonaMemEvoQAResult(
                            index=item.index, persona_id=item.persona_id,
                            user_id=user_id, question=str(item.user_query),
                            correct_answer=item.correct_answer,
                            chain_key=item.chain_key, answer=None,
                            error=f"search failed: {exc}",
                        ))
                    qa_pbar.update()
                    return

                async with _metrics_lock:
                    _search_elapsed += time.monotonic() - search_start
                    _api_search_calls += 1

                contexts = self.format_context(
                    search.memories if hasattr(search, "memories") else []
                )

                # Build prompt with MCQ options and output format instruction
                options_str = "\n".join(
                    f"{letter}. {text}" for letter, text in sorted(item.options.items())
                )
                mcq_instruction = (
                    "Options:\n"
                    f"{options_str}\n\n"
                    "Please output your answer as 'Final Answer: <letter>' "
                    "(e.g. 'Final Answer: A')."
                )
                messages_prompt = self.build_messages(item.user_query, contexts)
                messages_prompt[-1]["content"] += f"\n\n{mcq_instruction}"

                # --- LLM answer ---
                llm_start = time.monotonic()
                try:
                    llm_result = await self._answer_llm.complete(messages_prompt)
                    answer_text = llm_result.content
                except Exception as exc:
                    async with _metrics_lock:
                        _llm_elapsed += time.monotonic() - llm_start
                        _answer_llm_calls += 1
                    async with results_lock:
                        results.append(PersonaMemEvoQAResult(
                            index=item.index, persona_id=item.persona_id,
                            user_id=user_id, question=str(item.user_query),
                            correct_answer=item.correct_answer,
                            chain_key=item.chain_key, answer=None,
                            error=f"llm failed: {exc}",
                            retrieved_memories=contexts, prompt=messages_prompt,
                        ))
                    qa_pbar.update()
                    return

                pt = llm_result.prompt_tokens
                ct = llm_result.completion_tokens
                async with _metrics_lock:
                    _llm_elapsed += time.monotonic() - llm_start
                    _answer_llm_calls += 1
                    _total_prompt_tokens += pt
                    _total_completion_tokens += ct

                predicted = extract_final_answer(answer_text)
                is_correct = check_mcq_correctness(
                    item.correct_letter, predicted, item.options
                )
                tqdm.write(
                    f"  [QA] item {item.index:3d}/{len(items):3d} "
                    f"(persona {item.persona_id}): "
                    f"answer={predicted} correct={str(is_correct).lower():>5} "
                    f"tokens={pt}+{ct}"
                )
                answer = PersonaMemEvoAnswer(
                    response=answer_text, predicted_answer=predicted,
                    is_correct=is_correct, correct_letter=item.correct_letter,
                    option_mapping=item.options,
                    prompt_tokens=pt, completion_tokens=ct,
                )
                async with results_lock:
                    results.append(PersonaMemEvoQAResult(
                        index=item.index, persona_id=item.persona_id,
                        user_id=user_id, question=str(item.user_query),
                        correct_answer=item.correct_answer,
                        chain_key=item.chain_key, answer=answer,
                        retrieved_memories=contexts, prompt=messages_prompt,
                    ))
                qa_pbar.update()

        tasks = [_process_one(item) for item in items]
        await asyncio.gather(*tasks, return_exceptions=True)
        qa_pbar.close()


        search_elapsed = _search_elapsed
        llm_elapsed = _llm_elapsed
        api_search_calls = _api_search_calls
        answer_llm_calls = _answer_llm_calls
        total_elapsed = total_build_elapsed + search_elapsed + llm_elapsed

        # ------------------------------------------------------------------
        # Aggregate metrics
        # ------------------------------------------------------------------
        qa_metrics = calculate_personamem_evo_metrics(results)

        # Breakdown by ood_type / difficulty
        ood_type_groups: dict[str, list[bool]] = defaultdict(list)
        difficulty_groups: dict[str, list[bool]] = defaultdict(list)
        items_by_index = {it.index: it for it in items}
        for r in results:
            item = items_by_index.get(r.index)
            if item is not None and item.ood_type and r.answer is not None:
                ood_type_groups[item.ood_type].append(r.answer.is_correct)
            if item is not None and item.ood_difficulty and r.answer is not None:
                difficulty_groups[item.ood_difficulty].append(r.answer.is_correct)

        by_ood_type = {
            ot: {
                "correct": sum(1 for v in vals if v),
                "total": len(vals),
                "accuracy": sum(1 for v in vals if v) / len(vals) if vals else 0.0,
            }
            for ot, vals in sorted(ood_type_groups.items())
        }
        by_difficulty = {
            d: {
                "correct": sum(1 for v in vals if v),
                "total": len(vals),
                "accuracy": sum(1 for v in vals if v) / len(vals) if vals else 0.0,
            }
            for d, vals in sorted(difficulty_groups.items())
        }

        # Chain size distribution
        chain_counts: dict[tuple[str, str], int] = defaultdict(int)
        for r in results:
            chain_counts[r.chain_key] += 1
        chain_size_dist: dict[str, int] = {}
        for sz in sorted(set(chain_counts.values())):
            chain_size_dist[str(sz)] = sum(
                1 for v in chain_counts.values() if v == sz
            )

        step_correct = sum(1 for r in results if r.answer and r.answer.is_correct)

        metrics: dict[str, float] = {
            **qa_metrics,
            "step_total": float(len(results)),
            "step_correct": float(step_correct),
            "chain_size_distribution": str(chain_size_dist),
            "by_ood_type": str(by_ood_type),
            "by_difficulty": str(by_difficulty),
            "memory_added_messages": float(total_added_messages),
            "memory_returned_count": float(total_returned_from_add),
            "api_add_calls": float(total_add_calls),
            "api_feedback_calls": float(total_feedback_calls),
            "api_search_calls": float(api_search_calls),
            "prompt_tokens": float(_total_prompt_tokens),
            "completion_tokens": float(_total_completion_tokens),
            "answer_llm_calls": float(answer_llm_calls),
            "api_total_calls": float(
                total_add_calls + total_feedback_calls + api_search_calls
            ),
            "build_elapsed_seconds": total_build_elapsed,
            "qa_search_elapsed_seconds": search_elapsed,
            "qa_answer_elapsed_seconds": llm_elapsed,
            "feedback_elapsed_seconds": total_feedback_elapsed,
            "total_elapsed_seconds": total_elapsed,
        }

        return PersonaMemEvoRunResult(
            items=items, results=results, metrics=metrics,
            build_summaries=build_summaries,
        )
