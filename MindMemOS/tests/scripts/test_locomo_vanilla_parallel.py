from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

LOCOMO_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "locomo"
if str(LOCOMO_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(LOCOMO_SCRIPT_DIR))

import locomo_vanilla_eval as runner  # noqa: E402


def _dataset() -> list[dict]:
    return [
        {
            "sample_id": "conv-a",
            "conversation": {
                "session_1": [{"speaker": "speaker 1", "text": "hello"}],
                "session_1_date_time": "2024-01-01",
            },
            "qa": [
                {"question": "q1", "answer": "a1", "category": 1},
                {"question": "q2", "answer": "a2", "category": 2},
            ],
        },
        {
            "sample_id": "conv-b",
            "conversation": {
                "session_1": [{"speaker": "speaker 1", "text": "first"}],
                "session_1_date_time": "2024-01-01",
                "session_2": [{"speaker": "speaker 1", "text": "second"}],
                "session_2_date_time": "2024-01-02",
            },
            "qa": [
                {"question": "q3", "answer": "a3", "category": 1},
            ],
        },
    ]


class _FakeAddPipeline:
    def __init__(self, search_started: asyncio.Event) -> None:
        self.search_started = search_started

    async def add_sync(self, inp, context):
        if context.request_id.endswith(":add:1:session_2"):
            await asyncio.wait_for(self.search_started.wait(), timeout=1)
        return SimpleNamespace(memories=[])


class _FakeSearchPipeline:
    def __init__(
        self, *, search_started: asyncio.Event | None = None, require_parallel_questions: bool = False
    ) -> None:
        self.search_started = search_started
        self.require_parallel_questions = require_parallel_questions
        self.started = 0
        self.both_started = asyncio.Event()

    async def search(self, inp, context):
        if self.search_started is not None:
            self.search_started.set()
        if self.require_parallel_questions:
            self.started += 1
            if self.started == 2:
                self.both_started.set()
            await asyncio.wait_for(self.both_started.wait(), timeout=1)
        return SimpleNamespace(memories=[])


class _RecordingSearchPipeline:
    def __init__(self) -> None:
        self.inputs = []

    async def search(self, inp, context):
        self.inputs.append((inp, context))
        return SimpleNamespace(memories=[])


async def _fake_score_record(record):
    answer = {
        "conversation_index": record["conversation_index"],
        "question_index": record["question_index"],
        "question": record["question"],
        "locomo_answer": record["answer"],
        "model_answer": "answer",
        "category": str(record["category"]),
        "answer_memory_count": 0,
        "success": True,
    }
    metric = {
        "conversation_index": record["conversation_index"],
        "question_index": record["question_index"],
        "question": record["question"],
        "answer": str(record["answer"]),
        "response": "answer",
        "category": str(record["category"]),
        "llm_score": 1,
        "judge_label": "CORRECT",
        "success": True,
    }
    return answer, metric


def test_add_search_starts_search_when_one_conversation_is_ready(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "_score_record", _fake_score_record)
    search_started = asyncio.Event()

    async def run() -> None:
        await runner._run_add_search_phase(
            _FakeAddPipeline(search_started),
            _FakeSearchPipeline(search_started=search_started),
            _dataset(),
            output_dir=tmp_path,
            collection_prefix="locomo_test",
            top_k=5,
            add_concurrency=2,
            search_concurrency=1,
            score_concurrency=1,
            started_at=0.0,
            max_questions=None,
            categories=None,
        )

    asyncio.run(run())

    assert len(runner._read_jsonl(tmp_path / "add_results.jsonl")) == 3
    assert len(runner._read_jsonl(tmp_path / "search_results.jsonl")) == 3
    assert len(runner._read_jsonl(tmp_path / "evaluation_metrics.jsonl")) == 3


def test_search_phase_runs_questions_concurrently_within_ready_conversation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "_score_record", _fake_score_record)

    async def run() -> None:
        await runner._run_search_phase(
            _FakeSearchPipeline(require_parallel_questions=True),
            [_dataset()[0]],
            output_dir=tmp_path,
            collection_prefix="locomo_test",
            top_k=5,
            conversation_concurrency=1,
            search_concurrency=2,
            score_concurrency=1,
            started_at=0.0,
            max_questions=None,
            categories=None,
        )

    asyncio.run(run())

    assert len(runner._read_jsonl(tmp_path / "search_results.jsonl")) == 2


def test_search_question_filters_to_current_conversation_user() -> None:
    pipeline = _RecordingSearchPipeline()

    async def run() -> None:
        await runner._search_question(
            pipeline,
            _dataset()[0],
            conversation_index=3,
            question_index=0,
            qa={"question": "q", "answer": "a", "category": 1},
            collection_prefix="locomo_test",
            top_k=5,
        )

    asyncio.run(run())

    assert pipeline.inputs[0][0].filters == {"user_id": "conv_3"}
    assert pipeline.inputs[0][0].rerank is True
