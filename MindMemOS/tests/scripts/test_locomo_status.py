from __future__ import annotations

import json
import sys
from pathlib import Path

LOCOMO_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "locomo"
if str(LOCOMO_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(LOCOMO_SCRIPT_DIR))

from locomo_status import add_progress, full_progress, judge_progress, search_progress  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "conversation": {
                        "session_1": [{"speaker": "speaker 1", "text": "hello"}],
                        "session_1_date_time": "2024-01-01",
                        "session_2": [{"speaker": "speaker 2", "text": "later"}],
                        "session_2_date_time": "2024-01-02",
                    },
                    "qa": [
                        {"question": "q1", "category": 1},
                        {"question": "q2", "category": 2},
                        {"question": "ignored", "category": 5},
                    ],
                },
                {
                    "conversation": {
                        "session_1": [{"speaker": "speaker 1", "text": "second"}],
                        "session_1_date_time": "2024-01-03",
                    },
                    "qa": [
                        {"question": "q3", "category": 1},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )


def test_add_progress_groups_by_conversation_and_ready_count(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    out = tmp_path / "out"
    out.mkdir()
    _write_dataset(dataset)
    _write_jsonl(
        out / "add_results.jsonl",
        [
            {"conversation_index": 0, "session_key": "session_1", "success": True},
            {"conversation_index": 0, "session_key": "session_2", "success": True},
            {"conversation_index": 1, "session_key": "session_1", "success": False},
        ],
    )

    progress = add_progress(dataset, out)

    assert progress["ok"] == 2
    assert progress["recorded"] == 3
    assert progress["expected"] == 3
    assert progress["ready_conversations"] == 1
    assert progress["by_conversation"]["0"] == {
        "ok": 2,
        "recorded": 2,
        "failed": 0,
        "expected": 2,
        "percent": 100.0,
    }
    assert progress["by_conversation"]["1"]["failed"] == 1


def test_search_and_judge_progress_group_by_conversation_and_category(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    out = tmp_path / "out"
    out.mkdir()
    _write_dataset(dataset)
    _write_jsonl(
        out / "search_results.jsonl",
        [
            {"conversation_index": 0, "question_index": 0, "category": 1, "success": False},
            {"conversation_index": 0, "question_index": 0, "category": 1, "success": True},
            {"conversation_index": 1, "question_index": 0, "category": 1, "success": True},
        ],
    )
    _write_jsonl(
        out / "evaluation_metrics.jsonl",
        [
            {"conversation_index": 0, "question_index": 0, "category": 1, "llm_score": 1},
            {"conversation_index": 0, "question_index": 1, "category": 2, "success": False, "llm_score": 0},
        ],
    )

    search = search_progress(dataset, out)
    judge = judge_progress(dataset, out)

    assert search["ok"] == 2
    assert search["recorded"] == 2
    assert search["expected"] == 3
    assert search["by_conversation"]["0"]["ok"] == 1
    assert search["by_conversation"]["0"]["expected"] == 2
    assert search["by_category"]["1"]["ok"] == 2
    assert search["by_category"]["1"]["expected"] == 2
    assert search["by_category"]["2"]["expected"] == 1

    assert judge["ok"] == 1
    assert judge["recorded"] == 2
    assert judge["by_conversation"]["0"]["failed"] == 1
    assert judge["by_category"]["1"]["ok"] == 1
    assert judge["by_category"]["2"]["failed"] == 1


def test_full_progress_contains_all_status_sections(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    out = tmp_path / "out"
    out.mkdir()
    _write_dataset(dataset)

    progress = full_progress(dataset, out)

    assert set(progress) == {"add", "search", "answer_generation", "judge", "scores_so_far"}
    assert progress["add"]["expected"] == 3
    assert progress["search"]["expected"] == 3
    assert progress["scores_so_far"]["overall"] == {"llm_score": 0.0, "count": 0, "correct": 0}
