"""SpreadsheetBench evaluation environment."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ...agents import Message, RunResult
from ...evolve.algo import EvolveOutcome
from .data import prepare_data_root
from .evaluator import compare_workbooks

Split = Literal["train", "val", "test", "all"]

SYSTEM_PROMPT = (
    "You are an expert spreadsheet assistant. Your working directory contains a "
    "source Excel file named 'input.xlsx'. Do NOT modify 'input.xlsx'. Instead, "
    "produce a new file 'output.xlsx' in the same directory that fully satisfies "
    "the user's request (start from a copy of 'input.xlsx' and apply your changes).\n"
    "Work by writing and running Python (openpyxl is available) through the shell "
    "tool — do not answer from memory. Inspect the sheets first, apply the changes, "
    "save to 'output.xlsx', and verify.\n"
    "IMPORTANT: 'output.xlsx' is graded by reading cached cell VALUES, with no "
    "formula recalculation. Write the final computed values into the target cells "
    "(not bare formulas), since an unevaluated formula reads back as empty.\n"
    "When you are done, stop without calling any tool."
)


@dataclass
class SpreadsheetBenchCase:
    """One SpreadsheetBench task."""

    id: str
    prompt: str
    split: Split
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpreadsheetBenchCaseResult:
    """Result for one SpreadsheetBench task rollout."""

    case_id: str
    split: str
    score: float
    finished: bool
    turns: int
    workdir: str
    messages: list[Message]
    error: str | None = None
    score_message: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    rollout: int = 0


@dataclass
class SpreadsheetBenchRunResult:
    """Aggregate result for a SpreadsheetBench run."""

    split: str
    total: int
    correct: int
    accuracy: float
    results: list[SpreadsheetBenchCaseResult]
    run_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpreadsheetBenchEnv:
    """SpreadsheetBench Verified-400 benchmark data, prompts, and scoring."""

    name = "spreadsheetbench"
    default_run_dir = "results/spreadsheetbench_eval"
    default_concurrency = 40
    input_name = "input.xlsx"
    output_name = "output.xlsx"

    def __init__(
        self,
        data_root: Path | str,
        run_dir: Path | str,
        *,
        trajectory_path: Path | str | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._trajectory_path = Path(trajectory_path) if trajectory_path is not None else None

        self.verified = self.data_root / "spreadsheetbench_verified_400"
        self.split_root = self.data_root / "spreadsheetbench_id_split"
        with (self.verified / "dataset.json").open(encoding="utf-8") as file:
            self._dataset: dict[str, dict[str, Any]] = {str(record["id"]): record for record in json.load(file)}

    def load_cases(self, split: Split = "all") -> list[SpreadsheetBenchCase]:
        if split == "all":
            return self.load_all_cases()

        with (self.split_root / split / "items.json").open(encoding="utf-8") as file:
            items = json.load(file)

        cases: list[SpreadsheetBenchCase] = []
        for item in items:
            record_id = str(item["id"])
            cases.append(self._case_from_record(record_id, self._dataset[record_id], split))
        return cases

    def load_all_cases(self) -> list[SpreadsheetBenchCase]:
        """Load every task in ``dataset.json`` without train/val/test split files."""
        return [
            self._case_from_record(record_id, self._dataset[record_id], "all")
            for record_id in sorted(self._dataset, key=_record_sort_key)
        ]

    def setup_case(self, case: SpreadsheetBenchCase, workdir: Path) -> None:
        init = self._workbook(case.data["src_dir"], "init")
        shutil.copyfile(init, workdir / self.input_name)

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_messages(self, case: SpreadsheetBenchCase) -> list[Message]:
        return [
            {
                "role": "user",
                "content": (
                    "The source file 'input.xlsx' is in your working directory.\n\n"
                    f"Task:\n{case.prompt}\n\n"
                    "Complete the task and save the result as 'output.xlsx' (do not modify 'input.xlsx')."
                ),
            }
        ]

    def score(self, case: SpreadsheetBenchCase, workdir: Path) -> tuple[float, str]:
        output = workdir / self.output_name
        if not output.exists():
            return 0.0, "output.xlsx not found"
        golden = self._workbook(case.data["src_dir"], "golden")
        ok, message = compare_workbooks(golden, output, self.answer_position(case))
        return (1.0 if ok else 0.0), message

    def case_workdir(self, case: SpreadsheetBenchCase, rollout: int = 0) -> Path:
        return self.run_dir / "cases" / case.id / f"rollout_{rollout}"

    def build_result(
        self,
        case: SpreadsheetBenchCase,
        run_result: RunResult,
        *,
        workdir: Path,
        score: float,
        score_message: str,
        error: str | None,
        started_at: float,
        ended_at: float,
        rollout: int = 0,
    ) -> SpreadsheetBenchCaseResult:
        return SpreadsheetBenchCaseResult(
            case_id=case.id,
            split=case.split,
            score=score,
            finished=run_result.finished,
            turns=run_result.turns,
            workdir=str(workdir),
            messages=run_result.messages,
            error=error,
            score_message=score_message,
            started_at=started_at,
            ended_at=ended_at,
            rollout=rollout,
        )

    def trajectory_path(self, split: str) -> Path:
        if self._trajectory_path is not None:
            return self._trajectory_path
        return self.run_dir / f"{split}_trajectories.jsonl"

    def evolution_events_path(self, split: str) -> Path:
        return self.run_dir / f"{split}_evolution_events.jsonl"

    def append_trajectory(self, result: SpreadsheetBenchCaseResult) -> None:
        path = self.trajectory_path(result.split)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    def append_evolution_event(
        self,
        *,
        split: str,
        batch_index: int,
        batch_start: int,
        batch_end: int,
        batch_results: list[SpreadsheetBenchCaseResult],
        started_at: float,
        ended_at: float,
        outcomes: list[EvolveOutcome] | None = None,
        error: BaseException | None = None,
    ) -> None:
        path = self.evolution_events_path(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "evolved_at": _iso_utc(ended_at),
            "started_at": _iso_utc(started_at),
            "ended_at": _iso_utc(ended_at),
            "duration_seconds": ended_at - started_at,
            "split": split,
            "batch_index": batch_index,
            "batch_start": batch_start,
            "batch_end": batch_end,
            "case_ids": [result.case_id for result in batch_results],
            "outcomes": [asdict(outcome) for outcome in (outcomes or [])],
            "error": f"{type(error).__name__}: {error}" if error is not None else None,
        }
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def summarize(self, split: str, results: list[SpreadsheetBenchCaseResult]) -> SpreadsheetBenchRunResult:
        total = len(results)
        correct = sum(1 for result in results if result.score >= 1.0)
        accuracy = correct / total if total else 0.0
        run_result = SpreadsheetBenchRunResult(
            split=split,
            total=total,
            correct=correct,
            accuracy=accuracy,
            results=results,
            run_dir=str(self.run_dir),
        )
        summary_path = self.run_dir / f"{split}_summary.json"
        summary_path.write_text(json.dumps(run_result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return run_result

    def _case_from_record(self, record_id: str, record: dict[str, Any], split: Split) -> SpreadsheetBenchCase:
        return SpreadsheetBenchCase(
            id=record_id,
            prompt=record["instruction"],
            split=split,
            data={
                "src_dir": str(self.verified / record["spreadsheet_path"]),
                "answer_position": record["answer_position"],
                "answer_sheet": record.get("answer_sheet"),
                "instruction_type": record.get("instruction_type"),
            },
        )

    @staticmethod
    def answer_position(case: SpreadsheetBenchCase) -> str:
        position = str(case.data.get("answer_position") or "")
        sheet = case.data.get("answer_sheet")
        if position and sheet and "!" not in position:
            position = f"{sheet}!{position}"
        return position

    @staticmethod
    def _workbook(src_dir: str, kind: Literal["init", "golden"]) -> Path:
        key = "init" if kind == "init" else "golden"
        hits = sorted(Path(src_dir).glob(f"*{key}*.xlsx"))
        if not hits and kind == "init":
            hits = sorted(Path(src_dir).glob("initial.xlsx"))
        if not hits and kind == "golden":
            hits = sorted(Path(src_dir).glob("golden.xlsx"))
        if not hits:
            raise FileNotFoundError(f"No {kind} workbook in {src_dir}")
        return hits[0]


def build_env(args: Any) -> SpreadsheetBenchEnv:
    """Build SpreadsheetBench from CLI arguments."""
    data_root = prepare_data_root(Path(args.data_root), args.data_url, download=args.download)
    return SpreadsheetBenchEnv(
        data_root=data_root,
        run_dir=args.run_dir or SpreadsheetBenchEnv.default_run_dir,
        trajectory_path=args.trajectory_path,
    )


def _record_sort_key(record_id: str) -> tuple[int, int | str]:
    return (0, int(record_id)) if record_id.isdigit() else (1, record_id)


def _iso_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
