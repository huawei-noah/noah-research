"""Skill benchmark CLI argument groups."""

from __future__ import annotations

import argparse
import os

DEFAULT_SPREADSHEETBENCH_DATA_URL = (
    "https://huggingface.co/datasets/KAKA22/SpreadsheetBench/resolve/main/spreadsheetbench_verified_400.tar.gz"
)
DEFAULT_SPREADSHEETBENCH_SEED = 1447


def add_common_skill_args(parser: argparse.ArgumentParser) -> None:
    """Register arguments shared by skill benchmark datasets."""
    parser.add_argument(
        "--run-dir",
        metavar="PATH",
        help="Directory where benchmark workdirs, summaries, trajectories, and evolution events are written.",
    )
    parser.add_argument(
        "--trajectory-path",
        metavar="PATH",
        help="JSONL path for per-task reasoning trajectories; defaults to the benchmark run directory when unset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of shuffled benchmark cases to run; must be parseable as an integer.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of benchmark cases to run concurrently; must be parseable as an integer.",
    )
    parser.add_argument(
        "--skill",
        action="append",
        default=None,
        metavar="PATH",
        help="Skill directory containing SKILL.md to stage for each case; may be repeated.",
    )


def add_openai_llm_args(parser: argparse.ArgumentParser) -> None:
    """Register OpenAI-compatible agent LLM arguments."""
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        metavar="MODEL",
        help="OpenAI-compatible model name used by the benchmark agent.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        metavar="KEY",
        help="API key for the benchmark agent LLM; defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_ENDPOINT"),
        metavar="URL",
        help="OpenAI-compatible base URL for the benchmark agent LLM; defaults to OPENAI_ENDPOINT.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Sampling temperature for the benchmark agent LLM; must be parseable as a float.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Maximum output tokens per benchmark agent LLM call; must be parseable as an integer.",
    )


def add_skill_evolution_args(parser: argparse.ArgumentParser) -> None:
    """Register MindMemOS skill-evolution service arguments."""
    parser.add_argument(
        "--evolve",
        action="store_true",
        help="Whether to trigger skill evolution after each batch of tasks; omit for no-evolution baseline.",
    )
    parser.add_argument(
        "--evolve-every",
        type=int,
        default=1,
        metavar="N",
        help="Number of tasks per evolution batch; must be parseable as an integer and is clamped to at least 1.",
    )
    parser.add_argument(
        "--evolution-base-url",
        default=os.getenv("MINDMEMOS_BASE_URL"),
        metavar="URL",
        help="MindMemOS FastAPI base URL hosting /v1/skills/evolve; required for real evolution.",
    )
    parser.add_argument(
        "--evolution-api-key",
        default=os.getenv("MINDMEMOS_API_KEY"),
        metavar="KEY",
        help="MindMemOS API key used for skill registration, memory trace recording, and evolution calls.",
    )
    parser.add_argument(
        "--continue-on-evolution-error",
        action="store_true",
        help="Whether to log evolution failures and continue instead of failing the run.",
    )


def add_spreadsheetbench_args(parser: argparse.ArgumentParser) -> None:
    """Register SpreadsheetBench dataset arguments."""
    parser.add_argument(
        "--data-root",
        default="data/SpreadsheetBench",
        metavar="PATH",
        help="Directory containing or receiving SpreadsheetBench data; created or populated when download is enabled.",
    )
    parser.add_argument(
        "--data-url",
        default=DEFAULT_SPREADSHEETBENCH_DATA_URL,
        metavar="URL",
        help="SpreadsheetBench dataset tar.gz URL used only when the local dataset is missing and download is enabled.",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to download and extract the dataset when --data-root does not contain it; use --download or --no-download.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SPREADSHEETBENCH_SEED,
        metavar="N",
        help="Shuffle seed for case order; must be parseable as an integer.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to show a tqdm progress bar with running score; use --progress or --no-progress.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=15,
        metavar="N",
        help="Maximum agent tool-calling turns per case; must be parseable as an integer.",
    )
    parser.add_argument(
        "--python-path",
        metavar="PATH",
        help="Python executable path used by the agent shell tool inside case workdirs.",
    )
    add_openai_llm_args(parser)
    add_skill_evolution_args(parser)
