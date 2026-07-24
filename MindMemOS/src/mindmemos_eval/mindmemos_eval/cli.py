"""MindMemOS evaluation CLI."""

from __future__ import annotations

import argparse
import asyncio

try:
    from .memory import add_memory_args, run_benchmark_matrix
    from .skills import add_skill_args, run_skill_benchmark
except ImportError:  # pragma: no cover - used when running this file directly
    from mindmemos_eval.memory import add_memory_args, run_benchmark_matrix
    from mindmemos_eval.skills import add_skill_args, run_skill_benchmark


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the MindMemOS evaluation CLI parser."""
    parser = argparse.ArgumentParser(description="Run MindMemOS evaluation benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    memory = subparsers.add_parser("memory", help="Run memory benchmark matrix add/search evaluations.")
    add_memory_args(memory)

    skill = subparsers.add_parser("skill", help="Run skill benchmark self-evolution evaluation.")
    add_skill_args(skill)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    if args.command == "memory":
        asyncio.run(run_benchmark_matrix(args))
        return 0
    if args.command == "skill":
        return run_skill_benchmark(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
