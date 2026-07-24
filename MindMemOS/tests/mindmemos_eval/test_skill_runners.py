"""Tests for shared skill benchmark runners."""

from __future__ import annotations

import argparse

from mindmemos_eval.skills import runners


def test_skill_runner_dispatches_selected_env(monkeypatch) -> None:
    calls: list[str] = []

    def _run(args: argparse.Namespace) -> int:
        calls.append(args.env)
        return 7

    monkeypatch.setitem(runners.SKILL_RUNNERS, "toy", (lambda _parser: None, _run))

    assert runners.run_skill_benchmark(argparse.Namespace(env="toy")) == 7
    assert calls == ["toy"]
