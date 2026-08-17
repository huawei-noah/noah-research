# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

"""Tests for bash stdout/stderr consecutive block deduplication and traceback distillation."""

from pathlib import Path

from scienceflow.core.tools.bash_tool import _dedup_repeated_blocks, _distill_tracebacks


class TestDedupRepeatedBlocks:
    def test_single_line_repeated_three_times_collapsed(self) -> None:
        text = "warn\nwarn\nwarn"
        out = _dedup_repeated_blocks(text, min_repeat=3)
        assert "warn" in out
        assert out.count("warn") == 1
        assert "repeated 3 times total" in out

    def test_single_line_two_repeats_unchanged(self) -> None:
        text = "a\na"
        assert _dedup_repeated_blocks(text, min_repeat=3) == text

    def test_two_line_block_repeated_three_times_collapsed(self) -> None:
        text = "W\nC\n" * 3
        out = _dedup_repeated_blocks(text, min_repeat=3)
        assert out.startswith("W\nC\n")
        assert "2-line block repeated 3 times total" in out
        assert out.count("W") == 1
        assert out.count("C") == 1

    def test_no_repetition_unchanged(self) -> None:
        text = "one\ntwo\nthree"
        assert _dedup_repeated_blocks(text, min_repeat=3) == text

    def test_min_repeat_zero_disables(self) -> None:
        text = "x\nx\nx\nx"
        assert _dedup_repeated_blocks(text, min_repeat=0) == text

    def test_empty_string(self) -> None:
        assert _dedup_repeated_blocks("", min_repeat=3) == ""

    def test_dedup_then_trim_independent(self) -> None:
        """Unique lines are unchanged by dedup; _trim_output still applies afterward."""
        from scienceflow.core.tools.bash_tool import _trim_output

        text = "\n".join(f"unique-{i}" for i in range(400))
        deduped = _dedup_repeated_blocks(text, min_repeat=3)
        assert deduped == text
        trimmed = _trim_output(deduped, max_chars=200)
        assert len(trimmed) < len(deduped)
        assert "truncated" in trimmed


class TestDistillTracebacks:
    def test_collapses_consecutive_library_frames(self, tmp_path: Path) -> None:
        ws = tmp_path.resolve()
        sol = ws / "solution.py"
        sk_base = "/fake/.venv/lib/python3.12/site-packages/sklearn/base.py"
        sk_val = "/fake/.venv/lib/python3.12/site-packages/sklearn/utils/validation.py"
        tb = (
            "Traceback (most recent call last):\n"
            f'  File "{sol}", line 206, in <module>\n'
            "    main()\n"
            f'  File "{sol}", line 141, in train_model\n'
            "    model.fit(X_train, y_train)\n"
            f'  File "{sk_base}", line 10, in wrapper\n'
            "    return fit_method(estimator, *args, **kwargs)\n"
            f'  File "{sk_val}", line 20, in check\n'
            "    raise ValueError(msg_err)\n"
            "ValueError: Input y contains NaN.\n"
        )
        out = _distill_tracebacks(tb, ws)
        assert "library frames collapsed" in out
        assert "sklearn.base -> sklearn.utils.validation" in out
        assert 'File "solution.py"' in out
        assert str(sol) not in out
        assert sk_base not in out
        assert "ValueError: Input y contains NaN." in out

    def test_all_user_frames_no_collapse_line(self, tmp_path: Path) -> None:
        ws = tmp_path.resolve()
        a = ws / "a.py"
        b = ws / "b.py"
        tb = (
            "Traceback (most recent call last):\n"
            f'  File "{a}", line 1, in <module>\n'
            "    x()\n"
            f'  File "{b}", line 2, in x\n'
            "    1 / 0\n"
            "ZeroDivisionError: division by zero\n"
        )
        out = _distill_tracebacks(tb, ws)
        assert "library frames collapsed" not in out
        assert 'File "a.py"' in out
        assert 'File "b.py"' in out
        assert str(a) not in out
        assert "ZeroDivisionError" in out

    def test_non_traceback_text_unchanged(self, tmp_path: Path) -> None:
        ws = tmp_path.resolve()
        text = "hello\nworld\n"
        assert _distill_tracebacks(text, ws) == text

    def test_enabled_false_returns_original(self, tmp_path: Path) -> None:
        ws = tmp_path.resolve()
        sol = ws / "solution.py"
        sk = "/x/site-packages/sklearn/base.py"
        tb = (
            "Traceback (most recent call last):\n"
            f'  File "{sol}", line 1, in <module>\n'
            "    x\n"
            f'  File "{sk}", line 1, in y\n'
            "    z\n"
            "RuntimeError: boom\n"
        )
        assert _distill_tracebacks(tb, ws, enabled=False) == tb

    def test_two_separate_tracebacks(self, tmp_path: Path) -> None:
        ws = tmp_path.resolve()
        sol = ws / "t.py"
        sk = "/z/site-packages/pkg/mod.py"
        text = (
            "header line\n"
            "Traceback (most recent call last):\n"
            f'  File "{sol}", line 1, in <module>\n'
            "    a\n"
            f'  File "{sk}", line 9, in inner\n'
            "    b\n"
            "ValueError: first\n"
            "---\n"
            "Traceback (most recent call last):\n"
            f'  File "{sol}", line 2, in <module>\n'
            "    c\n"
            f'  File "{sk}", line 3, in x\n'
            "    d\n"
            "RuntimeError: second\n"
        )
        out = _distill_tracebacks(text, ws)
        assert out.count("library frames collapsed") == 2
        assert "ValueError: first" in out
        assert "RuntimeError: second" in out
        assert "header line" in out
        assert 'File "t.py"' in out
        assert str(sol) not in out

    def test_user_frame_path_under_subdir_is_relative(self, tmp_path: Path) -> None:
        ws = tmp_path.resolve()
        sub = ws / "pkg" / "mod.py"
        sub.parent.mkdir(parents=True)
        tb = (
            "Traceback (most recent call last):\n"
            f'  File "{sub}", line 1, in <module>\n'
            "    x\n"
            "ValueError: x\n"
        )
        out = _distill_tracebacks(tb, ws)
        assert 'File "pkg/mod.py"' in out
        assert str(sub) not in out
