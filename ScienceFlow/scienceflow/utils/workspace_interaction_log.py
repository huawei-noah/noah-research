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

"""Append-only interaction logs under ``<workspace_dir>/.logs/``.

- ``interaction.log`` — this node only (same as historical behavior). LLM stream chunks
  may be appended as raw text (no timestamp prefix) between formatted lines; see
  :func:`write_raw_to_interaction_log`.
- ``traj_interaction.log`` — same mirror lines, appended after any content seeded by
  :meth:`NodeWorkspaceManager._init_traj_log` (ancestor history for improve nodes).
- For new long-horizon REPL runs, callers may request ``layout="split"`` so the same
  two logs are written under ``.logs/interaction/`` and ``.logs/traj_interaction/``.

Optional ANSI colors (``scienceflow_interaction_log_color`` / ``SCIENCEFLOW_INTERACTION_LOG_COLOR``)
are intended for terminal viewers (``tail -f``, ``less -R``). Editors often show raw
escape sequences; set ``scienceflow_interaction_log_color: false`` (or env off) to disable.
File logs do not honor ``NO_COLOR`` so IDE-injected ``NO_COLOR=1`` still allows colors.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from contextvars import ContextVar, Token
from pathlib import Path

from scienceflow.utils.node_paths import legacy_node_logs_dir, node_logs_dir

_interaction_log_ctx_tag: ContextVar[str] = ContextVar("interaction_log_ctx_tag", default="")


def build_interaction_log_context_tag(
    session: str | None,
    phase: str | None,
) -> str:
    """Human-readable prefix for ``interaction.log`` lines, e.g. ``[improve|explore]``."""
    s = (session or "draft").strip().lower() or "draft"
    p = (phase or "").strip().lower() or "?"
    return f"[{s}|{p}]"


def set_interaction_log_context_tag(tag: str) -> Token[str]:
    """Bind LNR session+phase for workspace interaction log lines (ContextVar)."""
    return _interaction_log_ctx_tag.set(tag)


def reset_interaction_log_context_tag(token: Token[str]) -> None:
    """Restore previous context-tag binding (pair with :func:`set_interaction_log_context_tag`)."""
    _interaction_log_ctx_tag.reset(token)


class _InteractionContextFilter(logging.Filter):
    """Inject ``nlhf_ctx`` (trailing space when non-empty) from the context var."""

    def filter(self, record: logging.LogRecord) -> bool:
        tag = _interaction_log_ctx_tag.get()
        record.nlhf_ctx = f"{tag} " if tag else ""
        return True

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_interaction_ansi(text: str) -> str:
    """Remove ANSI SGR sequences (e.g. ``\\033[2m``) from *text*."""
    return _ANSI_ESCAPE_RE.sub("", text)


def collapse_consecutive_repeated_lines_in_text(
    text: str,
    *,
    min_repeat: int = 3,
    max_preserve_first: int = 1,
    normalize: Callable[[str], str] | None = None,
    summary_prefix: str = "[log-dedup]",
) -> str:
    """Collapse runs of *consecutive* duplicate lines (after optional *normalize*).

    - Blank lines (after ``str.strip()``) break runs and are never deduplicated.
    - When a run has fewer than *min_repeat* lines, all lines are kept verbatim.
    - Otherwise the first *max_preserve_first* original lines are kept, then one summary line.
    """
    if not text:
        return text
    min_rep = max(2, int(min_repeat))
    preserve = max(1, int(max_preserve_first))

    def norm_key(line: str) -> str | None:
        if not line.strip():
            return None
        if normalize is not None:
            return normalize(line)
        return strip_interaction_ansi(line).rstrip("\r\n")

    lines = text.splitlines()
    out: list[str] = []
    buf: list[str] = []
    cur_key: str | None = None

    def flush_buf() -> None:
        nonlocal buf, cur_key
        if not buf:
            cur_key = None
            return
        if len(buf) < min_rep:
            out.extend(buf)
        else:
            n_dup = len(buf) - preserve
            out.extend(buf[:preserve])
            out.append(
                f"... {summary_prefix} omitted "
                f"{n_dup} duplicate consecutive line(s)",
            )
        buf = []
        cur_key = None

    for line in lines:
        k = norm_key(line)
        if k is None:
            flush_buf()
            out.append(line)
            continue
        if cur_key is not None and k != cur_key:
            flush_buf()
        if not buf:
            cur_key = k
        buf.append(line)
    flush_buf()
    return "\n".join(out)


class ConsecutiveLineDeduper:
    """Streaming cousin of :func:`collapse_consecutive_repeated_lines_in_text`.

    Call :meth:`flush` when the stream ends so a trailing run still collapses correctly.
    """

    def __init__(
        self,
        *,
        min_repeat: int = 3,
        max_preserve_first: int = 1,
        normalize: Callable[[str], str] | None = None,
        summary_prefix: str = "[log-dedup]",
    ) -> None:
        self._min_repeat = max(2, int(min_repeat))
        self._max_preserve_first = max(1, int(max_preserve_first))
        self._normalize = normalize
        self._summary_prefix = summary_prefix
        self._buf: list[str] = []
        self._key: str | None = None

    def _norm_key(self, line: str) -> str | None:
        if not line.strip():
            return None
        if self._normalize is not None:
            return self._normalize(line)
        return strip_interaction_ansi(line).rstrip("\r\n")

    def _flush_buf(self) -> list[str]:
        if not self._buf:
            self._key = None
            return []
        buf = self._buf
        self._buf = []
        self._key = None
        if len(buf) < self._min_repeat:
            return list(buf)
        n_dup = len(buf) - self._max_preserve_first
        summary = (
            f"... {self._summary_prefix} omitted "
            f"{n_dup} duplicate consecutive line(s)"
        )
        return buf[: self._max_preserve_first] + [summary]

    def feed_line(self, line: str) -> list[str]:
        """Feed one logical line (no trailing ``\\n`` required). Returns lines to emit now."""
        line = line.rstrip("\r\n")
        k = self._norm_key(line)
        if k is None:
            out = self._flush_buf()
            out.append(line)
            return out
        out: list[str] = []
        if self._key is not None and k != self._key:
            out.extend(self._flush_buf())
        if not self._buf:
            self._key = k
        self._buf.append(line)
        return out

    def flush(self) -> list[str]:
        """Flush a trailing run after the subprocess stream closes."""
        return self._flush_buf()


def normalize_lightgbm_warning_line(line: str) -> str | None:
    """Return a canonical key for a LightGBM warning line, or ``None`` if not a match.

    Used to collapse consecutive duplicate ``[LightGBM] [Warning]`` lines in
    ``[full-run-stream]`` mirrors without depending on stream prefixes or colors.
    """
    raw = strip_interaction_ansi(line)
    if "[LightGBM] [Warning]" not in raw:
        return None
    idx = raw.find("[LightGBM] [Warning]")
    return raw[idx:].strip()


def collapse_consecutive_lightgbm_warnings_in_text(text: str) -> str:
    """Collapse consecutive duplicate ``[LightGBM] [Warning]`` lines in a multi-line string.

    Used when appending full-run ``=== stdout ===`` / ``=== stderr ===`` blocks so they
    stay consistent with the streamlined ``[full-run-stream]`` output (same dedup rule).
    """
    if not text:
        return ""
    lines = text.splitlines()
    out: list[str] = []
    prev_norm: str | None = None
    pending = 0

    def flush_pending() -> None:
        nonlocal pending
        if pending > 0:
            out.append(
                "... [log-dedup] omitted "
                f"{pending} duplicate consecutive line(s) (same LightGBM warning)"
            )
            pending = 0

    for line in lines:
        norm = normalize_lightgbm_warning_line(line)
        if norm is None:
            flush_pending()
            prev_norm = None
            out.append(line)
            continue
        if prev_norm is not None and norm == prev_norm:
            pending += 1
            continue
        flush_pending()
        out.append(line)
        prev_norm = norm
    flush_pending()
    return "\n".join(out)


class FullRunLightGBMStreamDeduper:
    """Collapse consecutive identical LightGBM warning lines for full-run stream logging.

    Call :meth:`flush` after the subprocess stream ends so a trailing run of
    duplicates still emits one summary line.
    """

    def __init__(
        self,
        emit: Callable[[str], None],
        *,
        format_line: Callable[[str, str], str],
    ) -> None:
        self._emit = emit
        self._format_line = format_line
        self._prev_norm: str | None = None
        self._pending = 0

    def __call__(self, stream: str, line: str) -> None:
        line = line.rstrip("\n\r")
        norm = normalize_lightgbm_warning_line(line)
        if norm is None:
            self._flush_pending()
            self._prev_norm = None
            self._emit(self._format_line(stream, line))
            return
        if self._prev_norm is not None and norm == self._prev_norm:
            self._pending += 1
            return
        self._flush_pending()
        self._emit(self._format_line(stream, line))
        self._prev_norm = norm

    def _flush_pending(self) -> None:
        if self._pending > 0:
            self._emit(
                "... [log-dedup] omitted "
                f"{self._pending} duplicate consecutive line(s) (same LightGBM warning)"
            )
            self._pending = 0

    def flush(self) -> None:
        """Emit summary for trailing duplicate warnings (call after stream closes)."""
        self._flush_pending()
        # Next subprocess / next log segment must not inherit continuity from the prior stream.
        self._prev_norm = None


class BashStreamFilter:
    """Filter bash stream chunks to training-progress lines for interaction logs."""

    _WHITE_RE = re.compile(
        r"(?i)(\[(?:epoch|fold)\]|epoch\s*\d|training(?:\s+model)?|validation|val[_ ]|"
        r"\br2\b|score|metric|rmse|mae|loss|accuracy|fold\s*\d|loading data|"
        r"\bshape\b|saved|submission|traceback|error|exception|final|exit=)",
    )
    _BLACK_RE = re.compile(
        r"(?i)(\[warning\]|deprecationwarning|futurewarning|userwarning|"
        r"convergencewarning|\[lightgbm\]\s*\[warning\]|warnings\.filterwarnings)",
    )

    def __init__(
        self,
        emit: Callable[[str], None],
        *,
        prefix: str = "[bash-stream] ",
        dedup_enabled: bool = True,
        dedup_min_repeat: int = 3,
        dedup_summary_prefix: str = "[log-dedup]",
        dedup_normalize: Callable[[str], str] | None = None,
    ) -> None:
        self._emit = emit
        self._prefix = prefix
        self._buf = ""
        self.lines_total = 0
        self.lines_emitted = 0
        self._line_deduper: ConsecutiveLineDeduper | None = None
        if dedup_enabled:
            self._line_deduper = ConsecutiveLineDeduper(
                min_repeat=int(dedup_min_repeat),
                summary_prefix=dedup_summary_prefix,
                normalize=dedup_normalize,
            )

    def _keep_line(self, line: str) -> bool:
        s = strip_interaction_ansi(line).strip()
        if not s:
            return False
        if self._BLACK_RE.search(s):
            return False
        if s.startswith("\r") and ("epoch" not in s.lower()):
            return False
        return self._WHITE_RE.search(s) is not None

    def feed_chunk(self, chunk: str) -> None:
        """Consume a subprocess chunk that may contain partial lines."""
        if not chunk:
            return
        self._buf += chunk
        parts = self._buf.split("\n")
        self._buf = parts.pop() if parts else ""
        for line in parts:
            self.lines_total += 1
            if self._keep_line(line):
                stripped = line.rstrip("\r\n")
                if self._line_deduper is not None:
                    for piece in self._line_deduper.feed_line(stripped):
                        self.lines_emitted += 1
                        self._emit(f"{self._prefix}{piece}\n")
                else:
                    self.lines_emitted += 1
                    self._emit(f"{self._prefix}{stripped}\n")

    def flush(self) -> None:
        """Flush trailing partial line (if any) through the same filter."""
        if not self._buf:
            if self._line_deduper is not None:
                for piece in self._line_deduper.flush():
                    self.lines_emitted += 1
                    self._emit(f"{self._prefix}{piece}\n")
            return
        line = self._buf
        self._buf = ""
        self.lines_total += 1
        if self._keep_line(line):
            stripped = line.rstrip("\r\n")
            if self._line_deduper is not None:
                for piece in self._line_deduper.feed_line(stripped):
                    self.lines_emitted += 1
                    self._emit(f"{self._prefix}{piece}\n")
                for piece in self._line_deduper.flush():
                    self.lines_emitted += 1
                    self._emit(f"{self._prefix}{piece}\n")
            else:
                self.lines_emitted += 1
                self._emit(f"{self._prefix}{stripped}\n")
        elif self._line_deduper is not None:
            for piece in self._line_deduper.flush():
                self.lines_emitted += 1
                self._emit(f"{self._prefix}{piece}\n")

    def summary(self) -> str:
        return f"streamed {self.lines_emitted}/{self.lines_total} lines to interaction.log"

_loggers: dict[str, logging.Logger] = {}

_RESET = "\033[0m"


def write_raw_to_interaction_log(lg: logging.Logger, text: str) -> None:
    """Append *text* to every ``FileHandler`` on *lg* without formatter/timestamp, then flush.

    Used to mirror LLM streaming chunks into ``interaction.log`` / ``traj_interaction.log``
    so viewers see tokens as they arrive. Must not raise.
    """
    if not text:
        return
    try:
        for handler in lg.handlers:
            stream = getattr(handler, "stream", None)
            if isinstance(handler, logging.FileHandler) and stream is not None:
                stream.write(text)
                stream.flush()
    except OSError:
        pass
_DIM = "\033[2m"


def colorize_interaction_message(msg: str, *, enabled: bool = True) -> str:
    """Add ANSI colors to a single log *message* (not the timestamp line).

    When *enabled* is false, returns *msg* unchanged. Does not read ``NO_COLOR`` — file
    output is opt-in via config; terminal ``NO_COLOR`` (e.g. from Cursor) must not
    strip colors from ``interaction.log``.
    """
    if not enabled:
        return msg
    lines = msg.split("\n")
    if not lines:
        return msg
    first = lines[0]
    head = _head_style_for_first_line(first)
    if head is None:
        return msg
    out: list[str] = [head + first + _RESET]
    for line in lines[1:]:
        out.append(_DIM + line + _RESET)
    return "\n".join(out)


def _head_style_for_first_line(first: str) -> str | None:
    """Return ANSI SGR prefix for the first line, or None to leave the whole message plain."""
    if first.startswith("[user]"):
        return "\033[96m"  # bright cyan
    if first.startswith("[assistant-thinking]"):
        return "\033[35m"  # magenta
    if first.startswith("[thought]"):
        return "\033[34m"  # blue
    if first.startswith("[tool-call]"):
        return "\033[32m"  # green
    if first.startswith("[tool-call-body]"):
        return "\033[33m"  # yellow
    if first.startswith("[tool-result]"):
        if "exit=err" in first:
            return "\033[31m"  # red
        return "\033[32m"  # green
    if "====== Agent Iteration" in first or first.startswith("[repl-run"):
        return "\033[1;33m"  # bold yellow
    return None


class InteractionPlainFormatter(logging.Formatter):
    """Timestamp + optional ``[session|phase]`` prefix + message."""

    def __init__(self, *, datefmt: str = "%Y-%m-%d %H:%M:%S") -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(nlhf_ctx)s%(message)s",
            datefmt=datefmt,
        )

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "nlhf_ctx"):
            record.nlhf_ctx = ""
        return super().format(record)


def _make_interaction_formatter() -> logging.Formatter:
    return InteractionPlainFormatter()


class InteractionColorFormatter(logging.Formatter):
    """Format like the plain interaction formatter but colorize ``%(message)s``."""

    def __init__(self, *, datefmt: str = "%Y-%m-%d %H:%M:%S") -> None:
        super().__init__(datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "nlhf_ctx"):
            record.nlhf_ctx = ""
        asctime = self.formatTime(record, self.datefmt)
        levelname = f"{record.levelname:<8}"
        raw = record.getMessage()
        msg = colorize_interaction_message(raw, enabled=True)
        return f"{asctime} | {levelname} | {record.nlhf_ctx}{msg}"


def _normalise_layout(layout: str | None) -> str:
    val = str(layout or "flat").strip().lower()
    return "split" if val in {"split", "lhr_split"} else "flat"


def _logger_cache_key(
    root: Path,
    *,
    color: bool,
    layout: str,
    log_dir: Path | None = None,
) -> str:
    extra = f"|log_dir={log_dir}" if log_dir is not None else ""
    return f"{root}|color={int(bool(color))}|layout={_normalise_layout(layout)}{extra}"


def close_workspace_interaction_logger(
    workspace_dir: str | Path,
    *,
    color: bool = False,
    layout: str = "flat",
    log_dir_override: str | Path | None = None,
) -> None:
    try:
        root = Path(workspace_dir).resolve()
        log_dir = Path(log_dir_override).resolve(strict=False) if log_dir_override is not None else None
    except OSError:
        return
    key = _logger_cache_key(root, color=bool(color), layout=layout, log_dir=log_dir)
    lg = _loggers.pop(key, None)
    if lg is None:
        return
    for handler in list(lg.handlers):
        try:
            handler.flush()
            handler.close()
        except OSError:
            pass
        lg.removeHandler(handler)


def attach_workspace_interaction_logger(
    workspace_dir: str | Path,
    *,
    color: bool = False,
    layout: str = "flat",
    log_dir_override: str | Path | None = None,
) -> logging.Logger | None:
    """Return a logger that appends to interaction and trajectory logs.

    Writes the same records to:

    - ``workspace_dir/.logs/interaction.log`` — per-node session only (fresh file each node).
    - ``workspace_dir/.logs/traj_interaction.log`` — continues after optional pre-seed from
      parent (full chain for improve nodes).

    With ``layout="split"``, the files move to ``.logs/interaction/interaction.log``
    and ``.logs/traj_interaction/traj_interaction.log``.

    One logger instance per (resolved workspace path, *color* flag, *layout*,
    optional direct log dir) in the current process. ``log_dir_override`` is for
    controller-owned LNR logs that must live outside the agent workspace.

    When *color* is true, messages are wrapped with ANSI escapes.
    """
    try:
        root = Path(workspace_dir).resolve()
        direct_log_dir = (
            Path(log_dir_override).resolve(strict=False)
            if log_dir_override is not None
            else None
        )
    except OSError:
        return None
    layout_norm = _normalise_layout(layout)
    key = _logger_cache_key(root, color=bool(color), layout=layout_norm, log_dir=direct_log_dir)
    if key in _loggers:
        return _loggers[key]

    log_dir = direct_log_dir if direct_log_dir is not None else node_logs_dir(root, create=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt: logging.Formatter = (
        InteractionColorFormatter() if color else _make_interaction_formatter()
    )
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    lg = logging.getLogger(f"scienceflow.workspace.interaction.{h}")
    lg.setLevel(logging.DEBUG)
    if not any(isinstance(f, _InteractionContextFilter) for f in lg.filters):
        lg.addFilter(_InteractionContextFilter())

    if layout_norm == "split":
        inter_dir = log_dir / "interaction"
        traj_dir = log_dir / "traj_interaction"
        inter_dir.mkdir(parents=True, exist_ok=True)
        traj_dir.mkdir(parents=True, exist_ok=True)
        inter_path = inter_dir / "interaction.log"
        traj_path = traj_dir / "traj_interaction.log"
    else:
        inter_path = log_dir / "interaction.log"
        traj_path = log_dir / "traj_interaction.log"

    fh_inter = logging.FileHandler(inter_path, mode="a", encoding="utf-8")
    fh_inter.setLevel(logging.DEBUG)
    fh_inter.setFormatter(fmt)
    lg.addHandler(fh_inter)

    fh_traj = logging.FileHandler(traj_path, mode="a", encoding="utf-8")
    fh_traj.setLevel(logging.DEBUG)
    fh_traj.setFormatter(fmt)
    lg.addHandler(fh_traj)

    legacy_log_dir = legacy_node_logs_dir(root)
    if (
        direct_log_dir is None
        and layout_norm == "flat"
        and legacy_log_dir.exists()
        and legacy_log_dir != log_dir
    ):
        try:
            fh_legacy_inter = logging.FileHandler(
                legacy_log_dir / "interaction.log",
                mode="a",
                encoding="utf-8",
            )
            fh_legacy_inter.setLevel(logging.DEBUG)
            fh_legacy_inter.setFormatter(fmt)
            lg.addHandler(fh_legacy_inter)

            fh_legacy_traj = logging.FileHandler(
                legacy_log_dir / "traj_interaction.log",
                mode="a",
                encoding="utf-8",
            )
            fh_legacy_traj.setLevel(logging.DEBUG)
            fh_legacy_traj.setFormatter(fmt)
            lg.addHandler(fh_legacy_traj)
        except OSError:
            logger = logging.getLogger("scienceflow")
            logger.debug("legacy interaction log mirror unavailable", exc_info=True)

    lg.propagate = False
    _loggers[key] = lg
    return lg
