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

"""Rich-based terminal UI for ScienceFlow REPL.

Provides turn-based rendering: each agent turn (thinking + execution)
is printed as a compact block after the turn completes, avoiding
Rich Live ANSI-cursor issues that caused intermediate Panels to vanish.
"""

from __future__ import annotations

import re
import sys

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

_PREVIEW_LINES = 10
_MAX_CMD_DISPLAY_CHARS = 96
_LINE_MAX = 100
_CODE_PREVIEW_LINES = 5


def truncate_middle(text: str, max_len: int = _MAX_CMD_DISPLAY_CHARS) -> str:
    """Keep head + tail, replace middle with ``...``."""
    s = text.replace("\n", " ↵ ").strip()
    if len(s) <= max_len:
        return s
    if max_len <= 5:
        return s[:max_len]
    inner = max_len - 3
    head = inner // 2
    tail = inner - head
    return f"{s[:head]}...{s[-tail:]}"


def shorten_long_lines(text: str, max_len: int = _LINE_MAX) -> str:
    """Per-line middle-truncation for terminal display only."""
    if not text or max_len < 8:
        return text
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        out.append(truncate_middle(line, max_len) if len(line) > max_len else line)
    return "\n".join(out)


class RichUI:
    """Terminal UI backed by the *rich* library.

    Step panels use ``console.print`` only. The LLM-wait spinner uses
    ``Live`` with ``redirect_stdout=False`` so stdout is not proxied
    (avoids batched / delayed output in some terminals).
    """

    def __init__(self, *, show_code: int = _CODE_PREVIEW_LINES) -> None:
        self.console = Console()
        self.show_code = show_code

    def _line_budget(self) -> int:
        w = self.console.width
        if w and w > 24:
            return max(48, min(_LINE_MAX, w - 4))
        return _LINE_MAX

    def _flush(self) -> None:
        """Push buffered Rich and stdio output to the terminal."""
        try:
            self.console.file.flush()
        except (AttributeError, OSError):
            pass
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, OSError):
                pass

    def thinking_spinner(self, input_tokens: int = 0):
        """Spinner while the LLM thinks (no stdout/stderr redirection)."""
        if input_tokens:
            if input_tokens >= 1000:
                qty = f"{input_tokens / 1000:.1f}k tok"
            else:
                qty = f"{input_tokens} tok"
            label = f"Thinking… (input: {qty})"
        else:
            label = "Thinking…"
        sp = Spinner(
            "dots",
            text=f"[dim]{label}[/dim]",
            style="status.spinner",
            speed=1.0,
        )
        return Live(
            sp,
            console=self.console,
            transient=True,
            refresh_per_second=12.5,
            redirect_stdout=False,
            redirect_stderr=False,
        )

    # ------------------------------------------------------------------
    # Code block preview
    # ------------------------------------------------------------------

    def _render_code_block(self, code: str, lang: str) -> None:
        """Render code preview between Rule and Panel.

        ``self.show_code`` controls how many lines:
        * 0  → hidden (only subtitle)
        * -1 → full code
        * N  → first N lines + "... (M more)"
        """
        if self.show_code == 0 or not code.strip():
            return

        lines = code.strip().splitlines()
        total = len(lines)

        if self.show_code < 0 or total <= self.show_code:
            preview = code.strip()
            footer = ""
        else:
            preview = "\n".join(lines[: self.show_code])
            hidden = total - self.show_code
            footer = f"[dim]  ... ({total} lines, {hidden} hidden)[/dim]"

        syn_lang = "python" if lang == "python" else "bash"
        self.console.print(
            Syntax(
                preview,
                syn_lang,
                theme="monokai",
                line_numbers=False,
                word_wrap=False,
                padding=(0, 1),
            )
        )
        if footer:
            self.console.print(footer)

    # ------------------------------------------------------------------
    # --stream: execution-only step (LLM was already streamed to stdout)
    # ------------------------------------------------------------------

    def render_exec_step(
        self,
        *,
        step: int,
        lang: str,
        code: str,
        output: str,
        returncode: int = 0,
        exec_time: float = 0.0,
    ) -> None:
        """Render execution result after the LLM response was streamed."""
        budget = self._line_budget()

        self.console.print(
            Rule(f"[bold]Step {step}[/bold]", style="dim")
        )
        self._render_code_block(code, lang)

        self._render_exec_panel(
            output=output,
            lang=lang,
            returncode=returncode,
            exec_time=exec_time,
            command=code,
            budget=budget,
        )
        self.console.print()
        self._flush()

    # ------------------------------------------------------------------
    # --no-stream: two-phase rendering (header before exec, result after)
    # ------------------------------------------------------------------

    def render_step_header(
        self,
        *,
        step: int,
        thinking: str,
        lang: str,
        code: str,
    ) -> None:
        """Render Step rule + thinking + code preview (before execution)."""
        self.console.print(
            Rule(f"[bold]Step {step}[/bold]", style="dim")
        )
        if thinking:
            self._render_markdown_enhanced(thinking)
            self.console.print()
        self._render_code_block(code, lang)
        self._flush()

    def render_step_result(
        self,
        *,
        lang: str,
        code: str,
        output: str,
        returncode: int = 0,
        exec_time: float = 0.0,
    ) -> None:
        """Render execution result panel (after execution)."""
        self._render_exec_panel(
            output=output,
            lang=lang,
            returncode=returncode,
            exec_time=exec_time,
            command=code,
            budget=self._line_budget(),
        )
        self.console.print()
        self._flush()

    def render_turn(
        self,
        *,
        step: int,
        thinking: str,
        lang: str,
        code: str,
        output: str,
        returncode: int = 0,
        exec_time: float = 0.0,
    ) -> None:
        """Render a full turn: thinking text + execution Panel."""
        self.render_step_header(
            step=step, thinking=thinking, lang=lang, code=code,
        )
        self.render_step_result(
            lang=lang, code=code, output=output,
            returncode=returncode, exec_time=exec_time,
        )

    # ------------------------------------------------------------------
    # Execution result panel (internal)
    # ------------------------------------------------------------------

    def _render_exec_panel(
        self,
        output: str,
        lang: str,
        returncode: int,
        exec_time: float,
        command: str,
        budget: int,
    ) -> None:
        ok = returncode == 0
        border = "green" if ok else "red"
        icon = "✓" if ok else "✗"
        title = f"{icon} {lang}  exit={returncode}  {exec_time:.1f}s"
        subtitle = truncate_middle(command.strip(), _MAX_CMD_DISPLAY_CHARS) if command else None

        stripped = output.strip()
        if not stripped:
            body = (
                f"[dim](executed, no text output; exit={returncode}"
                f"{', success' if ok else ', failed'})[/dim]"
            )
        else:
            shortened = shorten_long_lines(stripped, budget)
            lines = shortened.splitlines()
            total_raw = stripped.count("\n") + 1
            if len(lines) <= _PREVIEW_LINES:
                body = "\n".join(lines)
            else:
                preview = "\n".join(lines[:_PREVIEW_LINES])
                rest = total_raw - _PREVIEW_LINES
                nchars = len(stripped)
                body = (
                    f"{preview}\n"
                    f"[dim]... ({total_raw} lines / {nchars} chars total; "
                    f"panel shows only the first {_PREVIEW_LINES} lines, {rest} lines omitted)[/dim]"
                )

        self.console.print(
            Panel(
                body,
                title=title,
                subtitle=subtitle,
                title_align="left",
                border_style=border,
                expand=True,
                padding=(0, 1),
            )
        )

    # Keep public alias for backward compat (used by cli `run` command)
    def render_execution_result(
        self,
        output: str,
        lang: str = "bash",
        returncode: int = 0,
        exec_time: float = 0.0,
        command: str | None = None,
    ) -> None:
        self._render_exec_panel(
            output=output,
            lang=lang,
            returncode=returncode,
            exec_time=exec_time,
            command=command or "",
            budget=self._line_budget(),
        )

    # ------------------------------------------------------------------
    # Agent final reply (no code execution)
    # ------------------------------------------------------------------

    _CODE_BLOCK_RE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    _KNOWN_LANGS = frozenset({
        "python", "bash", "sh", "json", "yaml", "sql",
        "javascript", "typescript", "html", "css", "text",
    })

    def _render_markdown_enhanced(self, text: str) -> None:
        """Render text with code blocks highlighted in Panels."""
        budget = self._line_budget()
        parts: list[tuple] = []
        last_end = 0

        for m in self._CODE_BLOCK_RE.finditer(text):
            pre = text[last_end : m.start()].strip()
            if pre:
                parts.append(("md", pre))
            lang = m.group(1) or "text"
            code = m.group(2).strip()
            parts.append(("code", lang, code))
            last_end = m.end()

        tail = text[last_end:].strip()
        if tail:
            parts.append(("md", tail))
        if not parts:
            parts.append(("md", text))

        for part in parts:
            if part[0] == "code":
                lang, code = part[1], part[2]
                syn_lang = lang if lang in self._KNOWN_LANGS else "text"
                self.console.print(
                    Panel(
                        Syntax(
                            code, syn_lang, theme="monokai",
                            line_numbers=False, word_wrap=False,
                            padding=(0, 1),
                        ),
                        title=f"[dim]{lang}[/dim]",
                        title_align="left",
                        border_style="dim",
                        expand=True,
                        padding=(0, 0),
                    )
                )
            else:
                self.console.print(
                    Markdown(shorten_long_lines(part[1], budget))
                )

    def render_agent_reply(self, text: str) -> None:
        """Render agent final reply with enhanced code block highlighting."""
        self.console.print()
        self._render_markdown_enhanced(text)
        self.console.print()
        self._flush()

    # ------------------------------------------------------------------
    # Input prompt
    # ------------------------------------------------------------------

    def prompt_input(self) -> str | None:
        try:
            return self.console.input("[bold cyan]scienceflow>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            return None

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def welcome(self) -> None:
        self.console.print()
        self.console.print(
            Rule("[bold]ScienceFlow REPL[/bold]", style="cyan")
        )
        self.console.print(
            "[dim]Type [bold]exit[/bold], [bold]quit[/bold], or [bold]exit()[/bold] to leave.[/dim]"
        )
        self.console.print()
        self._flush()

    def goodbye(self) -> None:
        self.console.print()
        self.console.print("[dim]Bye.[/dim]")
        self._flush()
