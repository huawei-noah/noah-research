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

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from pathlib import Path

_HOST_ABSOLUTE_PATH_IN_COMMAND_RE = re.compile(
    r"(?<![\w.])/(?:home|work|mnt|kaggle|tmp)(?:/[^\s\"'`<>;&|)]*)?",
)


_SILENT_REDIRECT_RE = re.compile(
    r"2>/dev/null|2>&1\s*>/dev/null|>\s*/dev/null\s*2>&1|&>\s*/dev/null"
)


def _has_silent_redirect(cmd: str) -> bool:
    return bool(_SILENT_REDIRECT_RE.search(cmd or ""))


def _has_unquoted_background_operator(command: str) -> bool:
    """Return True when shell text contains a real ``&`` background operator."""
    quote = ""
    escaped = False
    text = str(command or "")
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char != "&":
            continue
        prev_char = text[index - 1] if index > 0 else ""
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if prev_char in {"&", ">"} or next_char in {"&", ">"}:
            continue
        return True
    return False


def background_resource_command_blocked_error(command: str) -> str | None:
    """Block long-running resource commands that detach from the managed bash tree."""
    s = _strip_heredoc_bodies_for_process_control(command).strip()
    if not s or not _SLOW_ENTRYPOINT_RE.search(s):
        return None
    lowered = s.lower()
    detached = (
        _has_unquoted_background_operator(s)
        or re.search(r"(^|[;\n&|()]\s*)(nohup|setsid)\b", lowered) is not None
        or re.search(r"(^|[;\n&|()]\s*)disown\b", lowered) is not None
    )
    if not detached:
        return None
    return (
        "Blocked: background long-running resource commands are not allowed in managed bash. "
        "Run training, inference, and feature extraction in the foreground so Safety can "
        "track stdout, artifacts, resources, and stop/replan decisions. Write verbose logs to a "
        "workspace file and print a compact tail instead of using `&`, `nohup`, `setsid`, or `disown`."
    )


_HEREDOC_FILE_WRITE_RE = re.compile(r"(^|\n)\s*cat\s*>{1,2}\s*[^\n]+<<")
_HEREDOC_PYTHON_FILE_WRITE_RE = re.compile(
    r"(^|[;\n&|()]\s*)cat\s*>{1,2}\s*"
    r"(?P<path>(?:\x27[^\x27]+\.py\x27|\"[^\"]+\.py\"|[^\s\x27\"<>;&|]+\.py))\s*<<",
    re.IGNORECASE,
)
_PYTHON_SCRIPT_RUN_RE = re.compile(
    r"(^|[;\n&|()]\s*)"
    r"(?:(?:env\s+)?[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*"
    r"(?:uv\s+run\s+)?python(?:3)?(?:\s+-u)?\s+"
    r"(?P<path>(?:\x27[^\x27]+\.py\x27|\"[^\"]+\.py\"|[^\s;&|()\"\x27]+\.py))"
    r"(?=$|\s|[;&|)])",
    re.IGNORECASE,
)


def _normalize_shell_script_path(path: str) -> str:
    normalized = str(path or "").strip().strip("\"" + chr(39))
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return re.sub(r"/+", "/", normalized)


def _executes_heredoc_written_python_script(raw_command: str, stripped_command: str) -> bool:
    written_paths = {
        _normalize_shell_script_path(match.group("path"))
        for match in _HEREDOC_PYTHON_FILE_WRITE_RE.finditer(raw_command)
    }
    if not written_paths:
        return False
    for match in _PYTHON_SCRIPT_RUN_RE.finditer(stripped_command):
        if _normalize_shell_script_path(match.group("path")) in written_paths:
            return True
    return False


def mixed_file_write_execution_blocked_error(command: str) -> str | None:
    """Block heredoc file writes that execute the written script or launch long resource work.

    A write+run bundle hides a long foreground command behind a ``bash_kind=write``
    turn, which makes stage logs and resource intervention harder to interpret.
    """
    raw = str(command or "")
    if not _HEREDOC_FILE_WRITE_RE.search(raw):
        return None
    s = _strip_heredoc_bodies_for_process_control(raw).strip()
    if not s or not _HEREDOC_FILE_WRITE_RE.search(s):
        return None
    raw_lower = raw.lower()
    shell_launches_python = (
        re.search(
            r"(^|[;\n&|()]\s*)(?:(?:env\s+)?[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*python(?:3)?\b",
            s,
        )
        is not None
    )
    script_has_train_evidence = any(
        token in raw_lower
        for token in (
            "scienceflow_resource_intent=gpu_train",
            ".train(",
            ".fit(",
            "epochs=",
            "torchrun",
            "deepspeed",
            "accelerate",
        )
    )
    executes_written_script = _executes_heredoc_written_python_script(raw, s)
    launches_resource_work = bool(
        executes_written_script
        or _SLOW_ENTRYPOINT_RE.search(s)
        or _INLINE_RESOURCE_PYTHON_RE.search(s)
        or (shell_launches_python and script_has_train_evidence)
    )
    if not launches_resource_work:
        return None
    return (
        "Blocked: do not combine a heredoc file write with Python script execution or long training, inference, "
        "validation, feature-extraction, or submission command in the same bash call. "
        "Split it into two bash tool calls: first write/update the script, wait for that "
        "result, then run the script as a separate foreground bash command so Safety "
        "can classify logs, track resources, and stop/replan the execution if needed."
    )


_OUTPUT_TRUNCATING_PIPE_RE = re.compile(
    r"\|\s*(?:tail|head)\b",
    re.IGNORECASE,
)

_INLINE_RESOURCE_PYTHON_RE = re.compile(
    r"(^|[;\n&|()]\s*)(?:uv\s+run\s+)?python(?:3)?(?:\s+-u)?\s+-c\s+"
    r"(?=.*\b("
    r"SentenceTransformer|transformers|tensorflow|keras|torch|"
    r"model\.encode|predict_notebooks|validate|submission|"
    r"fit|epochs?|training"
    r")\b)",
    re.IGNORECASE | re.DOTALL,
)


def truncated_resource_output_blocked_error(command: str) -> str | None:
    """Block long resource work whose primary stdout is piped through tail/head.

    The resource monitor needs the full foreground stdout stream to infer cadence,
    artifacts, metric updates, and stop/replan state. Inspecting an existing log
    with ``tail -30 train.log`` is fine; this guard only matches commands that
    also launch a slow training/inference entrypoint or clearly long inline Python
    work in the same shell text.
    """
    s = _strip_heredoc_bodies_for_process_control(command).strip()
    if not s or not _OUTPUT_TRUNCATING_PIPE_RE.search(s):
        return None
    launches_resource_work = bool(
        _SLOW_ENTRYPOINT_RE.search(s)
        or _INLINE_RESOURCE_PYTHON_RE.search(s)
    )
    if not launches_resource_work:
        return None
    return (
        "Blocked: do not pipe long-running training, inference, or feature extraction "
        "directly to `tail`/`head`. Run the resource command in the foreground so the "
        "Safety can observe full stdout, metrics, artifacts, resource usage, and "
        "recoverable-stop markers. If output is verbose, make the script write a log "
        "file and print compact periodic summaries while it runs."
    )


_SLOW_ENTRYPOINT_RE = re.compile(
    r"(^|[;\n&|()]\s*)"
    r"(?:nohup\s+)?"
    r"(?:timeout\s+\S+\s+)?"
    r"(?:(?:env\s+)?[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*"
    r"(?:"
    r"(?:uv\s+run\s+)?python(?:3)?(?:\s+-u)?\s+"
    r"(?:\./)?(?:[^\s;&|()\"']*/)?"
    r"(?:run_)?(?:solution|train|training|fit|finetune|fine_tune|predict|prediction|infer|inference|submit|submission|ensemble|extract_features?|feature_extract|classifier|model|score|scoring|validate|validation|eval|evaluate|metric|metrics)"
    r"[\w.-]*\.py\b"
    r"|(?:bash|sh)\s+(?:\./)?(?:[^\s;&|()\"']*/)?"
    r"(?:run_)?(?:train|training|fit|finetune|fine_tune|predict|prediction|infer|inference|submit|submission|ensemble|extract_features?|feature_extract|classifier|model|score|scoring|validate|validation|eval|evaluate|metric|metrics)"
    r"[\w.-]*\.sh\b"
    r"|(?:torchrun|accelerate|deepspeed)\b"
    r")",
    re.IGNORECASE,
)


_RM_RECURSIVE_RE = re.compile(
    r"(^|[;\n&|()]\s*)rm\s+(?=[^;\n|&]*?(?:--recursive\b|-[A-Za-z]*r[A-Za-z]*\b))",
    re.IGNORECASE,
)


def _split_shell_control_segments(command: str) -> list[str]:
    text = str(command or "")
    segments: list[str] = []
    quote = ""
    escaped = False
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if char == "\\":
            escaped = True
            i += 1
            continue
        if quote:
            if char == quote:
                quote = ""
            i += 1
            continue
        if char in {"'", '"'}:
            quote = char
            i += 1
            continue
        sep_len = 0
        if text.startswith("&&", i) or text.startswith("||", i):
            sep_len = 2
        elif char in {";", "\n"}:
            sep_len = 1
        if sep_len:
            segment = text[start:i].strip()
            if segment:
                segments.append(segment)
            i += sep_len
            start = i
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        segments.append(tail)
    return segments


def _segment_starts_with_rm(segment: str) -> bool:
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        tokens = str(segment or "").split()
    return bool(tokens and tokens[0] == "rm")


def _segment_launches_resource_work(segment: str) -> bool:
    return bool(_SLOW_ENTRYPOINT_RE.search(segment) or _INLINE_RESOURCE_PYTHON_RE.search(segment))


def dangerous_delete_command_blocked_error(command: str) -> str | None:
    """Block recursive rm and rm-prefixed long resource commands."""

    s = _strip_heredoc_bodies_for_process_control(command).strip()
    if not s:
        return None
    if _RM_RECURSIVE_RE.search(s):
        return (
            "Blocked: recursive rm is not allowed in managed bash. "
            "Use a narrow Python Path.unlink(missing_ok=True) for a known file, or create a unique output/checkpoint path."
        )
    segments = _split_shell_control_segments(s)
    for index, segment in enumerate(segments[:-1]):
        if not _segment_starts_with_rm(segment):
            continue
        if any(_segment_launches_resource_work(later) for later in segments[index + 1 :]):
            return (
                "Blocked: do not prefix long training, inference, feature, or validation commands with shell rm. "
                "Use a unique output/checkpoint path, or perform a narrow Path.unlink(missing_ok=True) inside the script for one known file."
            )
    return None


_GLOBAL_SCAN_ROOTS = frozenset({"/", "/home", "/work", "/mnt", "/kaggle", "/tmp"})
_WORKSPACE_SCOPE_ALLOWED_ABSOLUTE_PATHS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})
_WORKSPACE_SCOPE_TEXT_COMMANDS = frozenset({"echo", "printf"})
_WORKSPACE_SCOPE_ABSOLUTE_LITERAL_RE = re.compile(
    r"(?<![\w:/.-])"
    r"/(?:home|work|mnt|kaggle|tmp|etc|root|var|usr|bin|lib|lib64|proc|sys|opt|srv)"
    r"(?:/[^\s\"'`<>;&|)]*)?",
)
_WORKSPACE_SCOPE_REDIR_INLINE_RE = re.compile(r"^(?:\d*)?(?:>>?|<|<>|&>)(.+)$")
_WORKSPACE_SCOPE_REDIR_TOKEN_RE = re.compile(r"^(?:\d*)?(?:>>?|<|<>|&>)$")
_WORKSPACE_SCOPE_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=(.+)$")


def _is_global_scan_root(token: str) -> bool:
    text = str(token or "").strip().strip("'\"")
    if not text or text.startswith("-"):
        return False
    if text == "/*":
        return True
    if text.endswith("/*"):
        text = text[:-2]
    text = text.rstrip("/") or "/"
    return text in _GLOBAL_SCAN_ROOTS


def _has_short_or_long_flag(tokens: list[str], short: str, long_names: set[str]) -> bool:
    for token in tokens:
        if not token.startswith("-") or token == "-":
            continue
        if token in long_names:
            return True
        if not token.startswith("--") and short in token[1:]:
            return True
    return False


def _nested_shell_c_commands(tokens: list[str]) -> list[str]:
    commands: list[str] = []
    for index, token in enumerate(tokens[:-1]):
        if token not in {"bash", "sh"}:
            continue
        for pos in range(index + 1, len(tokens) - 1):
            opt = tokens[pos]
            if opt == "-c" or (opt.startswith("-") and not opt.startswith("--") and "c" in opt[1:]):
                commands.append(tokens[pos + 1])
                break
    return commands


def _split_pipeline_token_groups(tokens: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token == "|":
            if current:
                groups.append(current)
                current = []
            continue
        current.append(token)
    if current:
        groups.append(current)
    return groups


def _has_parent_dir_component(value: str) -> bool:
    text = str(value or "").strip().strip("'\"").replace("\\", "/")
    if not text or "://" in text or text.startswith("$"):
        return False
    parts = [part for part in text.split("/") if part not in {"", "."}]
    return any(part == ".." for part in parts)


def _workspace_scope_absolute_path_blocked(value: str) -> bool:
    text = str(value or "").strip().strip("'\"")
    if not text or "://" in text or text.startswith("$"):
        return False
    if text.endswith("/*"):
        text = text[:-2] or "/"
    text = text.rstrip("/") if text != "/" else text
    if text in _WORKSPACE_SCOPE_ALLOWED_ABSOLUTE_PATHS:
        return False
    return text.startswith("/")


def _workspace_scope_literal_violation(value: str) -> str | None:
    text = str(value or "")
    for match in _WORKSPACE_SCOPE_ABSOLUTE_LITERAL_RE.finditer(text):
        candidate = match.group(0)
        if candidate not in _WORKSPACE_SCOPE_ALLOWED_ABSOLUTE_PATHS:
            return candidate
    return None


def _workspace_scope_candidate_violation(value: str) -> str | None:
    text = str(value or "").strip().strip("'\"")
    if not text or text in {"-", "--"} or "://" in text:
        return None
    if _workspace_scope_absolute_path_blocked(text):
        return text
    if _has_parent_dir_component(text):
        return text
    return _workspace_scope_literal_violation(text)


def _workspace_scope_token_violation(
    token: str,
    *,
    command_name: str,
    redirection_target: bool = False,
) -> str | None:
    text = str(token or "").strip()
    if not text:
        return None
    inline_redir = _WORKSPACE_SCOPE_REDIR_INLINE_RE.match(text)
    if inline_redir:
        return _workspace_scope_candidate_violation(inline_redir.group(1))
    if command_name in _WORKSPACE_SCOPE_TEXT_COMMANDS and not redirection_target:
        return None
    if text.startswith("-"):
        if "=" not in text:
            return None
        return _workspace_scope_candidate_violation(text.split("=", 1)[1])
    assignment = _WORKSPACE_SCOPE_ASSIGNMENT_RE.match(text)
    if assignment:
        return _workspace_scope_candidate_violation(assignment.group(1))
    return _workspace_scope_candidate_violation(text)


def _format_workspace_scope_path(value: str) -> str:
    text = str(value or "").strip().strip("'\"")
    if text.startswith("/"):
        name = Path(text.rstrip("/ ")).name or "/"
        return f"`<host-path:{name}>`"
    return f"`{text}`"


def workspace_scope_path_blocked_error(
    command: str,
    workspace_dir: Path | str,
    allowed_roots: Sequence[Path | str] = (),
) -> str | None:
    """Block explicit paths outside the logical workspace bash boundary.

    This is intentionally stricter than checking whether a resolved path happens
    to live under the current workspace. LNR memories must stay spatially stable,
    so managed bash should use workspace-relative paths and let workspace symlinks
    such as ``dataset/`` hide host-specific locations.
    """
    _ = workspace_dir, allowed_roots
    s = _strip_heredoc_bodies_for_process_control(command).strip()
    if not s:
        return None
    queue = _split_shell_control_segments(s) or [s]
    seen: set[str] = set()
    violations: list[str] = []
    while queue:
        segment = queue.pop(0)
        if segment in seen:
            continue
        seen.add(segment)
        tokens = _split_shell_words(segment)
        queue.extend(_nested_shell_c_commands(tokens))
        for group in _split_pipeline_token_groups(tokens):
            runnable = _strip_shell_command_wrappers(_strip_leading_env_and_timeout(group))
            command_name = runnable[0].lower() if runnable else ""
            redirection_pending = False
            for token in group:
                if _WORKSPACE_SCOPE_REDIR_TOKEN_RE.match(token):
                    redirection_pending = True
                    continue
                violation = _workspace_scope_token_violation(
                    token,
                    command_name=command_name,
                    redirection_target=redirection_pending,
                )
                redirection_pending = False
                if violation and violation not in violations:
                    violations.append(violation)
                    if len(violations) >= 3:
                        break
            if len(violations) >= 3:
                break
    if not violations:
        return None
    shown = ", ".join(_format_workspace_scope_path(v) for v in violations[:3])
    return (
        "Blocked: bash is workspace-scoped in this REPL. "
        f"Found path outside the logical workspace boundary: {shown}. "
        "cwd is already the task workspace; use relative paths such as "
        "`dataset/`, `solution.py`, `submission.csv`, or `tmp/...`. "
        "Do not use host absolute paths, `/`, or `..` parent-directory escapes."
    )


def _global_scan_tokens_blocked(tokens: list[str]) -> bool:
    if not tokens:
        return False
    first = tokens[0].lower()
    has_global_target = any(_is_global_scan_root(token) for token in tokens[1:])
    if not has_global_target:
        return False
    if first == "find":
        return True
    if first == "du":
        return True
    if first == "rg":
        return True
    if first == "tree":
        return True
    if first in {"grep", "egrep", "fgrep"}:
        return _has_short_or_long_flag(tokens[1:], "R", {"--recursive", "--dereference-recursive"})
    if first == "ls":
        return _has_short_or_long_flag(tokens[1:], "R", {"--recursive"})
    return False


def global_filesystem_scan_blocked_error(command: str) -> str | None:
    """Block broad filesystem scans that can hang workers or walk unrelated mounts.

    The managed REPL starts in the task workspace. Discovery should be scoped to
    the workspace or an explicit known subdirectory, not host roots such as `/`,
    `/home`, or `/work`. Heredoc bodies are ignored so source text examples do not
    trigger this guard.
    """
    s = _strip_heredoc_bodies_for_process_control(command).strip()
    if not s:
        return None
    queue = _split_shell_control_segments(s) or [s]
    seen: set[str] = set()
    while queue:
        segment = queue.pop(0)
        if segment in seen:
            continue
        seen.add(segment)
        tokens = _strip_shell_command_wrappers(_strip_leading_env_and_timeout(_split_shell_words(segment)))
        if _global_scan_tokens_blocked(tokens):
            return (
                "Blocked: global filesystem scans are not allowed in managed bash. "
                "Search the current workspace or a known narrow path instead, e.g. "
                "`find . ...`, `find tmp/deps ...`, `command -v lmplz`, or "
                "`python3 -c 'import shutil; print(shutil.which(\"lmplz\"))'`."
            )
        queue.extend(_nested_shell_c_commands(tokens))
    return None


_SLOW_CMD_KEYWORDS = frozenset(
    {
        "find",
        "tar",
        "pip",
        "uv",
        "apt",
        "apt-get",
        "wget",
        "curl",
        "git",
        "npm",
        "yarn",
        "torchrun",
        "accelerate",
        "deepspeed",
    }
)


def _parse_cpu_set_string(cpu_set: str) -> list[int]:
    """Parse a taskset-style cpu-set string (e.g. '0-3,8,10-11') into a sorted list of ints."""
    ids: list[int] = []
    for part in cpu_set.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                lo, hi = part.split("-", 1)
                ids.extend(range(int(lo), int(hi) + 1))
            except ValueError:
                pass
        else:
            try:
                ids.append(int(part))
            except ValueError:
                pass
    return sorted(set(ids))


def _format_cpu_set_compact(cpus: list[int]) -> str:
    """Format sorted CPU ids into a compact taskset range string (e.g. '0-3,8,10-11')."""
    if not cpus:
        return ""
    arr = sorted(set(cpus))
    parts: list[str] = []
    start = arr[0]
    prev = arr[0]
    for c in arr[1:]:
        if c == prev + 1:
            prev = c
            continue
        parts.append(f"{start}-{prev}" if start != prev else str(start))
        start = c
        prev = c
    parts.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(parts)


def _command_executes_under_readonly_dir(
    command: str, readonly_dirs: list[str]
) -> str | None:
    """Return the first matched directory basename if *command* appears to execute under it.

    Read-only dirs are workspace-relative names (e.g. ``parent_workspace``). We block
    obvious execution (python/bash on paths under those dirs) while allowing inspection
    commands like ``cat parent_workspace/result.md``.
    """
    cmd = (command or "").strip()
    if not cmd or not readonly_dirs:
        return None
    for d in readonly_dirs:
        name = (d or "").strip()
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            continue
        esc = re.escape(name)
        # python/ipython: interpreter and readonly path must appear in the same shell
        # segment (do not match across && ; | — avoids FP e.g. cp parent_workspace/... && python x.py).
        if re.search(rf"\b(python3?|ipython)\b[^;&|]*\b{esc}/", cmd, re.IGNORECASE):
            return name
        # bash/sh/source or dot-source: script path under readonly dir in same segment
        if re.search(
            rf"(?:\b(?:bash|sh|source)\b|\.\s+)[^;&|]*\b{esc}/[^\s;|&`'\"]+\.(?:py|sh)\b",
            cmd,
            re.IGNORECASE,
        ):
            return name
        # cd into readonly dir then execute
        if re.search(rf"cd\s+[^\n;&|()]*\b{esc}\b", cmd, re.IGNORECASE) and re.search(
            r"&&\s*\b(python3?|bash|sh|ipython)\b", cmd, re.IGNORECASE
        ):
            return name
    return None


def hidden_workspace_path_usage_blocked_error(
    command: str,
    denied_prefixes: Sequence[str | Path] = (),
) -> str | None:
    prefixes = _normalise_hidden_prefixes(denied_prefixes)
    if not prefixes:
        return None
    for chunk in _SHELL_CHAIN_SPLIT_RE.split(str(command or "")):
        if not chunk or chunk.strip() in {"&&", "||", ";"}:
            continue
        tokens = _strip_shell_command_wrappers(_strip_leading_env_and_timeout(_split_shell_words(chunk)))
        for token in tokens:
            if _token_mentions_hidden_prefix(token, prefixes):
                return (
                    "Blocked: this path is system-visible but hidden from the agent. "
                    "Use ordinary workspace files such as `tmp/...` for agent notes and logs."
                )
    return None


def _leading_cd_abs_path_missing(
    command: str,
    workspace_dir: Path | str | None = None,
    allowed_roots: Sequence[Path | str] = (),
) -> str | None:
    """Check a command that starts with ``cd <absolute_path>`` and return error text when blocked.

    Blocks when:
    - The target path does not exist (original behaviour).
    - The target exists but is outside the workspace, all *allowed_roots*, and /tmp
      (new: catches ``cd /home`` style hallucinations that happen to be real directories).

    *workspace_dir* and *allowed_roots* are optional; when omitted only the
    existence check is performed (preserves backward-compatibility for callers
    that do not yet pass a workspace).
    """
    cmd = (command or "").strip()
    if not cmd:
        return None
    m = re.match(r"^\s*cd\s+([^\s;&|]+)", cmd)
    if not m:
        return None
    target = m.group(1).strip().strip("\"'")
    if not target.startswith("/"):
        return None

    try:
        target_path = Path(target).resolve()
        exists = target_path.exists()
    except OSError:
        exists = False

    if not exists:
        return (
            "cd target is an absolute host path that does not exist or is not usable. "
            "Use paths relative to the workspace (e.g. `dataset/`). "
            "Run `pwd` to confirm cwd if needed; do not use external notebook-style paths."
        )

    # If we know the workspace, also block cd into directories outside allowed trees.
    if workspace_dir is None:
        return None

    ws = Path(workspace_dir).resolve()

    # Build the set of allowed absolute root trees.
    _allowed: list[Path] = [ws, Path("/tmp")]
    for r in allowed_roots:
        try:
            _allowed.append(Path(r).resolve())
        except (TypeError, ValueError):
            pass

    def _under_any(p: Path) -> bool:
        for root in _allowed:
            try:
                p.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    if _under_any(target_path):
        return None

    return (
        "cd target is outside the task workspace. "
        "cwd is already set to the workspace — run commands directly (e.g. `python3 solution.py`) "
        "without prefixing an absolute-path `cd`. "
        "Use workspace-relative paths (e.g. `dataset/`) instead."
    )


def _normalise_hidden_prefixes(prefixes: Sequence[str | Path]) -> tuple[tuple[str, ...], ...]:
    out: list[tuple[str, ...]] = []
    for raw in prefixes:
        text = str(raw or "").strip().replace("\\", "/").strip("/")
        if not text:
            continue
        parts = tuple(part for part in text.split("/") if part not in {"", "."})
        if not parts or any(part == ".." for part in parts):
            continue
        if parts not in out:
            out.append(parts)
    return tuple(out)


def _token_mentions_hidden_prefix(token: str, prefixes: tuple[tuple[str, ...], ...]) -> bool:
    text = str(token or "").strip().strip("'\"").replace("\\", "/")
    if not text or text.startswith("-") or "://" in text:
        return False
    for sep in (">", "<"):
        if sep in text:
            text = text.split(sep, 1)[-1]
    text = text.strip().strip("'\"")
    if text.startswith("./"):
        text = text[2:]
    if not text or text.startswith("/"):
        return False
    parts = tuple(part for part in text.split("/") if part not in {"", "."})
    if not parts:
        return False
    return any(len(parts) >= len(prefix) and parts[: len(prefix)] == prefix for prefix in prefixes)


def _host_absolute_path_usage_blocked_error(command: str) -> str | None:
    """Return a model-facing error when a command uses host absolute paths.

    This guard is intentionally opt-in. It is used by lnr where
    the agent is expected to operate entirely from the task workspace and keep
    memory spatially stable across restore/rewind operations.
    """
    cmd = str(command or "")
    matches: list[str] = []
    for m in _HOST_ABSOLUTE_PATH_IN_COMMAND_RE.finditer(cmd):
        value = m.group(0)
        if value not in matches:
            matches.append(value)
        if len(matches) >= 3:
            break
    if not matches:
        return None
    shown = ", ".join(f"`<host-path:{Path(x.rstrip('/ ')).name or 'path'}>`" for x in matches)
    return (
        "Blocked: host absolute paths are not allowed in this workspace REPL. "
        f"Found {shown}. cwd is already the task workspace; use relative paths "
        "such as `dataset/`, `solution.py`, `submission.csv`, or `tmp/...`. "
        "Do not use host absolute paths."
    )


def _normalize_python_to_python3(command: str) -> str:
    """Replace bare ``python`` with ``python3`` when it starts a command segment.

    Many Linux images have no ``python`` symlink (exit 127). Does not alter
    ``python3``, paths like ``/usr/bin/python``, or the word ``python`` inside strings.
    """
    if not command:
        return command
    out = re.sub(r"(?m)^\s*python(\s|$)", r"python3\1", command)
    out = re.sub(r"(&&\s*)python(\s|$)", r"\1python3\2", out)
    out = re.sub(r"(\|\|\s*)python(\s|$)", r"\1python3\2", out)
    out = re.sub(r"(;\s*)python(\s|$)", r"\1python3\2", out)
    out = re.sub(r"(\|\s*)python(\s|$)", r"\1python3\2", out)
    return out


_SHELL_CHAIN_SPLIT_RE = re.compile(r"(\s*(?:&&|\|\||;)\s*)")


_ENV_ASSIGN_PREFIX_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*=(?:(?:'[^']*'|\"[^\"]*\"|\S+))\s+)+",
)


def _rewrite_bare_pip_command(command: str) -> tuple[str, bool]:
    """Rewrite bare ``pip`` / ``pip3`` at command start to ``uv pip ...``.

    Leaves ``uv pip``, ``python -m pip``, conda wrappers, etc. unchanged.
    Returns ``(new_command, did_rewrite)``.
    """
    if not (command or "").strip():
        return command, False
    m = _ENV_ASSIGN_PREFIX_RE.match(command)
    prefix = m.group(0) if m else ""
    rest = command[len(prefix) :].lstrip()
    if not rest:
        return command, False
    parts = rest.split(None, 1)
    first = parts[0]
    if first not in ("pip", "pip3"):
        return command, False
    tail = rest[len(first) :].lstrip()
    body = f"uv pip {tail}".strip() if tail else "uv pip"
    return prefix + body, True


def _rewrite_bare_pip_in_shell_chain(command: str) -> tuple[str, bool]:
    """Rewrite bare ``pip`` / ``pip3`` at the start of each ``&&`` / ``||`` / ``;`` segment."""
    if not (command or "").strip():
        return command, False
    parts = _SHELL_CHAIN_SPLIT_RE.split(command)
    if len(parts) == 1:
        return _rewrite_bare_pip_command(parts[0])
    out_chunks: list[str] = []
    any_rewrite = False
    for i, chunk in enumerate(parts):
        if i % 2 == 1:
            out_chunks.append(chunk)
            continue
        seg, did = _rewrite_bare_pip_command(chunk)
        any_rewrite = any_rewrite or did
        out_chunks.append(seg)
    return "".join(out_chunks), any_rewrite


_PRIV_ESC_FIRST_TOKENS = frozenset({"sudo", "su", "doas", "pkexec"})


_PROCESS_CONTROL_FIRST_TOKENS = frozenset({"kill", "pkill", "killall"})


_PROCESS_CONTROL_SPLIT_RE = re.compile(r"(?:&&|\|\||[;\n|])")


_HEREDOC_START_RE = re.compile(r"<<-?\s*[\'\"]?([A-Za-z_][A-Za-z0-9_-]*)[\'\"]?")


def _segment_first_executable_token(segment: str) -> str | None:
    """First shell word after optional leading ``VAR=value`` assignments; None if absent."""
    seg = (segment or "").strip()
    if not seg:
        return None
    m = _ENV_ASSIGN_PREFIX_RE.match(seg)
    prefix = m.group(0) if m else ""
    rest = seg[len(prefix) :].lstrip()
    if not rest:
        return None
    return rest.split(None, 1)[0]


_PY_ENV_WRITE_SUBCOMMANDS = frozenset({"install", "uninstall", "sync"})


_UV_ENV_WRITE_SUBCOMMANDS = frozenset({"sync", "add", "remove"})


_LOCAL_INSTALL_FLAGS = ("--target", "--prefix")


def _split_shell_words(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _strip_leading_env_and_timeout(tokens: list[str]) -> list[str]:
    out = list(tokens)
    changed = True
    while out and changed:
        changed = False
        while out and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", out[0]):
            out.pop(0)
            changed = True
        if out and out[0] == "timeout":
            out.pop(0)
            if out:
                out.pop(0)
            changed = True
        if out and out[0] == "env":
            out.pop(0)
            while out and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", out[0]):
                out.pop(0)
            changed = True
    return out


def _strip_shell_command_wrappers(tokens: list[str]) -> list[str]:
    out = list(tokens)
    while out and out[0] in {"command", "builtin", "exec"}:
        out.pop(0)
    return out


def _has_local_install_target(tokens: list[str]) -> bool:
    for token in tokens:
        if token in _LOCAL_INSTALL_FLAGS:
            return True
        if any(token.startswith(flag + "=") for flag in _LOCAL_INSTALL_FLAGS):
            return True
    return False


def shared_python_env_write_blocked_error(command: str) -> str | None:
    """Block commands that mutate the shared interpreter environment.

    Workspace-local dependency installs are still possible via --target, --prefix,
    or an explicitly created workspace venv such as tmp/venv/bin/python.
    """
    s = (command or "").strip()
    if not s:
        return None
    low = s.lower()
    mutates_site = bool(re.search(r"\brm\s+[^\n;]*-[^\n;]*[rf][^\n;]*(?:\$site|site-packages|dist-packages|numpy|scipy|sklearn)", low))
    mentions_shared_site = "site.getsitepackages" in low or "site-packages" in low or "dist-packages" in low or "$site" in low
    if mutates_site and mentions_shared_site:
        return _shared_python_env_block_message("direct site-packages mutation")

    parts = _SHELL_CHAIN_SPLIT_RE.split(s)
    for i, chunk in enumerate(parts):
        if i % 2 == 1:
            continue
        tokens = _strip_leading_env_and_timeout(_split_shell_words(chunk))
        if not tokens:
            continue
        first = tokens[0]
        if first == "uv":
            if len(tokens) >= 2 and tokens[1] in _UV_ENV_WRITE_SUBCOMMANDS:
                return _shared_python_env_block_message(f"uv {tokens[1]}")
            if len(tokens) >= 3 and tokens[1] == "pip" and tokens[2] in _PY_ENV_WRITE_SUBCOMMANDS:
                if tokens[2] == "install" and _has_local_install_target(tokens[3:]):
                    continue
                return _shared_python_env_block_message(f"uv pip {tokens[2]}")
            if len(tokens) >= 4 and tokens[1] == "run" and tokens[2] in {"pip", "pip3"} and tokens[3] in _PY_ENV_WRITE_SUBCOMMANDS:
                if tokens[3] == "install" and _has_local_install_target(tokens[4:]):
                    continue
                return _shared_python_env_block_message(f"uv run pip {tokens[3]}")
            if len(tokens) >= 6 and tokens[1] == "run" and tokens[2] in {"python", "python3"} and tokens[3:5] == ["-m", "pip"] and tokens[5] in _PY_ENV_WRITE_SUBCOMMANDS:
                if tokens[5] == "install" and _has_local_install_target(tokens[6:]):
                    continue
                return _shared_python_env_block_message(f"uv run python -m pip {tokens[5]}")
        if first in {"pip", "pip3"} and len(tokens) >= 2 and tokens[1] in _PY_ENV_WRITE_SUBCOMMANDS:
            if tokens[1] == "install" and _has_local_install_target(tokens[2:]):
                continue
            return _shared_python_env_block_message(f"{first} {tokens[1]}")
        if first in {"python", "python3"} and len(tokens) >= 4 and tokens[1:3] == ["-m", "pip"] and tokens[3] in _PY_ENV_WRITE_SUBCOMMANDS:
            if tokens[3] == "install" and _has_local_install_target(tokens[4:]):
                continue
            return _shared_python_env_block_message(f"{first} -m pip {tokens[3]}")
    return None


def _shared_python_env_block_message(reason: str) -> str:
    return (
        "Blocked: modifying the shared Python environment is not allowed in managed resource runs "
        f"({reason}). Use installed packages, implement a code fallback, or install into a workspace-local target/venv "
        "such as `uv pip install --target tmp/deps <pkg>` or `python3 -m venv tmp/venv && tmp/venv/bin/python -m pip install <pkg>`."
    )


def privilege_escalation_blocked_error(command: str) -> str | None:
    """If *command* starts a chain segment with sudo/su/doas/pkexec, return a block message.

    Only the first executable token of each ``&&`` / ``||`` / ``;`` segment is checked
    (after env-assign prefixes), so ``echo sudo`` is allowed.
    """
    s = (command or "").strip()
    if not s:
        return None
    parts = _SHELL_CHAIN_SPLIT_RE.split(s)
    for i, chunk in enumerate(parts):
        if i % 2 == 1:
            continue
        tok = _segment_first_executable_token(chunk)
        if tok is not None and tok.lower() in _PRIV_ESC_FIRST_TOKENS:
            return (
                "Blocked: privilege escalation commands (sudo, su, doas, pkexec) are not allowed. "
                "Use the task environment without elevating privileges "
                "(e.g. `uv pip install ...`)."
            )
    return None


def _strip_heredoc_bodies_for_process_control(command: str) -> str:
    lines = str(command or "").splitlines()
    if not lines:
        return ""
    out: list[str] = []
    pending_delimiters: list[str] = []
    for line in lines:
        stripped = line.strip()
        if pending_delimiters:
            if stripped == pending_delimiters[-1]:
                pending_delimiters.pop()
            continue
        out.append(line)
        for match in _HEREDOC_START_RE.finditer(line):
            pending_delimiters.append(match.group(1))
    return "\n".join(out)


def process_control_blocked_error(command: str) -> str | None:
    """Block direct process-control commands from agent bash.

    Worker-local cleanup must go through the safety/resource-arbiter path. Direct
    shell process control is not task-scoped: commands like ``pkill -f python`` can
    terminate other workers and the run controller. Heredoc bodies are ignored so
    normal code-writing commands are not blocked by source text.
    """
    s = _strip_heredoc_bodies_for_process_control(command).strip()
    if not s:
        return None
    for chunk in _PROCESS_CONTROL_SPLIT_RE.split(s):
        if not chunk.strip():
            continue
        stripped = chunk.lstrip()
        if stripped.startswith("#"):
            continue
        tokens = _strip_shell_command_wrappers(_strip_leading_env_and_timeout(_split_shell_words(chunk)))
        if not tokens:
            continue
        first = tokens[0].lower()
        if first in _PROCESS_CONTROL_FIRST_TOKENS:
            return _process_control_block_message(first)
        if first == "xargs" and any(tok.lower() in _PROCESS_CONTROL_FIRST_TOKENS for tok in tokens[1:]):
            return _process_control_block_message("xargs kill")
        for idx, token in enumerate(tokens[:-1]):
            if token == "-exec" and tokens[idx + 1].lower() in _PROCESS_CONTROL_FIRST_TOKENS:
                return _process_control_block_message(f"-exec {tokens[idx + 1].lower()}")
    return None


def _process_control_block_message(reason: str) -> str:
    return (
        "Blocked: direct process-control commands from agent bash are not allowed "
        f"({reason}). They can terminate other workers or the run controller. "
        "Only task-scoped cleanup through the safety/resource arbiter is allowed."
    )


_INTERACTIVE_PROGRAMS = frozenset(
    {"htop", "less", "man", "more", "nano", "select", "top", "vi", "vim"}
)


def interactive_stdin_blocked_error(command: str) -> str | None:
    """Reject commands that require a human-controlled terminal or stdin.

    Explicit stdin redirection remains valid, for example
    ``read -r line < dataset/train.txt``. The execution layer also supplies
    ``/dev/null`` as the default stdin so an unrecognized interactive program
    cannot wait on the worker's tmux terminal indefinitely.
    """
    text = _strip_heredoc_bodies_for_process_control(command).strip()
    if not text:
        return None
    if re.search(r"(?:^|[^>])/dev/tty(?:\s|$)", text):
        return _interactive_stdin_block_message("/dev/tty")
    for segment in _split_shell_control_segments(text):
        tokens = _strip_shell_command_wrappers(
            _strip_leading_env_and_timeout(_split_shell_words(segment))
        )
        if not tokens:
            continue
        first = tokens[0].lower().rsplit("/", 1)[-1]
        if first == "read":
            has_explicit_input = any("<" in token for token in tokens[1:])
            if not has_explicit_input:
                return _interactive_stdin_block_message("read without redirected input")
        if first in _INTERACTIVE_PROGRAMS:
            return _interactive_stdin_block_message(first)
        if first in {"bash", "sh", "python", "python3"} and "-i" in tokens[1:]:
            return _interactive_stdin_block_message(f"{first} interactive mode")
    return None


def _interactive_stdin_block_message(reason: str) -> str:
    return (
        "Blocked: interactive terminal/stdin commands are not allowed in managed bash "
        f"({reason}). Use a non-interactive command with explicit input from a workspace "
        "file, pipe, or heredoc."
    )


def normalize_bash_command_for_agent(command: str) -> tuple[str, bool]:
    """Apply ``python``→``python3`` and bare-``pip``/``pip3``→``uv pip``; returns ``(cmd, pip_rewritten)``."""
    s = (command or "").strip()
    if not s:
        return s, False
    out = _normalize_python_to_python3(s)
    out, pip_rewritten = _rewrite_bare_pip_in_shell_chain(out)
    return out, pip_rewritten


def _infer_timeout(command: str, default_sec: float, slow_sec: float) -> float:
    s = command.strip()
    if not s:
        return default_sec
    if _SLOW_ENTRYPOINT_RE.search(s):
        return slow_sec
    parts = s.split()
    if not parts:
        return default_sec
    first = parts[0].lstrip("./")
    # After bare-pip rewrite: uv pip install ...
    # Metadata-only pip commands should stay on the normal timeout; otherwise a
    # harmless `pip list` probe can monopolize a long-run worker for hours.
    if len(parts) >= 2 and first == "uv" and parts[1] == "pip":
        subcmd = parts[2] if len(parts) >= 3 else ""
        if subcmd in {"install", "sync", "compile"}:
            return slow_sec
        return default_sec
    if first in _SLOW_CMD_KEYWORDS:
        return slow_sec
    # python3 -m pip ... (unchanged by agent)
    if (
        len(parts) >= 3
        and first in ("python", "python3")
        and parts[1] == "-m"
        and parts[2] == "pip"
    ):
        return slow_sec
    return default_sec
