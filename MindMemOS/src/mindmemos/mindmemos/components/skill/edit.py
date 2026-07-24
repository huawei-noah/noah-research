"""Line-addressed structured editing of a ``SKILL.md`` document.

The evolve pipeline used to make the LLM re-emit the COMPLETE ``SKILL.md`` for
every patch, which wastes a full skill's worth of output tokens per batch and
risks the model silently rewriting untouched content. An earlier fix had the
model emit *verbatim-anchored* find-and-replace ops (copy the existing text into
``old``/``anchor``), but that traded one failure mode for another: LLMs cannot
reliably reproduce existing text byte-for-byte (whitespace, smart quotes, elided
spans), so anchors routinely failed to match or matched ambiguously, and the
model also had to hand-manage separators and JSON-escape large multi-line spans.

This component instead addresses edits by LINE NUMBER. The apply prompt shows the
current ``SKILL.md`` with a ``N|`` gutter (see :func:`format_numbered`), and the
model emits ops that reference those numbers and carry ONLY the new text -- it
never copies existing content back. That removes the verbatim-copy, uniqueness,
separator, and JSON-escaping failure modes at once, and shrinks the payload.

Wire format (the apply prompt instructs the model to emit this JSON):

    {"edits": [
        {"op": "replace", "start": 6, "end": 6, "new": "...", "old_string_prefix": "..."},
        {"op": "delete",  "start": 10, "end": 12, "old_string_prefix": "..."},
        {"op": "insert",  "after": 5, "new": "..."}
    ]}

Line numbers are 1-based and inclusive. ``insert`` places ``new`` AFTER the given
line (``after: 0`` = top of file, ``after: <line count>`` = end). ``old_string_prefix``
is an OPTIONAL safety belt on ``replace``/``delete``: when present it is compared
(whitespace-normalized) against the prefix of the document line at ``start`` to
catch off-by-one addressing without asking the model to copy very long lines.
``old_first_line`` is still accepted as a backward-compatible alias. A mismatch
raises :class:`SkillEditError` showing the actual line, so the chat
``format_parser`` retry can re-read the gutter and correct the number.

Edits are validated to reference real lines and to have non-overlapping
replace/delete ranges, then applied BOTTOM-UP (highest line first) against the
ORIGINAL line numbers, so one edit never shifts the numbering another edit relies
on. The model never has to reason about post-edit offsets.

A bare top-level list is also accepted. An empty edit list is valid and returns
the document unchanged (the "no edits needed" case).
"""

from __future__ import annotations

import json
import re
from typing import Any

from ...errors import SkillEditError


def format_numbered(skill_md: str) -> str:
    """Render ``skill_md`` with a 1-based ``N|`` line-number gutter for the model.

    The numbering matches the line indices the apply ops reference: line ``k`` in
    the gutter is the ``k``-th element of ``str.splitlines()``, so the model and
    :func:`apply_edit_ops` agree on what each number means.
    """

    lines = skill_md.splitlines()
    if not lines:
        return "(empty document)"
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}}| {line}" for i, line in enumerate(lines, start=1))


def parse_edit_ops(raw: str) -> list[dict[str, Any]]:
    """Parse the model's reply into a list of edit-op dicts.

    Accepts an optional ```json fence and either ``{"edits": [...]}`` or a bare
    top-level list.

    Raises:
        SkillEditError: If the payload is not valid JSON or not an op list.
    """

    text = _strip_code_fence((raw or "").strip())
    if not text:
        raise SkillEditError("empty edit payload")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SkillEditError(f"edit payload is not valid JSON: {exc}") from exc

    if isinstance(data, dict):
        data = data.get("edits", data.get("ops"))
    if not isinstance(data, list):
        raise SkillEditError("edit payload must be a list or an object with an 'edits' list")
    for i, op in enumerate(data):
        if not isinstance(op, dict):
            raise SkillEditError(f"edit #{i} is not an object")
    return data


def apply_edit_ops(skill_md: str, edits: list[dict[str, Any]]) -> str:
    """Apply line-addressed edit ops to ``skill_md`` and return the new document.

    Ops reference the ORIGINAL line numbers (1-based, inclusive). They are applied
    bottom-up so earlier-applied edits never shift the numbering later ones rely on;
    the relative order of ops at the same position is preserved. An empty list
    returns ``skill_md`` unchanged.

    Raises:
        SkillEditError: On an unknown op, an out-of-range or inverted line range,
            overlapping replace/delete ranges, or an ``old_string_prefix`` that
            does not match the addressed line.
    """

    lines = skill_md.splitlines(keepends=True)
    n = len(lines)

    # Resolve every op to a (slice_start, slice_end, replacement_lines) plan against
    # the ORIGINAL line list, validating as we go. slice bounds are 0-based Python
    # slice indices into ``lines``.
    plans: list[tuple[int, int, list[str], str]] = []
    ranges: list[tuple[int, int, int]] = []  # (start, end, op_index) for overlap check
    for i, op in enumerate(edits):
        kind = op.get("op")
        if kind in ("replace", "delete"):
            start = _require_line(op, "start", i, lo=1, hi=n)
            end = _require_line(op, "end", i, lo=start, hi=n)
            _check_line_guard(lines, start, op, i)
            new = "" if kind == "delete" else _as_str(op.get("new", ""))
            plans.append((start - 1, end, _to_lines(new), kind))
            ranges.append((start, end, i))
        elif kind == "insert":
            after = _require_line(op, "after", i, lo=0, hi=n)
            new = _as_str(op.get("new", ""))
            if new == "":
                raise SkillEditError(f"edit #{i}: insert 'new' must be non-empty")
            plans.append((after, after, _to_lines(new), kind))
        else:
            raise SkillEditError(f"edit #{i} has unknown op {kind!r}")

    _check_overlap(ranges)

    # Bottom-up: apply the highest slice_start first so an edit never shifts the
    # numbering a later-applied edit relies on. For ties at the same position, apply
    # the later-declared op first so the final text keeps declaration order.
    for _idx, (slice_start, slice_end, replacement, kind) in sorted(
        enumerate(plans), key=lambda item: (item[1][0], item[0]), reverse=True
    ):
        if (
            kind == "insert"
            and slice_start > 0
            and lines[slice_start - 1]
            and not lines[slice_start - 1].endswith("\n")
        ):
            lines[slice_start - 1] += "\n"
        lines[slice_start:slice_end] = replacement
    return "".join(lines)


def apply_patch_ops(skill_md: str, raw: str) -> str:
    """Parse the model reply and apply it to ``skill_md`` in one shot.

    Used as the chat ``format_parser`` in the evolve pipeline so a non-matching
    edit triggers an automatic retry.
    """

    return apply_edit_ops(skill_md, parse_edit_ops(raw))


_WS = re.compile(r"\s+")


def _normalize(line: str) -> str:
    """Collapse whitespace for tolerant line-prefix guard comparison."""

    return _WS.sub(" ", line).strip()


def _quote(value: str, limit: int = 160) -> str:
    shown = value if len(value) <= limit else value[:limit] + "…"
    return repr(shown)


def _to_lines(new: str) -> list[str]:
    """Split ``new`` into keepends lines, guaranteeing a trailing newline.

    The model supplies plain replacement text; we own the separators so an
    inserted/replaced block can never fuse onto the following line. An empty
    ``new`` yields no lines (so a ``replace`` with empty ``new`` deletes the span).
    """

    if new == "":
        return []
    parts = new.splitlines(keepends=True)
    if not parts[-1].endswith("\n"):
        parts[-1] += "\n"
    return parts


def _require_line(op: dict[str, Any], key: str, index: int, *, lo: int, hi: int) -> int:
    value = op.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SkillEditError(f"edit #{index}: missing required integer line field {key!r}")
    if value < lo or value > hi:
        raise SkillEditError(
            f"edit #{index}: {key}={value} is out of range; valid {key} is {lo}..{hi} "
            "for the current SKILL.md (line numbers are 1-based, from the N| gutter)."
        )
    return value


def _check_line_guard(lines: list[str], start: int, op: dict[str, Any], index: int) -> None:
    """Optional safety belt: verify the addressed line starts with the model's prefix."""

    expected = op.get("old_string_prefix")
    guard_name = "old_string_prefix"
    if expected is None:
        expected = op.get("old_first_line")
        guard_name = "old_first_line"
    if expected is None:
        return
    if not isinstance(expected, str):
        raise SkillEditError(f"edit #{index}: {guard_name!r} must be a string")
    expected = _normalize(expected)
    if not expected:
        raise SkillEditError(f"edit #{index}: {guard_name!r} must be non-empty")
    actual = lines[start - 1].rstrip("\n") if start - 1 < len(lines) else ""
    actual_normalized = _normalize(actual)
    if guard_name == "old_first_line":
        matched = actual_normalized == expected
    else:
        matched = actual_normalized.startswith(expected)
    if not matched:
        raise SkillEditError(
            f"edit #{index}: line {start} does not match {guard_name!r}, so the line "
            f"number is likely off. Line {start} is actually {_quote(actual)}, but you "
            f"gave {_quote(expected)}. Re-read the N| gutter and fix the line number."
        )


def _check_overlap(ranges: list[tuple[int, int, int]]) -> None:
    ordered = sorted(ranges)
    for (s1, e1, i1), (s2, e2, i2) in zip(ordered, ordered[1:]):
        if s2 <= e1:
            raise SkillEditError(
                f"edits #{i1} and #{i2} target overlapping line ranges "
                f"({s1}..{e1} and {s2}..{e2}); make replace/delete ranges disjoint."
            )


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise SkillEditError(f"expected a string, got {type(value).__name__}")
    return value


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
