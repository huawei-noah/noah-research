#!/usr/bin/env python3
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

"""Rewrite tasks/ml/mlebench/*/description_lite.md to <= 30 lines."""

from __future__ import annotations

import re
from pathlib import Path

TASKS = Path(__file__).resolve().parent.parent / "tasks" / "ml" / "mlebench"
MAX_LINES = 30
WRAP = 118

SECTION_ORDER = [
    "Task description",
    "Task objective",
    "Target metric (evaluation)",
    "Brief background",
    "Submission",
    "Dataset and construction",
]

# Per-section body line budget (physical lines under each ##). Sum + title + blanks must stay <= MAX_LINES.
DEFAULT_BODY_LINES = (2, 1, 2, 1, 3, 3)


def extract_title(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# ") and "Lite Task Description" in s:
            return s
    return "# Lite Task Description"


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    parts = re.split(r"^##\s+", text, flags=re.MULTILINE)
    for i in range(1, len(parts)):
        block = parts[i]
        nl = block.find("\n")
        if nl == -1:
            name, body = block.strip(), ""
        else:
            name, body = block[:nl].strip(), block[nl + 1 :].strip()
        if name in sections:
            sections[name] += "\n\n" + body
        else:
            sections[name] = body
    return sections


def fence_replace(m: re.Match[str]) -> str:
    inner = m.group(1).strip()
    lines = [ln.strip() for ln in inner.splitlines() if ln.strip()]
    if not lines:
        return ""
    head = lines[0].replace("`", "")
    if len(lines) >= 8 or (len(lines) > 3 and "," in head and head.count(",") >= 2):
        return (
            f"Tabular example (abridged): `{head}` — follow `sample_submission.csv` / official spec for full columns."
        )
    if len(lines) <= 4:
        return "Example: " + " | ".join(lines[:3]) + (" …" if len(lines) > 3 else "")
    return f"Example starts: `{lines[0]}` (+ {len(lines) - 1} more lines; see sample submission)."


def strip_markdown_noise(body: str) -> str:
    body = re.sub(r"```(?:[^\n`]*\n)?(.*?)```", lambda m: fence_replace(m), body, flags=re.DOTALL)
    body = re.sub(r"^---\s*$", "", body, flags=re.MULTILINE)
    body = re.sub(r"^#{1,4}\s+.+$", "", body, flags=re.MULTILINE)
    body = re.sub(r"\*\*([^*]+)\*\*", r"\1", body)
    body = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", body)
    body = re.sub(r"\s+", " ", body).strip()
    return body


def dedupe_sentences(text: str) -> str:
    """Remove repeated sentence-like chunks (order preserved)."""
    text = text.strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        q = p.strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return " ".join(out)


def shrink_at_sentence(text: str, max_chars: int) -> str:
    text = dedupe_sentences(text)
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars + 1]
    for sep in (". ", "; ", " — ", "? ", "! "):
        pos = cut.rfind(sep)
        if pos > max(20, max_chars // 3):
            return cut[: pos + len(sep)].strip()
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > 50 else cut[:max_chars]).rstrip() + "…"


def wrap_hard(text: str, width: int, max_lines: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    out: list[str] = []
    rest = text
    while rest and len(out) < max_lines:
        if len(rest) <= width:
            out.append(rest)
            break
        br = rest.rfind(" ", 0, width + 1)
        if br < width // 3:
            br = width
        chunk = rest[:br].strip()
        rest = rest[br:].strip()
        if chunk:
            out.append(chunk)
    if rest:
        if out:
            merged = (out[-1] + " " + rest).strip()
            out[-1] = shrink_at_sentence(merged, width * max_lines)
        else:
            out.append(shrink_at_sentence(rest, width * max_lines))
    # drop consecutive duplicates
    deduped: list[str] = []
    for ln in out:
        if deduped and ln == deduped[-1]:
            continue
        deduped.append(ln)
    return deduped


def match_section(cleaned: dict[str, str], canonical: str) -> str:
    for k, v in cleaned.items():
        if k == canonical or k.startswith(canonical):
            return v.strip()
    return ""


def compress_text(raw: str) -> str:
    title = extract_title(raw)
    sections = split_sections(raw)
    cleaned = {k: strip_markdown_noise(v) for k, v in sections.items()}

    desc = match_section(cleaned, "Task description")
    obj = match_section(cleaned, "Task objective")
    if not desc and obj:
        desc = shrink_at_sentence(obj, 240)

    # If description largely repeats objective, keep shorter description only
    if desc and obj:
        d0 = desc[:120].lower()
        o0 = obj[:120].lower()
        if d0 == o0 or (len(d0) > 40 and d0 == o0[: len(d0)]):
            desc = shrink_at_sentence(desc, min(200, len(desc)))

    fields: dict[str, str] = {
        "Task description": desc,
        "Task objective": obj,
        "Target metric (evaluation)": match_section(cleaned, "Target metric"),
        "Brief background": match_section(cleaned, "Brief background"),
        "Submission": match_section(cleaned, "Submission"),
        "Dataset and construction": match_section(cleaned, "Dataset and construction"),
    }

    char_caps = (380, 320, 420, 360, 520, 560)

    def render(body_lines: tuple[int, ...]) -> str:
        lines: list[str] = [title, ""]
        for name, mlines, cap in zip(SECTION_ORDER, body_lines, char_caps, strict=True):
            body = fields.get(name, "").strip()
            if not body:
                continue
            body = shrink_at_sentence(body, cap * 2 if mlines > 1 else cap)
            wrapped = wrap_hard(body, WRAP, mlines)
            lines.append(f"## {name}")
            if wrapped:
                lines.extend(wrapped)
            else:
                lines.append("—")
            lines.append("")
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines) + "\n"

    # Try default budgets; if still too long, reduce body lines.
    for body_lines in (
        DEFAULT_BODY_LINES,
        (1, 1, 2, 1, 2, 2),
        (1, 1, 1, 1, 2, 2),
        (1, 1, 1, 1, 1, 2),
        (1, 1, 1, 1, 1, 1),
    ):
        out = render(body_lines)
        if len(out.splitlines()) <= MAX_LINES:
            return out

    # Last resort: shrink caps
    char_caps_small = (220, 200, 240, 200, 280, 300)

    lines = [title, ""]
    for name, cap in zip(SECTION_ORDER, char_caps_small, strict=True):
        body = fields.get(name, "").strip()
        if not body:
            continue
        body = shrink_at_sentence(body, cap)
        w = wrap_hard(body, WRAP, 1)
        lines.append(f"## {name}")
        lines.extend(w if w else ["—"])
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def main() -> None:
    if not TASKS.is_dir():
        raise SystemExit(f"tasks dir not found: {TASKS}")
    n = 0
    for path in sorted(TASKS.glob("*/description_lite.md")):
        raw = path.read_text(encoding="utf-8")
        new = compress_text(raw)
        if len(new.splitlines()) > MAX_LINES:
            raise SystemExit(f"still too long: {path} ({len(new.splitlines())} lines)")
        path.write_text(new, encoding="utf-8")
        n += 1
    print(f"compressed {n} files to <= {MAX_LINES} lines")


if __name__ == "__main__":
    main()
