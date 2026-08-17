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

"""Generate tasks/ml/mlebench/<slug>/description_lite.md from MLE-bench descriptions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_SLUG_TITLE_OVERRIDES: dict[str, str] = {
    "ai4code": "AI4Code",
}


def slug_display_title(slug: str) -> str:
    low = slug.lower()
    if low in _SLUG_TITLE_OVERRIDES:
        return _SLUG_TITLE_OVERRIDES[low]
    return slug.replace("-", " ").strip().title()


def strip_images(text: str) -> str:
    return re.sub(r"!\[[^\]]*\]\([^)]*\)\s*", "", text)


def links_to_label(text: str) -> str:
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)


def strip_latex_blocks(text: str) -> str:
    text = re.sub(r"\$\$[\s\S]*?\$\$", " *(formula in full description)* ", text)
    text = re.sub(r"\$[^$]+\$", " ", text)
    return text


def collapse_ws(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean(text: str, max_chars: int | None = None) -> str:
    text = text.replace("\u00a0", " ")
    text = strip_images(text)
    text = links_to_label(text)
    text = strip_latex_blocks(text)
    text = collapse_ws(text)
    if max_chars and len(text) > max_chars:
        cut = text[: max_chars + 1]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        text = cut.rstrip(",;:") + " … *(see full `description.md`.)*"
    return text


def add_metric_reporting_guidance(metric_part: str) -> str:
    guidance = (
        "Report `Final Validation Score` using the raw validation metric in the same units and "
        "direction as the leaderboard metric, and set `lower_is_better` to match that direction. "
        "Do not use transformed or normalized proxy scores as the primary score unless the "
        "competition metric itself is defined that way."
    )
    if not metric_part:
        return guidance
    if "raw validation metric" in metric_part or "Final Validation Score" in metric_part:
        return metric_part
    return metric_part.rstrip() + "\n\n" + guidance


def extract_h2(text: str, title: str) -> str | None:
    pat = rf"^##\s+{re.escape(title)}\s*\n(.*?)(?=^##\s+|\Z)"
    m = re.search(pat, text, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else None


def extract_h3(text: str, title: str) -> str | None:
    pat = rf"^###\s+{re.escape(title)}\s*\n(.*?)(?=^###\s+|^##\s+|\Z)"
    m = re.search(pat, text, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else None


def extract_header_depth2_to_5(text: str, title: str) -> str | None:
    """Extract a Markdown section, including any deeper nested subsections."""

    def normalize_heading(value: str) -> str:
        value = links_to_label(value)
        value = re.sub(r"[*_`]+", "", value)
        return re.sub(r"\s+", " ", value).strip().casefold()

    requested = normalize_heading(title)
    headings = list(re.finditer(r"^(#{2,5})\s+(.+?)\s*$", text, re.MULTILINE))
    for index, heading in enumerate(headings):
        depth = len(heading.group(1))
        actual = normalize_heading(heading.group(2))
        matches = actual == requested
        if requested == "evaluation":
            matches = matches or actual.endswith(" evaluation")
        if not matches:
            continue

        end = len(text)
        for following in headings[index + 1 :]:
            if len(following.group(1)) <= depth:
                end = following.start()
                break
        body = text[heading.end() : end].strip()
        if body:
            return body
    return None


def extract_from_h1(text: str, title: str) -> str | None:
    pat = rf"^#\s+{re.escape(title)}\s*\n(.*?)(?=^#\s+[^#]|\Z)"
    m = re.search(pat, text, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else None


def split_submission_from_evaluation(eval_body: str) -> tuple[str, str]:
    """Return (metric_part, submission_part)."""
    submission_heading = re.search(
        r"^#{2,5}\s+[*_`]*Submission(?: File| Format)?[*_`]*\s*$",
        eval_body,
        re.MULTILINE | re.IGNORECASE,
    )
    if not submission_heading:
        return eval_body.strip(), ""
    metric = eval_body[: submission_heading.start()].strip()
    submission = eval_body[submission_heading.start() :].strip()
    return metric, submission


def get_description_block(text: str) -> str:
    d = extract_h2(text, "Description")
    if d:
        return d
    ov = extract_h2(text, "Overview")
    if ov:
        inner = extract_h3(ov, "Description")
        if inner:
            return inner
        return ov
    return ""


def get_context_block(text: str) -> str:
    desc = get_description_block(text)
    ctx = extract_h3(desc, "Context") if desc else ""
    if ctx:
        return ctx
    c2 = extract_h2(text, "Context")
    return c2 or ""


def first_paragraphs(body: str, max_chars: int = 700) -> str:
    body = body.strip()
    if not body:
        return ""
    paras = [p.strip() for p in re.split(r"\n\n+", body) if p.strip()]
    out: list[str] = []
    n = 0
    for p in paras:
        if p.startswith("#"):
            continue
        out.append(p)
        n += len(p) + 2
        if n >= max_chars:
            break
        if len(out) >= 2 and n >= max_chars // 2:
            break
    return clean("\n\n".join(out), max_chars=max_chars)


def challenge_snippet(text: str) -> str:
    for label in ("Your Challenge", "The Challenge", "Challenge"):
        h = extract_h3(get_description_block(text) or text, label)
        if h:
            return clean(h, max_chars=600)
    return ""


def code_competition_notes(text: str) -> str:
    parts: list[str] = []
    for title in ("Code Requirements", "This is a Code Competition", "Kernels Requirements"):
        block = extract_h3(text, title) or extract_h2(text, title)
        if block and title not in "".join(parts):
            parts.append(block)
    cr = extract_h2(text, "Code Requirements")
    if cr and cr not in "".join(parts):
        parts.append(cr)
    if not parts:
        return ""
    return clean("\n\n".join(parts), max_chars=1200)


def strip_redundant_dataset_heading(body: str) -> str:
    body = re.sub(r"^##\s+Dataset Description\s*\n+", "", body.strip(), flags=re.MULTILINE)
    return body.strip()


def dataset_block(text: str) -> str:
    for h1_title in ("Data", "Dataset Description"):
        data_h1 = extract_from_h1(text, h1_title)
        if data_h1:
            return clean(strip_redundant_dataset_heading(data_h1), max_chars=2500)
    dd = extract_h2(text, "Dataset Description")
    if dd:
        return clean(strip_redundant_dataset_heading(dd), max_chars=2500)
    files = extract_h2(text, "Files")
    if files:
        return clean(files, max_chars=2000)
    return ""


def standalone_submission(text: str) -> str:
    for title in ("Submission File", "Submission"):
        s = extract_header_depth2_to_5(text, title) or extract_h2(text, title)
        if s:
            return clean(s, max_chars=1500)
    return ""


def drop_where_after_formula_placeholder(text: str) -> str:
    if "formula in full description" not in text:
        return text
    idx = text.find("\nWhere:\n")
    if idx != -1:
        return text[:idx].strip()
    return text


def objective_from_description(raw: str, desc_full: str) -> str:
    ch = challenge_snippet(raw)
    if ch:
        return ch
    if not desc_full:
        return ""
    paras = [p.strip() for p in re.split(r"\n\n+", desc_full) if p.strip() and not p.startswith("#")]
    task_re = re.compile(
        r"\b(predict|forecast|classify|detect|identify|segment|rank|order|reconstruct|"
        r"estimate|translate|match|tag|localize|recommend)\b",
        re.I,
    )
    for p in paras:
        if task_re.search(p) and len(p) > 40:
            return clean(p, max_chars=800)
    if paras:
        return clean(paras[0], max_chars=700)
    return ""


def _overlap_prefix(a: str, b: str, n: int = 220) -> bool:
    if not a or not b:
        return False
    ca = a.strip()[:n]
    cb = b.strip()[:n]
    return ca == cb or ca in b.strip() or cb in a.strip()


def objective_from_submission_block(submission: str) -> str:
    if not submission:
        return ""
    for line in submission.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("For each ") or s.startswith("You must predict"):
            one = re.split(r"(?<=[.!?])\s+", s, maxsplit=1)[0].strip()
            if one.endswith((".", "!", "?")):
                return clean(one, max_chars=500)
            return clean(s, max_chars=500)
    return ""


def truncate_metric_if_lets_clause_broken(text: str) -> str:
    m = re.search(r"\nLet\s+\S", text)
    if not m:
        m2 = re.search(r"\nLet\s*$", text, re.MULTILINE)
        if m2:
            return text[: m2.start()].strip()
        return text
    # "Let $S$ be" becomes "Let  be" after math strip — drop from first broken Let onward
    frag = text[m.start() : m.start() + 24]
    if re.search(r"\nLet\s+be\b", frag, re.I):
        return text[: m.start()].strip()
    return text


def generate_lite(slug: str, raw: str) -> str:
    title = slug_display_title(slug)
    desc_full = get_description_block(raw)
    context = get_context_block(raw)

    task_desc = first_paragraphs(desc_full, max_chars=750) if desc_full else ""

    eval_sec = extract_header_depth2_to_5(raw, "Evaluation") or extract_h2(raw, "Evaluation")
    metric_part, sub_from_eval = ("", "")
    if eval_sec:
        metric_part, sub_from_eval = split_submission_from_evaluation(eval_sec)
    metric_part = clean(metric_part, max_chars=1400) if metric_part else ""
    metric_part = drop_where_after_formula_placeholder(metric_part)
    metric_part = truncate_metric_if_lets_clause_broken(metric_part)

    sub_parts = [p for p in (sub_from_eval, standalone_submission(raw)) if p]
    submission = "\n\n".join(sub_parts)
    submission = clean(submission, max_chars=1800) if submission else ""
    code_notes = code_competition_notes(raw)
    if code_notes:
        submission = (submission + "\n\n**Notebook / kernel constraints (if applicable):**\n\n" + code_notes).strip()

    # Background: prefer Context; else tail of description after first para
    background = ""
    if context:
        background = clean(context, max_chars=900)
    elif desc_full:
        paras = [p.strip() for p in re.split(r"\n\n+", desc_full) if p.strip() and not p.startswith("#")]
        if len(paras) > 2:
            background = clean("\n\n".join(paras[2:]), max_chars=900)
        elif len(paras) > 1:
            background = clean(paras[1], max_chars=600)

    objective = objective_from_description(raw, desc_full or "")
    objective = clean(objective, max_chars=1000) if objective else ""
    if _overlap_prefix(objective, task_desc) or _overlap_prefix(objective, task_desc.replace("**", "")):
        alt = objective_from_submission_block(submission)
        if alt:
            objective = alt

    data = dataset_block(raw)

    lines: list[str] = [
        f"# {title} — Lite Task Description",
        "",
        "## Task description",
        "",
        task_desc or clean(desc_full, max_chars=600) or f"Kaggle competition `{slug}` (see full `description.md`).",
        "",
        "## Task objective",
        "",
        objective or "See **Task description** and **Target metric**; full `description.md` has complete rules.",
        "",
        "## Target metric (evaluation)",
        "",
        add_metric_reporting_guidance(metric_part or "See full `description.md` → Evaluation."),
        "",
        "## Brief background",
        "",
        background or "See full `description.md` → Description / Context.",
        "",
        "## Submission",
        "",
        submission or "See full `description.md` → Submission File / Code Requirements.",
        "",
        "## Dataset and construction",
        "",
        data or "See full `description.md` → Data / Dataset Description and prepared `public/` layout.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data/mlebench_all_data"),
    )
    ap.add_argument(
        "--task-list",
        type=Path,
        default=None,
        help="Newline-separated task slugs (default: <data-root>/split75.txt)",
    )
    ap.add_argument(
        "--tasks-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tasks" / "ml" / "mlebench",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing description_lite.md")
    args = ap.parse_args()

    data_root: Path = args.data_root.expanduser().resolve()
    tasks_dir: Path = args.tasks_dir.expanduser().resolve()
    task_list_path = args.task_list or (data_root / "split75.txt")
    if not task_list_path.is_file():
        raise SystemExit(f"task list not found: {task_list_path}")

    names = [
        ln.strip()
        for ln in task_list_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]

    written = 0
    skipped = 0
    errors: list[str] = []
    for slug in names:
        out_path = tasks_dir / slug / "description_lite.md"
        if out_path.is_file() and not args.force:
            skipped += 1
            continue
        src = data_root / slug / "prepared" / "public" / "description.md"
        if not src.is_file():
            errors.append(f"missing source: {src}")
            continue
        raw = src.read_text(encoding="utf-8", errors="replace")
        try:
            lite = generate_lite(slug, raw)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{slug}: {e}")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(lite, encoding="utf-8")
        written += 1

    print(f"written={written} skipped_existing={skipped} tasks={len(names)}")
    if errors:
        print("errors:")
        for e in errors:
            print(" ", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
