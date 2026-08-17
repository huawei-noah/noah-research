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

"""
Generate tasks/ml/mlebench/<slug>/description_lite.md in the same structured style as AI4Code:
Task description / Task objective (Input/Output) / Target metric / Brief background /
Submission (schema + fence) / Dataset bullets.

Skips slug AI4Code (preserve hand-written file). Reads mlebench prepared/public/description.md.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Reuse parsers from gen_description_lite
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_description_lite as g  # noqa: E402


SKIP_SLUGS = {"ai4code"}
MAX_LINES = 30
def _overlap_prefix(a: str, b: str, n: int = 200) -> bool:
    if not a or not b:
        return False
    ca, cb = a.strip()[:n], b.strip()[:n]
    return ca == cb or ca in b.strip() or cb in a.strip()


def _shorten(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars + 1]
    sp = cut.rfind(" ")
    if sp > max_chars // 2:
        cut = cut[:sp]
    return cut.rstrip(",;:") + " …"


def _first_code_fence(text: str) -> tuple[str, list[str]]:
    m = re.search(r"```(?:[^\n]*)\n(.*?)```", text, flags=re.DOTALL)
    if not m:
        return "", []
    inner = m.group(1).strip()
    lines = [ln.rstrip() for ln in inner.splitlines() if ln.strip()]
    return (lines[0] if lines else ""), lines[:4]


def _submission_filename(text: str) -> str:
    if re.search(r"submission\.csv", text, re.I):
        return "`submission.csv`"
    if re.search(r"sample_submission", text, re.I):
        return "`submission.csv` (match `sample_submission.csv`)"
    return "`submission.csv`"


def _kernels_or_notebook_note(raw: str) -> str:
    if re.search(r"Kernels-only|kernel submission", raw, re.I):
        return "Submit via Kaggle **Kernels-only**; see kernel rules."
    if re.search(r"Code Competition|through Notebooks", raw, re.I):
        return "Submit via Kaggle **Notebook / Code competition**."
    return ""


def _dataset_bullets(data: str, max_n: int = 4) -> list[str]:
    if not data:
        return ["See prepared `public/` and full `description.md` for file layout."]
    bullets: list[str] = []
    for line in data.splitlines():
        s = line.strip()
        if s.startswith("- ") and len(s) > 3:
            item = re.sub(r"^\-\s+", "", s)
            item = g.clean(item, max_chars=220)
            if item:
                bullets.append(item)
        if len(bullets) >= max_n:
            break
    if len(bullets) < 2:
        # fallback: split on **name** — description
        parts = re.split(r"(?<=\.)\s+", g.clean(data, max_chars=1200))
        for p in parts[:max_n]:
            p = p.strip()
            if len(p) > 15:
                bullets.append(_shorten(p, 200))
    return bullets[:max_n] if bullets else ["See full `description.md` → Data."]


def _input_output_bullets(
    submission: str,
    objective_sentence: str,
    slug: str,
) -> tuple[str, str]:
    inp = ""
    out = ""
    for line in submission.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("For each ") or s.startswith("You must"):
            # first sentence only for output hint
            one = re.split(r"(?<=[.!?])\s+", s, maxsplit=1)[0].strip()
            out = g.clean(one, max_chars=380)
            m = re.search(
                r"For each\s+`?([^`\n]+)`?\s+in the test set",
                s,
                re.I,
            )
            if m:
                inp = f"Each test row / notebook identified by `{m.group(1).strip()}` (and any features in released `test` / `public` data)."
            else:
                inp = "Each test sample as defined by the competition `test` split and `sample_submission.csv` rows."
            break
    if not out:
        out = _shorten(objective_sentence or f"Predict targets for `{slug}` test set per official rules.", 380)
    if not inp:
        inp = "Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train."
    return inp, out


def build_document(slug: str, raw: str) -> str:
    title = g.slug_display_title(slug)
    desc_full = g.get_description_block(raw)
    task_desc = ""
    if desc_full:
        paras = [
            p.strip()
            for p in re.split(r"\n\n+", desc_full)
            if p.strip() and not p.strip().startswith("#")
        ]
        for p in paras:
            if re.search(r"\b(predict|forecast|classify|detect|identify|segment|tag|rank)\b", p, re.I) and len(p) > 50:
                task_desc = g.clean(p, max_chars=620)
                break
        if not task_desc:
            task_desc = g.first_paragraphs(desc_full, max_chars=620)
    task_desc = g.clean(task_desc, max_chars=620)
    # Drop obvious fluff lines
    task_desc = re.sub(
        r"^(Who's a good dog\?|Think you can use your data science).*$",
        "",
        task_desc,
        flags=re.MULTILINE,
    ).strip()
    if not task_desc.strip():
        task_desc = _shorten(g.clean(desc_full or "", max_chars=400), 400) or f"Kaggle competition `{slug}`."

    eval_sec = g.extract_header_depth2_to_5(raw, "Evaluation") or g.extract_h2(raw, "Evaluation")
    metric_part, sub_from_eval = ("", "")
    if eval_sec:
        metric_part, sub_from_eval = g.split_submission_from_evaluation(eval_sec)
    metric_part = g.clean(metric_part, max_chars=900) if metric_part else ""
    metric_part = g.drop_where_after_formula_placeholder(metric_part)
    metric_part = g.truncate_metric_if_lets_clause_broken(metric_part)
    low_m = metric_part.lower()
    if "task consists" in low_m and "metric" not in low_m[:140]:
        for sent in re.split(r"(?<=[.!?])\s+", metric_part):
            s2 = sent.strip()
            if re.search(r"\b(metric|evaluated|score|loss|auc|map|error|precision|recall|f1)\b", s2, re.I):
                metric_part = s2
                break
    metric_part = _shorten(metric_part, 320)

    sub_parts = [p for p in (sub_from_eval, g.standalone_submission(raw)) if p]
    submission_raw = "\n\n".join(sub_parts)
    submission_raw = g.clean(submission_raw, max_chars=1600) if submission_raw else ""

    header_line, fence_lines = _first_code_fence(submission_raw)
    schema = header_line if header_line else "id,prediction"
    if len(schema) > 55 or schema.count(",") > 2:
        first = schema.split(",")[0].strip()
        schema = f"{first}, … (remaining columns per sample_submission.csv)"
    fence_lines = fence_lines[:3]
    fence_body = "\n".join(fence_lines) if fence_lines else schema

    obj_sentence = g.objective_from_description(raw, desc_full or "")
    obj_sentence = g.clean(obj_sentence, max_chars=500) if obj_sentence else ""
    if _overlap_prefix(obj_sentence, task_desc):
        alt = g.objective_from_submission_block(submission_raw)
        if alt:
            obj_sentence = alt

    inp_b, out_b = _input_output_bullets(submission_raw, obj_sentence, slug)

    ctx = g.get_context_block(raw)
    if ctx:
        bg = _shorten(g.clean(ctx, max_chars=400), 200)
    else:
        bg = "Hosted benchmark task; see full `description.md` for citations and organizers."
    bg = _shorten(bg, 220)

    data = g.dataset_block(raw)
    ds_bullets = _dataset_bullets(data)

    path_note = _kernels_or_notebook_note(raw)
    fname = _submission_filename(submission_raw + raw)

    td = _shorten(task_desc, 320)
    mp = g.add_metric_reporting_guidance(_shorten(metric_part, 240))

    lines: list[str] = [
        f"# {title} — Lite Task Description",
        "",
        "## Task description",
        td,
        "",
        "## Task objective",
        f"- **Input:** {_shorten(inp_b, 200)}",
        f"- **Output:** {_shorten(out_b, 220)}",
        "",
        "## Target metric (evaluation)",
        mp,
        "",
        "## Brief background",
        _shorten(bg, 180),
        "",
        "## Submission",
    ]
    file_line = f"- **File:** {fname}"
    if path_note:
        file_line += f" — {path_note.rstrip('.')}."
    else:
        file_line += "."
    sub_lines = [
        file_line,
        f"- **Schema:** {schema}; one row per test key (see `sample_submission.csv`).",
        "```",
    ]
    lines.extend(sub_lines)
    lines.extend(fence_body.splitlines())
    lines.append("```")
    lines.append("")
    lines.append("## Dataset and construction")
    for b in ds_bullets:
        lines.append(f"- {b}")

    out = "\n".join(lines).rstrip() + "\n"

    def trim_to_max(text: str) -> str:
        tl = text.splitlines()
        guard = 0
        while len(tl) > MAX_LINES and guard < 200:
            guard += 1
            ds_hdr = next((i for i, line in enumerate(tl) if line == "## Dataset and construction"), -1)
            if ds_hdr >= 0:
                last_bullet = -1
                for i in range(len(tl) - 1, ds_hdr, -1):
                    if tl[i].startswith("- "):
                        last_bullet = i
                        break
                if last_bullet > ds_hdr:
                    tl.pop(last_bullet)
                    continue
            candidates = [
                i
                for i, ln in enumerate(tl)
                if ln
                and not ln.startswith("#")
                and not ln.startswith("```")
                and not ln.startswith("- **Input")
                and not ln.startswith("- **Output")
                and not ln.startswith("- **File")
                and not ln.startswith("- **Schema")
                and not ln.startswith("- ")
            ]
            if not candidates:
                break
            j = max(candidates, key=lambda k: len(tl[k]))
            new_val = _shorten(tl[j], max(40, len(tl[j]) * 2 // 3))
            if new_val == tl[j] or len(new_val) < 20:
                tl.pop(j)
            else:
                tl[j] = new_val
        return "\n".join(tl).rstrip() + "\n"

    out = trim_to_max(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("./data/mlebench_all_data"))
    ap.add_argument(
        "--tasks-dir",
        type=Path,
        default=_REPO_ROOT / "tasks" / "ml" / "mlebench",
    )
    ap.add_argument(
        "--task-list",
        type=Path,
        default=None,
        help="Default: <data-root>/split75.txt",
    )
    args = ap.parse_args()

    data_root = args.data_root.expanduser().resolve()
    tasks_dir = args.tasks_dir.expanduser().resolve()
    list_path = args.task_list or (data_root / "split75.txt")
    if not list_path.is_file():
        raise SystemExit(f"missing task list: {list_path}")

    names = [
        ln.strip()
        for ln in list_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]

    written = 0
    skipped = 0
    errors: list[str] = []
    for slug in names:
        if slug.lower() in SKIP_SLUGS:
            skipped += 1
            continue
        src = data_root / slug / "prepared" / "public" / "description.md"
        if not src.is_file():
            errors.append(f"missing {src}")
            continue
        raw = src.read_text(encoding="utf-8", errors="replace")
        try:
            doc = build_document(slug, raw)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{slug}: {e}")
            continue
        if len(doc.splitlines()) > MAX_LINES:
            errors.append(f"{slug}: still {len(doc.splitlines())} lines after trim")
            continue
        out_path = tasks_dir / slug / "description_lite.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(doc, encoding="utf-8")
        written += 1

    print(f"written={written} skipped_AI4Code={skipped} total_listed={len(names)}")
    if errors:
        for e in errors[:20]:
            print("error:", e)
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
