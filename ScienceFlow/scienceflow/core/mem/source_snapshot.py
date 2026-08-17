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

"""Source-code snapshots and code-map summaries for memory projection."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_AUTO_SNAPSHOT_PREFIX = "[auto-snapshot after successful write:"


_CODE_MAP_PREFIX = "[tool-summary code-map:"


_CHANGED_RANGE_PREFIX = "[changed-range excerpt:"


_SYMBOL_SUMMARY_PREFIX = "[symbol-level summary:"


@dataclass(frozen=True)
class SourceSymbol:
    name: str
    kind: str
    start: int
    end: int
    signature: str
    docstring: str
    body_preview: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.name}"


@dataclass(frozen=True)
class SymbolReadCoverage:
    symbol_name: str
    symbol_kind: str
    start: int
    end: int
    symbol_sha: str
    raw_id: str


def _sha256_short_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:16]


def _format_numbered_source_lines(lines: list[str], *, start: int) -> str:
    return "\n".join(f"{i:>6}|{line}" for i, line in enumerate(lines, start=start))


def _merge_read_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    s = sorted(intervals)
    out: list[tuple[int, int]] = [s[0]]
    for lo, hi in s[1:]:
        plo, phi = out[-1]
        if lo <= phi + 1:
            out[-1] = (plo, max(phi, hi))
        else:
            out.append((lo, hi))
    return out


def _range_fully_covered_by_intervals(lo: int, hi: int, merged: list[tuple[int, int]]) -> bool:
    """True if every line in [lo, hi] lies in the union of merged intervals."""
    if lo > hi:
        return True
    for a, b in merged:
        if a <= lo and hi <= b:
            return True
    return False


def _source_node_range(node: ast.AST) -> str:
    lo = int(getattr(node, "lineno", 0) or 0)
    hi = int(getattr(node, "end_lineno", 0) or lo)
    return f"{lo}-{hi}" if hi and hi != lo else str(lo or "?")


def _short_ast_expr(expr: ast.AST | None, *, max_chars: int = 24) -> str:
    if expr is None:
        return ""
    try:
        text = ast.unparse(expr)
    except Exception:
        return "..."
    text = " ".join(text.split())
    if len(text) > max_chars:
        return "..."
    return text


def _format_arg_for_code_map(arg: ast.arg, default: ast.AST | None = None) -> str:
    name = arg.arg
    if arg.annotation is not None:
        ann = _short_ast_expr(arg.annotation, max_chars=32)
        if ann:
            name = f"{name}: {ann}"
    if default is not None:
        rendered = _short_ast_expr(default)
        name = f"{name}={rendered if rendered else '...'}"
    return name


def _format_function_signature_for_code_map(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    max_params: int = 8,
) -> str:
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults: list[ast.AST | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    parts = [
        _format_arg_for_code_map(arg, default)
        for arg, default in zip(positional, defaults)
    ]
    if args.vararg is not None:
        parts.append("*" + _format_arg_for_code_map(args.vararg))
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(_format_arg_for_code_map(arg, default))
    if args.kwarg is not None:
        parts.append("**" + _format_arg_for_code_map(args.kwarg))
    if len(parts) > max_params:
        parts = parts[: max_params - 1] + ["..."]
    return f"{node.name}({','.join(parts)})"


def _collect_source_imports(tree: ast.Module) -> list[str]:
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".", 1)[0])
    out: list[str] = []
    seen: set[str] = set()
    for name in imports:
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out[:12]


def _collect_source_defs(tree: ast.Module) -> list[str]:
    defs: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _format_function_signature_for_code_map(node)
            defs.append(f"{sig}:{_source_node_range(node)}")
        elif isinstance(node, ast.ClassDef):
            methods = [
                _format_function_signature_for_code_map(m)
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            ][:5]
            suffix = f" methods={','.join(methods)}" if methods else ""
            defs.append(f"class {node.name}:{_source_node_range(node)}{suffix}")
    return defs[:16]


def _collect_argparse_flags(text: str) -> list[str]:
    flags: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"add_argument\(\s*(['\"])(--[A-Za-z0-9][A-Za-z0-9_-]*)\1", text):
        flag = m.group(2)
        if flag not in seen:
            seen.add(flag)
            flags.append(flag)
    return flags[:16]


def _collect_output_artifacts(text: str) -> list[str]:
    artifacts: list[str] = []
    seen: set[str] = set()
    patterns = (
        r"\.to_csv\(\s*(['\"])([^'\"]+\.(?:csv|tsv))\1",
        r"(?:joblib\.dump|pickle\.dump|torch\.save)\([^,\n]+,\s*(['\"])([^'\"]+\.(?:pkl|joblib|pt|pth|cbm|txt))\1",
    )
    for pat in patterns:
        for m in re.finditer(pat, text):
            val = m.group(2)
            if val and val not in seen:
                seen.add(val)
                artifacts.append(val)
    return artifacts[:12]


def _collect_model_libraries(imports: list[str], text: str) -> list[str]:
    candidates = (
        "lightgbm", "xgboost", "catboost", "sklearn", "torch",
        "tensorflow", "keras", "pandas", "numpy",
    )
    low = text.lower()
    found: list[str] = []
    for name in candidates:
        if name in imports or name in low:
            found.append(name)
    return found[:10]


def _format_class_signature_for_snapshot(node: ast.ClassDef) -> str:
    bases: list[str] = []
    for base in node.bases[:4]:
        try:
            rendered = ast.unparse(base)
        except Exception:
            rendered = "..."
        rendered = " ".join(rendered.split())
        if rendered:
            bases.append(rendered if len(rendered) <= 32 else "...")
    suffix = f"({', '.join(bases)})" if bases else ""
    return f"class {node.name}{suffix}:"


def _format_function_signature_for_snapshot(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    sig = _format_function_signature_for_code_map(node, max_params=12)
    ret = ""
    if node.returns is not None:
        rendered = _short_ast_expr(node.returns, max_chars=48)
        if rendered:
            ret = f" -> {rendered}"
    return f"{prefix}{sig}{ret}:"


def _node_docstring_first_line(node: ast.AST) -> str:
    try:
        doc = ast.get_docstring(node, clean=True)
    except Exception:
        doc = None
    if not doc:
        return ""
    return doc.strip().splitlines()[0].strip()


def _node_body_preview_lines(node: ast.AST, lines: list[str], *, max_lines: int) -> tuple[str, ...]:
    body = getattr(node, "body", None)
    if not isinstance(body, list) or not body or max_lines <= 0:
        return ()
    nodes = list(body)
    first = nodes[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(getattr(first, "value", None), ast.Constant)
        and isinstance(first.value.value, str)
    ):
        nodes = nodes[1:]
    picked: list[str] = []
    for child in nodes:
        lo = max(1, int(getattr(child, "lineno", 0) or 0))
        hi = max(lo, int(getattr(child, "end_lineno", 0) or lo))
        for line_no in range(lo, min(hi, lo + 2) + 1):
            if 1 <= line_no <= len(lines):
                raw = lines[line_no - 1].rstrip()
                if raw.strip():
                    picked.append(raw)
                    if len(picked) >= max_lines:
                        return tuple(picked)
    return tuple(picked)


def _parse_python_source_symbols(text: str, *, body_preview_lines: int = 3) -> list[SourceSymbol]:
    try:
        tree = ast.parse(text or "\n")
    except SyntaxError:
        return []
    lines = text.splitlines()
    out: list[SourceSymbol] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signature = _format_function_signature_for_snapshot(node)
            kind = "function"
        elif isinstance(node, ast.ClassDef):
            signature = _format_class_signature_for_snapshot(node)
            kind = "class"
        else:
            continue
        start = max(1, int(getattr(node, "lineno", 0) or 1))
        end = max(start, int(getattr(node, "end_lineno", 0) or start))
        out.append(
            SourceSymbol(
                name=getattr(node, "name", ""),
                kind=kind,
                start=start,
                end=end,
                signature=signature,
                docstring=_node_docstring_first_line(node),
                body_preview=_node_body_preview_lines(
                    node,
                    lines,
                    max_lines=body_preview_lines,
                ),
            )
        )
    return out


def _source_lines_for_range(lines: list[str], start: int, end: int) -> list[str]:
    if not lines:
        return []
    lo = max(1, min(start, len(lines)))
    hi = max(lo, min(end, len(lines)))
    return lines[lo - 1 : hi]


def _symbol_source_text(text: str, symbol: SourceSymbol) -> str:
    lines = text.splitlines()
    return "\n".join(_source_lines_for_range(lines, symbol.start, symbol.end))


def _symbol_sha(text: str, symbol: SourceSymbol) -> str:
    return _sha256_short_bytes(_symbol_source_text(text, symbol).encode("utf-8", errors="replace"))


def _symbol_containing_line(symbols: list[SourceSymbol], line_no: int) -> SourceSymbol | None:
    candidates = [s for s in symbols if s.start <= line_no <= s.end]
    if not candidates:
        return None
    return min(candidates, key=lambda s: (s.end - s.start, s.start))


def _line_no_for_offset(text: str, offset: int) -> int:
    if offset < 0:
        return 1
    return text[:offset].count("\n") + 1


def _locate_edit_target_line(
    *,
    current_text: str,
    previous_text: str | None,
    args: dict[str, Any] | None,
) -> int | None:
    if not args:
        return None
    new_s = args.get("new_str")
    if isinstance(new_s, str) and new_s:
        idx = current_text.find(new_s)
        if idx >= 0:
            return _line_no_for_offset(current_text, idx)
    old_s = args.get("old_str")
    if previous_text and isinstance(old_s, str) and old_s:
        idx = previous_text.find(old_s)
        if idx >= 0:
            return _line_no_for_offset(previous_text, idx)
    return None


def _choose_changed_symbol(
    *,
    current_text: str,
    previous_text: str | None,
    symbols: list[SourceSymbol],
    tool_name: str | None,
    args: dict[str, Any] | None,
) -> tuple[SourceSymbol | None, int | None]:
    if not symbols:
        return None, None
    if tool_name == "edit":
        line_no = _locate_edit_target_line(
            current_text=current_text,
            previous_text=previous_text,
            args=args,
        )
        if line_no is not None:
            return _symbol_containing_line(symbols, line_no), line_no
    if previous_text:
        previous_symbols = {
            s.key: _symbol_source_text(previous_text, s)
            for s in _parse_python_source_symbols(previous_text)
        }
        ranked: list[tuple[int, int, SourceSymbol]] = []
        for sym in symbols:
            cur_body = _symbol_source_text(current_text, sym)
            prev_body = previous_symbols.get(sym.key)
            if prev_body != cur_body:
                diff_size = abs(len(cur_body) - len(prev_body or "")) + len(cur_body)
                ranked.append((diff_size, sym.end - sym.start, sym))
        if ranked:
            ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return ranked[0][2], ranked[0][2].start
    largest = max(symbols, key=lambda s: (s.end - s.start, s.start))
    return largest, largest.start


def _build_changed_range_excerpt(
    rel: str,
    text: str,
    symbol: SourceSymbol | None,
    *,
    target_line: int | None,
    context_lines: int,
    max_chars: int,
) -> str:
    if max_chars <= 0:
        return ""
    lines = text.splitlines()
    if symbol is None:
        if target_line is None:
            return ""
        lo = max(1, target_line - max(1, context_lines))
        hi = min(len(lines), target_line + max(1, context_lines))
        block = "\n".join(
            [
                f"{_CHANGED_RANGE_PREFIX} {rel} lines {lo}-{hi}]",
                _format_source_excerpt("changed context", _source_lines_for_range(lines, lo, hi), start=lo),
            ],
        )
        return block[:max_chars].rstrip()

    full_lines = _source_lines_for_range(lines, symbol.start, symbol.end)
    header = (
        f"{_CHANGED_RANGE_PREFIX} {rel}::{symbol.name} "
        f"lines {symbol.start}-{symbol.end}]"
    )
    full = header + "\n" + _format_source_excerpt(f"{symbol.name} full", full_lines, start=symbol.start)
    if len(full) <= max_chars:
        return full

    tl = target_line if target_line is not None else symbol.start
    lo = max(symbol.start, tl - max(1, context_lines))
    hi = min(symbol.end, tl + max(1, context_lines))
    excerpt_lines = _source_lines_for_range(lines, lo, hi)
    parts = [
        header,
        (
            f"[changed symbol body truncated: full range {symbol.start}-{symbol.end}; "
            f"showing lines {lo}-{hi}]"
        ),
        symbol.signature,
    ]
    if symbol.docstring:
        parts.append(f'    """{symbol.docstring}"""')
    parts.append(_format_source_excerpt(f"{symbol.name} around change", excerpt_lines, start=lo))
    block = "\n".join(parts)
    if len(block) > max_chars:
        block = block[: max(1, max_chars - 28)].rstrip() + "\n...[changed excerpt truncated]"
    return block


def _build_symbol_level_summary(
    rel: str,
    symbols: list[SourceSymbol],
    *,
    max_chars: int,
) -> str:
    if max_chars <= 0 or not symbols:
        return ""
    parts = [f"{_SYMBOL_SUMMARY_PREFIX} {rel}]"]
    for sym in symbols:
        parts.append(f"{sym.signature}  # lines {sym.start}-{sym.end}")
        if sym.docstring:
            parts.append(f'    """{sym.docstring}"""')
        for line in sym.body_preview:
            parts.append(line)
        omitted = max(0, sym.end - sym.start + 1 - 1 - len(sym.body_preview))
        if omitted:
            parts.append(f"    # ... [{omitted} lines omitted, see code-map range {sym.start}-{sym.end}]")
        parts.append("")
        if len("\n".join(parts)) > max_chars:
            parts[-1] = "...[symbol summary truncated]"
            break
    block = "\n".join(parts).rstrip()
    if len(block) > max_chars:
        block = block[: max(1, max_chars - 28)].rstrip() + "\n...[symbol summary truncated]"
    return block


def _build_source_code_map(rel: str, text: str, *, max_chars: int = 1600) -> str:
    """Small deterministic source map for tool-result memory summaries."""
    lines = text.splitlines()
    short = _sha256_short_bytes(text.encode("utf-8", errors="replace"))
    out: list[str] = [
        f"{_CODE_MAP_PREFIX} {rel}]",
        f"- lines: {len(lines)}",
        f"- sha256:{short}",
    ]
    try:
        tree = ast.parse(text or "\n")
    except SyntaxError:
        out.append("- parse: syntax-error")
        block = "\n".join(out)
        return block[:max_chars]
    imports = _collect_source_imports(tree)
    defs = _collect_source_defs(tree)
    args = _collect_argparse_flags(text)
    artifacts = _collect_output_artifacts(text)
    libs = _collect_model_libraries(imports, text)
    if imports:
        out.append("- imports: " + ", ".join(imports))
    if libs:
        out.append("- model/libs: " + ", ".join(libs))
    if defs:
        out.append("- top-level defs: " + "; ".join(defs))
    if args:
        out.append("- argparse: " + ", ".join(args))
    if artifacts:
        out.append("- output artifacts: " + ", ".join(artifacts))
    block = "\n".join(out)
    if len(block) > max_chars:
        block = block[: max_chars - 32].rstrip() + "\n...[code-map truncated]"
    return block


def _format_source_excerpt(
    label: str,
    lines: list[str],
    *,
    start: int,
) -> str:
    return f"[source-excerpt: {label}]\n" + _format_numbered_source_lines(lines, start=start)


def _format_line_ranges(ranges: list[tuple[int, int]], *, max_ranges: int = 8) -> str:
    if not ranges:
        return "(none)"
    parts = [
        str(lo) if lo == hi else f"{lo}-{hi}"
        for lo, hi in ranges[:max_ranges]
    ]
    if len(ranges) > max_ranges:
        parts.append("...")
    return ", ".join(parts)


def _requested_read_range_for_file(args: dict[str, Any], total_lines: int) -> tuple[int, int]:
    start0 = max(0, int(args.get("offset") or 1) - 1)
    lim = int(args.get("limit") or 200)
    if lim <= 0:
        return 1, max(1, total_lines)
    lo = start0 + 1
    hi = min(max(1, total_lines), start0 + lim)
    return lo, hi


def _build_redundant_read_coverage_summary(
    rel: str,
    *,
    sha: str,
    requested: tuple[int, int],
    covered_ranges: list[tuple[int, int]],
) -> str:
    lo, hi = requested
    return "\n".join(
        [
            f"[read-coverage summary: {rel}]",
            f"- sha256: {sha}",
            f"- requested lines: {lo}-{hi}",
            f"- already covered ranges: {_format_line_ranges(covered_ranges)}",
            "- stored body: omitted because this same file range was already present in memory",
        ],
    )


def _build_redundant_symbol_read_coverage_summary(
    rel: str,
    *,
    requested: tuple[int, int],
    coverage: SymbolReadCoverage,
) -> str:
    lo, hi = requested
    raw = coverage.raw_id or "(unknown)"
    return "\n".join(
        [
            f"[read-coverage summary: {rel}::{coverage.symbol_name}]",
            f"- symbol: {coverage.symbol_kind} {coverage.symbol_name}",
            f"- symbol sha256: {coverage.symbol_sha}",
            f"- requested lines: {lo}-{hi}",
            f"- covered symbol range: {coverage.start}-{coverage.end}",
            f"- previous full read raw_id: {raw}",
            "- stored body: omitted because this symbol range was already present in memory",
        ],
    )


def _tool_output_raw_id_from_content(content: str) -> str:
    m = re.search(r"\[tool-output raw_id=([^\s\]]+)", content or "")
    return m.group(1) if m else ""


def _numbered_source_ranges_in_text(text: str) -> list[tuple[int, int]]:
    nums: list[int] = []
    for line in (text or "").splitlines():
        m = re.match(r"^\s*(\d+)\|", line)
        if m:
            nums.append(int(m.group(1)))
    if not nums:
        return []
    nums = sorted(set(nums))
    ranges: list[tuple[int, int]] = []
    lo = hi = nums[0]
    for n in nums[1:]:
        if n == hi + 1:
            hi = n
        else:
            ranges.append((lo, hi))
            lo = hi = n
    ranges.append((lo, hi))
    return ranges


def _build_write_auto_snapshot_block(
    rel: str,
    abs_path: Path,
    *,
    max_lines: int,
    max_chars: int,
    previous_text: str | None = None,
    tool_name: str | None = None,
    args: dict[str, Any] | None = None,
    changed_context_lines: int = 10,
    symbol_body_lines: int = 3,
) -> str:
    """Line-numbered snapshot of *abs_path* for appending to write tool memory."""
    text = abs_path.read_text(encoding="utf-8", errors="replace")
    raw = text.encode("utf-8")
    short = _sha256_short_bytes(raw)
    all_lines = text.splitlines()
    n = len(all_lines)
    code_map = _build_source_code_map(rel, text)
    intro = (
        f"{_AUTO_SNAPSHOT_PREFIX} {rel}]\n"
        f"[current `{rel}` snapshot summary; deterministic tool-result memory]\n"
    )
    if n == 0:
        return intro + f"[{rel}: 0 lines total, sha256~{short}, empty file]\n" + code_map

    meta = f"[{rel}: {n} lines total, sha256~{short}]"
    base = intro + meta + "\n" + code_map
    if len(base) >= max_chars:
        return base[: max(1, max_chars - 28)].rstrip() + "\n...[summary truncated]"

    symbols = _parse_python_source_symbols(text, body_preview_lines=symbol_body_lines)
    if PurePosixPath(rel).suffix.lower() == ".py" and symbols:
        sym, target_line = _choose_changed_symbol(
            current_text=text,
            previous_text=previous_text,
            symbols=symbols,
            tool_name=tool_name,
            args=args,
        )
        remaining = max_chars - len(base) - 2
        changed_budget = min(max(0, remaining), 2400)
        changed = _build_changed_range_excerpt(
            rel,
            text,
            sym,
            target_line=target_line,
            context_lines=changed_context_lines,
            max_chars=changed_budget,
        )
        parts: list[str] = [base]
        if changed:
            parts.append(changed)
        used = len("\n\n".join(parts))
        summary_budget = max(0, max_chars - used - 2)
        summary = _build_symbol_level_summary(
            rel,
            symbols,
            max_chars=summary_budget,
        )
        if summary:
            parts.append(summary)
        body = "\n\n".join(parts)
        if len(body) > max_chars:
            body = body[: max(1, max_chars - 28)].rstrip() + "\n...[snapshot truncated]"
        return body

    if n <= max_lines and len(base) + len(text) + 64 <= max_chars:
        body = _format_source_excerpt(f"{rel} full", all_lines, start=1)
        block = base + "\n" + body
        if len(block) > max_chars:
            block = block[: max(1, max_chars - 28)].rstrip() + "\n...[truncated]"
        return block

    head_n = max(1, min(n, max_lines // 2))
    tail_n = max(0, min(n - head_n, max_lines - head_n))
    head_lines = all_lines[:head_n]
    tail_lines = all_lines[n - tail_n :] if tail_n else []
    omitted = n - len(head_lines) - len(tail_lines)
    parts: list[str] = [
        base,
        f"[{rel}: source excerpts truncated for memory budget]",
        _format_source_excerpt("head", head_lines, start=1),
    ]
    if tail_lines and omitted > 0:
        parts.append(f"\n... ({omitted} middle lines omitted) ...\n")
        parts.append(_format_source_excerpt("tail", tail_lines, start=n - len(tail_lines) + 1))
    body = "\n".join(parts)
    while len(body) > max_chars and (head_n > 8 or tail_n > 8):
        head_n = max(8, head_n // 2)
        tail_n = max(8, tail_n // 2) if tail_n else 0
        head_lines = all_lines[:head_n]
        tail_lines = all_lines[n - tail_n :] if tail_n else []
        omitted = n - len(head_lines) - len(tail_lines)
        parts = [
            base,
            f"[{rel}: source excerpts truncated for memory budget]",
            _format_source_excerpt("head", head_lines, start=1),
        ]
        if tail_lines and omitted > 0:
            parts.append(f"\n... ({omitted} middle lines omitted) ...\n")
            parts.append(_format_source_excerpt("tail", tail_lines, start=n - len(tail_lines) + 1))
        body = "\n".join(parts)
    if len(body) > max_chars:
        keep = max(0, max_chars - len(base) - 64)
        body = base + "\n[source-excerpt omitted: summary budget exhausted]"
        if keep > 0 and len(body) > max_chars:
            body = body[:max_chars]
    return body


def extract_write_auto_snapshot_block(content: str) -> str | None:
    """Return the auto-snapshot suffix from a write tool memory string, if present."""
    if not content or _AUTO_SNAPSHOT_PREFIX not in content:
        return None
    idx = content.find(_AUTO_SNAPSHOT_PREFIX)
    if idx < 0:
        return None
    return content[idx:].strip()
