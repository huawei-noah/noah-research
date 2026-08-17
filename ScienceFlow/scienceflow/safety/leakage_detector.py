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

"""Validation-set leakage detection for ML solution code.

Four-phase heuristic analysis that detects whether ``solution.py`` trains
on holdout/validation data. Designed as a safety guard for the
NaiveLongHorizon pipeline (and reusable by any solver).

Phase 1 — Regex: explicit train+val variable names in combining APIs.
Phase 2 — Regex: ``.fit()`` directly on validation-named data.
Phase 3 — Regex: validation file read + generic row-combining / combined-var training.
Phase 4 — AST:   taint tracking from validation file reads through assignments.

Design goal: high precision; allow false negatives. Row-wise pd.concat of
train+val for fitting encoders/scalers (e.g. LabelEncoder) is treated as safe.
"""

from __future__ import annotations

import ast
import re

# =====================================================================
# Phase 1–3: regex-based heuristics
# =====================================================================

_FIT_KWARG_SKIP = ("validation_data", "validation_split", "eval_set")

_TRAIN_TOK = r"(?:\bX_train\b|\by_train\b|\btrain_df\b|\btrain_data\b|\bdf_train\b|\bdata_train\b|_train\b)"
_VAL_TOK = r"(?:\bX_val\b|\by_val\b|\bval_df\b|\bval_data\b|\bdf_val\b|\bdata_val\b|\bvalidation\b|_val\b)"
_RE_HAS_TRAIN = re.compile(_TRAIN_TOK, re.IGNORECASE)
_RE_HAS_VAL = re.compile(_VAL_TOK, re.IGNORECASE)


def _bracket_pair(api_re: str):
    """Build (train->val, val->train) bracket-enclosed combining regexes."""
    return (
        re.compile(
            rf"{api_re}\s*\(\s*\[[^\]]*{_TRAIN_TOK}[^\]]*{_VAL_TOK}[^\]]*\]",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            rf"{api_re}\s*\(\s*\[[^\]]*{_VAL_TOK}[^\]]*{_TRAIN_TOK}[^\]]*\]",
            re.IGNORECASE | re.DOTALL,
        ),
    )


_BRACKET_CHECKS: list[tuple[str, re.Pattern, re.Pattern]] = [
    ("pd.concat", *_bracket_pair(r"pd\.concat")),
    (
        "np.vstack/concatenate",
        *_bracket_pair(r"np\.(?:vstack|hstack|concatenate|column_stack|row_stack|stack)"),
    ),
    ("torch.cat", *_bracket_pair(r"torch\.cat")),
]

_RE_MERGE_TV = re.compile(
    rf"pd\.merge\s*\([^)]*{_TRAIN_TOK}[^)]*{_VAL_TOK}", re.IGNORECASE | re.DOTALL,
)
_RE_MERGE_VT = re.compile(
    rf"pd\.merge\s*\([^)]*{_VAL_TOK}[^)]*{_TRAIN_TOK}", re.IGNORECASE | re.DOTALL,
)

_LINE_COMBINE_RE = re.compile(
    r"\.append\s*\(|"
    r"\[\s*\*\w+.*,\s*\*\w+",
    re.IGNORECASE,
)

_RE_FIT_VAL_PATH = re.compile(
    r"\.fit\s*\(\s*[^)]*"
    r"(?:dataset[/\\][^)\s,]*validation|\bvalidation_(?:csv|parquet|path)\b|"
    r"\bdf_val\b|\bdata_val\b|\btrain_val\b)",
    re.IGNORECASE | re.DOTALL,
)

_RE_READS_VAL_FILE = re.compile(
    r"(?:read_csv|read_parquet|read_table|read_excel|read_feather|"
    r"np\.load(?:txt)?|np\.genfromtxt|"
    r"open)\s*\([^)]*?"
    r"(?:dataset\s*[/\\]\s*[^)]*?validation|['\"][\w/\\]*validation[\w]*\."
    r"(?:csv|parquet|tsv|pkl|npy|npz))",
    re.IGNORECASE,
)

# Deliberately excludes .append() — too common for generic list ops -> high FP.
_RE_ANY_ROW_COMBINE = re.compile(
    r"pd\.concat\s*\(\s*\[|"
    r"np\.(?:vstack|concatenate|row_stack|stack)\s*\(\s*\[|"
    r"torch\.cat\s*\(\s*\[",
    re.IGNORECASE,
)

_RE_FIT_COMBINED_VAR = re.compile(
    r"\.(?:fit|train)\s*\(\s*"
    r"(?:X_full|X_all|X_combined|combined|full_data|all_data|data_full|"
    r"X_both|X_total|total_data|data_all|data_combined)\b",
    re.IGNORECASE,
)

_RE_ASSIGN_COMBINED = re.compile(
    r"(?:X_full|X_all|X_combined|combined|full_data|all_data|data_full|"
    r"X_both|X_total|total_data|data_all|data_combined)\s*=\s*"
    r"(?:np\.(?:vstack|concatenate|hstack|stack|row_stack|column_stack)|"
    r"pd\.concat|torch\.cat)",
    re.IGNORECASE,
)

# =====================================================================
# Phase 4: AST taint tracking
# =====================================================================

_AST_FILE_READ_FUNCS = frozenset({
    "read_csv", "read_parquet", "read_table", "read_excel", "read_feather",
    "read_json", "read_pickle", "read_hdf",
    "load", "loadtxt", "genfromtxt",
})

_AST_COMBINE_FUNCS = frozenset({
    "concat", "merge",
    "vstack", "hstack", "concatenate", "column_stack", "row_stack", "stack",
    "cat",
})

_AST_TRAIN_METHODS = frozenset({"fit", "partial_fit"})

_AST_LEGIT_FIT_KW = frozenset({"validation_data", "validation_split", "eval_set"})

# Preprocessor / encoder classes whose .fit() on combined data is NOT leakage —
# they only learn value mappings or statistics, not predictive relationships.
_SAFE_PREPROCESSOR_NAMES = frozenset({
    # Encoders
    "LabelEncoder", "OrdinalEncoder", "OneHotEncoder", "LabelBinarizer",
    "MultiLabelBinarizer", "TargetEncoder",
    # Scalers
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
    "Normalizer", "QuantileTransformer", "PowerTransformer",
    # Imputers
    "SimpleImputer", "KNNImputer", "IterativeImputer", "MissingIndicator",
    # Feature transforms
    "PolynomialFeatures", "SplineTransformer", "Binarizer",
    "KBinsDiscretizer", "FunctionTransformer",
    # Pipeline / meta (fit delegates to safe transforms inside)
    "ColumnTransformer", "Pipeline",
    # Tokenizers / vectorizers
    "CountVectorizer", "TfidfVectorizer", "HashingVectorizer",
    "TfidfTransformer",
})


def _ast_func_leaf(call: ast.Call) -> str:
    """Leaf function/method name of a Call node."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def _ast_has_val_literal(node: ast.AST) -> bool:
    """True if any string constant in the subtree contains ``'validation'``."""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if "validation" in child.value.lower():
                return True
    return False


def _ast_names(node: ast.AST) -> set[str]:
    """All ``ast.Name.id`` values reachable from *node*."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _ast_assign_lhs(node: ast.Assign) -> set[str]:
    """Simple ``Name`` targets (handles tuple/list unpacking)."""
    out: set[str] = set()
    for t in node.targets:
        if isinstance(t, ast.Name):
            out.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            out.update(e.id for e in t.elts if isinstance(e, ast.Name))
    return out


def _ast_skip_tuple_unpack_propagation(asgn: ast.Assign) -> bool:
    """Skip taint propagation when RHS is a non-combining call and LHS unpacks multiple names.

    Otherwise ``X_train, X_val, ... = preprocess_data(train_df, val_df, ...)`` incorrectly
    marks *all* unpacked names (including ``X_train``) as validation-tainted because
    ``val_df`` appears in the call arguments.

    Combining APIs (``pd.concat``, ``merge``, …) and file reads still propagate so
    row-combining and read-based flows keep working.
    """
    if not isinstance(asgn.value, ast.Call):
        return False
    lhs = _ast_assign_lhs(asgn)
    if len(lhs) <= 1:
        return False
    fn = _ast_func_leaf(asgn.value)
    if fn in _AST_COMBINE_FUNCS or fn in _AST_FILE_READ_FUNCS:
        return False
    return True


def _ast_detect_val_leakage(code: str) -> list[str]:
    """AST-based validation-file taint tracking.

    1. Seed tainted vars from validation file reads.
    2. Propagate taint through assignment chains (fixed-point, capped).
    3. Track safe preprocessor instances (LabelEncoder, StandardScaler, …).
    4. Detect if tainted vars flow into combining or training calls,
       but suppress when the data only feeds safe preprocessor .fit().

    Returns human-readable issue strings.  Gracefully returns [] on
    unparseable code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    tainted: set[str] = set()
    assigns: list[ast.Assign] = []
    safe_vars: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        assigns.append(node)
        if isinstance(node.value, ast.Call):
            func_name = _ast_func_leaf(node.value)
            if func_name in _AST_FILE_READ_FUNCS:
                if _ast_has_val_literal(node.value):
                    tainted |= _ast_assign_lhs(node)
            if func_name in _SAFE_PREPROCESSOR_NAMES:
                safe_vars |= _ast_assign_lhs(node)

    if not tainted:
        return []

    _MAX_TAINT = 50
    for _ in range(10):
        prev = len(tainted)
        for asgn in assigns:
            if _ast_skip_tuple_unpack_propagation(asgn):
                continue
            if _ast_names(asgn.value) & tainted:
                tainted |= _ast_assign_lhs(asgn)
            if len(tainted) > _MAX_TAINT:
                break
        if len(tainted) == prev or len(tainted) > _MAX_TAINT:
            break

    issues: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = _ast_func_leaf(node)

        if fname in _AST_COMBINE_FUNCS:
            has_tainted_elem = False
            has_clean_elem = False
            tainted_found: set[str] = set()

            if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                for elt in node.args[0].elts:
                    elt_names = _ast_names(elt)
                    if elt_names & tainted:
                        has_tainted_elem = True
                        tainted_found |= elt_names & tainted
                    elif elt_names:
                        has_clean_elem = True
            elif fname == "merge" and len(node.args) >= 2:
                a1 = _ast_names(node.args[0])
                a2 = _ast_names(node.args[1])
                if a1 & tainted:
                    has_tainted_elem = True
                    tainted_found |= a1 & tainted
                if a2 & tainted:
                    has_tainted_elem = True
                    tainted_found |= a2 & tainted
                if (a1 and not a1 & tainted) or (a2 and not a2 & tainted):
                    has_clean_elem = True

            if has_tainted_elem and has_clean_elem:
                if _combine_feeds_only_safe(node, assigns, tree, safe_vars):
                    continue
                issues.append(
                    f"[AST] Validation-sourced variable(s) {tainted_found} combined "
                    f"with other data via {fname}()"
                )
                break

        if fname in _AST_TRAIN_METHODS:
            kw_set = {kw.arg for kw in node.keywords}
            if kw_set & _AST_LEGIT_FIT_KW:
                continue
            if _is_safe_fit_receiver(node, safe_vars):
                continue
            if node.args:
                first_names = _ast_names(node.args[0])
                if first_names & tainted:
                    issues.append(
                        f"[AST] .{fname}() receives validation-sourced data "
                        f"({first_names & tainted}) as training input"
                    )
                    break

    return issues


def _is_safe_fit_receiver(call: ast.Call, safe_vars: set[str]) -> bool:
    """True if ``.fit()`` / ``.partial_fit()`` is called on a known safe preprocessor."""
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        return call.func.value.id in safe_vars
    return False


def _combine_feeds_only_safe(
    combine_call: ast.Call,
    assigns: list[ast.Assign],
    tree: ast.AST,
    safe_vars: set[str],
) -> bool:
    """True when the result of a combining call only flows to safe preprocessor ``.fit()``."""
    # Case 1: combine is directly inside a safe .fit(), e.g. le.fit(pd.concat([...]))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = _ast_func_leaf(node)
        if fname not in _AST_TRAIN_METHODS:
            continue
        for arg in node.args:
            for child in ast.walk(arg):
                if child is combine_call:
                    return _is_safe_fit_receiver(node, safe_vars)

    # Case 2: combine assigned to variable(s), then used downstream
    result_vars: set[str] = set()
    for asgn in assigns:
        for child in ast.walk(asgn.value):
            if child is combine_call:
                result_vars |= _ast_assign_lhs(asgn)
                break
    if not result_vars:
        return False

    found_fit_using_result = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = _ast_func_leaf(node)
        if fname not in _AST_TRAIN_METHODS:
            continue
        all_arg_names: set[str] = set()
        for arg in node.args:
            all_arg_names |= _ast_names(arg)
        if not (all_arg_names & result_vars):
            continue
        found_fit_using_result = True
        if not _is_safe_fit_receiver(node, safe_vars):
            return False
    return found_fit_using_result


# =====================================================================
# Helpers
# =====================================================================


def _phase3_combined_only_safe_preprocessor_fit(code: str) -> bool:
    """True if ``combined``-style vars are only passed to safe preprocessor ``.fit()`` calls."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    combined_names = frozenset({
        "combined", "X_full", "X_all", "X_combined", "full_data", "all_data", "data_full",
        "X_both", "X_total", "total_data", "data_all", "data_combined",
    })
    safe_vars: set[str] = set()
    assigns: list[ast.Assign] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        assigns.append(node)
        if isinstance(node.value, ast.Call):
            if _ast_func_leaf(node.value) in _SAFE_PREPROCESSOR_NAMES:
                safe_vars |= _ast_assign_lhs(node)

    combined_targets: set[str] = set()
    for asgn in assigns:
        lhs = _ast_assign_lhs(asgn)
        if not (lhs & combined_names):
            continue
        if isinstance(asgn.value, ast.Call):
            fn = _ast_func_leaf(asgn.value)
            if fn in _AST_COMBINE_FUNCS or (
                isinstance(asgn.value.func, ast.Attribute)
                and asgn.value.func.attr in _AST_COMBINE_FUNCS
            ):
                combined_targets |= lhs & combined_names

    if not combined_targets:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _ast_func_leaf(node) not in _AST_TRAIN_METHODS:
            continue
        if {kw.arg for kw in node.keywords} & _AST_LEGIT_FIT_KW:
            continue
        arg_names: set[str] = set()
        for arg in node.args:
            arg_names |= _ast_names(arg)
        if not (arg_names & combined_targets):
            continue
        if not _is_safe_fit_receiver(node, safe_vars):
            return False
    return True


def _strip_comments(code: str) -> list[str]:
    """Return non-comment code lines (drops whole-line ``#`` comments)."""
    return [ln for ln in code.splitlines() if not ln.lstrip().startswith("#")]


# =====================================================================
# Public API
# =====================================================================

def validate_no_leakage(code: str) -> list[str]:
    """Return human-readable issues if *code* likely trains on holdout/validation data.

    Four-phase heuristic:

    Phase 1 -- Explicit combining of train+val named variables via known APIs.
    Phase 2 -- ``.fit()`` directly on validation data.
    Phase 3 -- Code reads ``dataset/validation*`` AND has row-combining or
               trains on a combined-name variable.
    Phase 4 -- AST-based taint tracking from validation file reads.

    Prefers false negatives over false positives.
    """
    if not (code or "").strip():
        return []

    live = _strip_comments(code)
    code_nc = "\n".join(live)
    issues: list[str] = []

    # ---- Phase 1: explicit train+val combining ----

    _AXIS1_TAIL = re.compile(
        r"axis\s*=\s*1\b|axis\s*=\s*['\"]columns['\"]|dim\s*=\s*1\b",
        re.IGNORECASE,
    )

    for name, re_tv, re_vt in _BRACKET_CHECKS:
        bracket_hit = False
        for pat in (re_tv, re_vt):
            for m in pat.finditer(code_nc):
                tail = code_nc[m.start() : min(len(code_nc), m.end() + 220)]
                if name == "pd.concat" and _AXIS1_TAIL.search(tail):
                    continue
                bracket_hit = True
                break
            if bracket_hit:
                break
        if bracket_hit:
            issues.append(f"{name} appears to combine training and validation/holdout data")

    if _RE_MERGE_TV.search(code_nc) or _RE_MERGE_VT.search(code_nc):
        issues.append("pd.merge appears to join training and validation/holdout frames")

    for line in live:
        if not (_RE_HAS_TRAIN.search(line) and _RE_HAS_VAL.search(line)):
            continue
        if re.match(r"\s*def\s+", line):
            continue
        if any(kw in line for kw in _FIT_KWARG_SKIP):
            continue
        if _LINE_COMBINE_RE.search(line):
            issues.append(
                "Line contains both train and validation tokens with a combining "
                "operation (.append / list unpack)"
            )
            break

    # ---- Phase 2: .fit() on validation data ----

    for line in live:
        if any(kw in line for kw in _FIT_KWARG_SKIP):
            continue
        if _RE_FIT_VAL_PATH.search(line):
            issues.append(".fit() may use validation/holdout data as training input")
            break

    for m in re.finditer(r"\.fit\s*\(", code_nc):
        i = m.end() - 1
        if i < 0 or i >= len(code_nc) or code_nc[i] != "(":
            continue
        depth, j = 1, i + 1
        while j < len(code_nc) and depth:
            if code_nc[j] == "(":
                depth += 1
            elif code_nc[j] == ")":
                depth -= 1
            j += 1
        if depth != 0:
            continue
        args_region = code_nc[i + 1 : j - 1]
        if not args_region or any(kw in args_region for kw in _FIT_KWARG_SKIP):
            continue
        first_arg = args_region.split(",")[0].strip()
        if re.match(
            r"^(?:X_val|y_val|val_df|validation|df_val|data_val)\b",
            first_arg,
            re.IGNORECASE,
        ):
            issues.append(".fit() first argument looks like validation/holdout data")
            break

    # ---- Phase 3: validation file read + generic combine ----

    ast_leak_issues: list[str] = []
    if _RE_READS_VAL_FILE.search(code_nc):
        ast_leak_issues = _ast_detect_val_leakage(code)
        p3_enc_safe = _phase3_combined_only_safe_preprocessor_fit(code)
        has_suspect_combine = False
        for m_comb in _RE_ANY_ROW_COMBINE.finditer(code_nc):
            snippet = code_nc[m_comb.start() : min(m_comb.start() + 200, len(code_nc))]
            if "pd.concat" in snippet.lower():
                if re.search(r"axis\s*=\s*1\b|axis\s*=\s*['\"]columns['\"]", snippet):
                    continue
            has_suspect_combine = True
            break
        # Regex-only row combine is noisy; require AST taint agreement unless
        # encoding-only ``combined`` pattern already cleared it.
        if has_suspect_combine and not p3_enc_safe and ast_leak_issues:
            issues.append(
                "Code reads dataset/validation* file AND uses a row-combining "
                "operation — potential leakage via aliased variables"
            )

        p3_combined_fit = _RE_FIT_COMBINED_VAR.search(code_nc)
        p3_combined_assign = _RE_ASSIGN_COMBINED.search(code_nc)
        if p3_combined_fit or p3_combined_assign:
            if not p3_enc_safe:
                if p3_combined_fit:
                    issues.append(
                        "Code reads dataset/validation* file AND calls .fit() on a "
                        "'full/combined/all' variable"
                    )
                if p3_combined_assign:
                    issues.append(
                        "Code reads dataset/validation* file AND assigns a "
                        "'full/combined/all' variable from a combining API"
                    )
    else:
        ast_leak_issues = _ast_detect_val_leakage(code)

    # ---- Phase 4: AST-based taint tracking ----
    issues.extend(ast_leak_issues)

    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for item in issues:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
