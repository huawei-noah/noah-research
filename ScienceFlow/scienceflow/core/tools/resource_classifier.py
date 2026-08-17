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

"""Static observe-only resource classification for bash commands.

The classifier is intentionally conservative. It labels obvious light commands,
obvious training commands, and inference/TT-style commands. Ambiguous executable
commands remain ``unknown_exec`` so later phases can observe them at runtime
instead of blocking them based on a fragile string rule.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

RESOURCE_READONLY_CPU = "readonly_cpu"
RESOURCE_LIGHT_CPU = "light_cpu"
RESOURCE_LIGHT_GPU_PROBE = "light_gpu_probe"
RESOURCE_UNKNOWN_EXEC = "unknown_exec"
RESOURCE_HEAVY_CPU_CANDIDATE = "heavy_cpu_candidate"
# Backward-compatible training class used by LNR resource monitors.
RESOURCE_HEAVY_GPU_CANDIDATE = "heavy_gpu_candidate"
# New LHR policy classes. Training keeps the legacy class above for compatibility.
RESOURCE_HEAVY_GPU_TRAIN = "heavy_gpu_train"
RESOURCE_PURE_TT_CPU = "pure_tt_cpu"
RESOURCE_GPU_TT_LIGHT = "gpu_tt_light"
RESOURCE_GPU_FEATURE_EXTRACT = "gpu_feature_extract"
RESOURCE_GPU_LIGHT_TRAIN = "explicit_light_train"
RESOURCE_UNKNOWN_GPU_EXEC = "unknown_gpu_exec"

_READONLY_TOOLS = {
    "awk",
    "cat",
    "cut",
    "df",
    "du",
    "echo",
    "find",
    "grep",
    "head",
    "jq",
    "ls",
    "pwd",
    "rg",
    "sed",
    "sort",
    "stat",
    "tail",
    "test",
    "wc",
}
_LIGHT_CPU_TOOLS = {"true", "false", "date", "which", "command", "python", "python3"}
_LIGHT_FS_SETUP_TOOLS = {"mkdir"}
_PIP_METADATA_SUBCOMMANDS = {"check", "freeze", "list", "show"}
_SHELL_TOOLS = {"bash", "sh", "/bin/bash", "/bin/sh"}
_PYTHON_TOOLS = {"python", "python3"}
_ENV_TOOL = "env"
_TRAIN_WORD_RE = re.compile(
    r"\b("
    r"accelerate|deepspeed|torchrun|"
    r"fit|train|training|trainer|epoch|epochs|fold|folds|"
    r"backward|optimizer|loss\.backward|"
    r"lightgbm|xgboost|catboost|bert|transformer"
    r")\b",
    re.IGNORECASE,
)
_GPU_FEATURE_WORD_RE = re.compile(
    r"\b("
    r"extract(?:_features?)?|feature(?:s)?|embedding(?:s)?|embed(?:s|ding)?|"
    r"encode(?:_images?)?|image_features?"
    r")\b",
    re.IGNORECASE,
)
_GPU_TT_WORD_RE = re.compile(
    r"\b("
    r"predict|prediction|predicting|infer|inference|tta|test_time|"
    r"ensemble|submission|submit|evaluate|eval"
    r")\b",
    re.IGNORECASE,
)
_CPU_TT_WORD_RE = re.compile(
    r"\b("
    r"blend(?:_submission)?|blending|average|averaging|rank|ranking|"
    r"calibrate|calibration|stack|stacking|submission_blend"
    r")\b",
    re.IGNORECASE,
)
_GPU_PROBE_RE = re.compile(
    r"(nvidia-smi|torch\.cuda|cuda\.is_available|cuda_available|device_count|get_device_name)",
    re.IGNORECASE,
)
_RESOURCE_INTENT_ENV_KEYS = {
    "SCIENCEFLOW_RESOURCE_INTENT",
    "_SCIENCEFLOW_RESOURCE_INTENT",
}
_RESOURCE_INTENT_CLASS = {
    "readonly": RESOURCE_READONLY_CPU,
    "readonly_cpu": RESOURCE_READONLY_CPU,
    "read_only_cpu": RESOURCE_READONLY_CPU,
    "light": RESOURCE_LIGHT_CPU,
    "light_cpu": RESOURCE_LIGHT_CPU,
    "cpu_light": RESOURCE_LIGHT_CPU,
    "cpu": RESOURCE_HEAVY_CPU_CANDIDATE,
    "cpu_support": RESOURCE_HEAVY_CPU_CANDIDATE,
    "heavy_cpu": RESOURCE_HEAVY_CPU_CANDIDATE,
    "heavy_cpu_candidate": RESOURCE_HEAVY_CPU_CANDIDATE,
    "cpu_train": RESOURCE_HEAVY_CPU_CANDIDATE,
    "gpu": RESOURCE_HEAVY_GPU_CANDIDATE,
    "heavy_gpu": RESOURCE_HEAVY_GPU_CANDIDATE,
    "heavy_gpu_candidate": RESOURCE_HEAVY_GPU_CANDIDATE,
    "gpu_train": RESOURCE_HEAVY_GPU_TRAIN,
    "gpu_training": RESOURCE_HEAVY_GPU_TRAIN,
    "train_gpu": RESOURCE_HEAVY_GPU_TRAIN,
    "heavy_gpu_train": RESOURCE_HEAVY_GPU_TRAIN,
    "light_train": RESOURCE_GPU_LIGHT_TRAIN,
    "gpu_light_train": RESOURCE_GPU_LIGHT_TRAIN,
    "light_gpu_train": RESOURCE_GPU_LIGHT_TRAIN,
    "explicit_light_train": RESOURCE_GPU_LIGHT_TRAIN,
    "gpu_tt": RESOURCE_GPU_TT_LIGHT,
    "gpu_infer": RESOURCE_GPU_TT_LIGHT,
    "gpu_inference": RESOURCE_GPU_TT_LIGHT,
    "gpu_predict": RESOURCE_GPU_TT_LIGHT,
    "gpu_feature": RESOURCE_GPU_FEATURE_EXTRACT,
    "gpu_feature_extract": RESOURCE_GPU_FEATURE_EXTRACT,
    "feature_gpu": RESOURCE_GPU_FEATURE_EXTRACT,
    "cpu_tt": RESOURCE_PURE_TT_CPU,
    "pure_tt_cpu": RESOURCE_PURE_TT_CPU,
    "cpu_predict": RESOURCE_PURE_TT_CPU,
    "submission_cpu": RESOURCE_PURE_TT_CPU,
    "unknown_gpu": RESOURCE_UNKNOWN_GPU_EXEC,
    "unknown_gpu_exec": RESOURCE_UNKNOWN_GPU_EXEC,
}
_CPU_RESOURCE_CLASSES = {
    RESOURCE_READONLY_CPU,
    RESOURCE_LIGHT_CPU,
    RESOURCE_HEAVY_CPU_CANDIDATE,
    RESOURCE_PURE_TT_CPU,
}
_CPU_ML_INLINE_RE = re.compile(
    r"\b("
    r"sklearn|scikit[-_ ]?learn|lightgbm|xgboost|catboost|"
    r"HistGradientBoosting|RandomForest|ExtraTrees|LogisticRegression|RidgeClassifier|"
    r"LGBMClassifier|LGBMRegressor|XGBClassifier|XGBRegressor|CatBoostClassifier|CatBoostRegressor"
    r")\b",
    re.IGNORECASE,
)
_CPU_ML_FIT_RE = re.compile(r"(?:\.fit\s*\(|\bfit\s*\()", re.IGNORECASE)
_GPU_ML_INLINE_RE = re.compile(
    r"("
    r"torch\.|tensorflow|keras|cuda|\.cuda\s*\(|to\(\s*[^,)\s]*cuda|"
    r"tree_method\s*=\s*[^,)\s]*gpu_hist|device_type\s*=\s*[^,)\s]*gpu|"
    r"device\s*=\s*[^,)\s]*cuda|task_type\s*=\s*[^,)\s]*gpu"
    r")",
    re.IGNORECASE,
)
_EXPLICIT_GPU_SIGNAL_RE = re.compile(
    r"("
    r"\b(torchrun|accelerate|deepspeed)\b|torch\.cuda|torch\.device\(\s*['\"]cuda|"
    r"\.cuda\s*\(|to\(\s*['\"]cuda|"
    r"tree_method\s*=\s*['\"]?gpu_hist|device_type\s*=\s*['\"]?gpu|"
    r"device\s*=\s*['\"]?cuda|task_type\s*=\s*['\"]?gpu"
    r")",
    re.IGNORECASE,
)
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")


@dataclass(frozen=True)
class ClassifiedCommand:
    resource_class: str
    executable: str
    normalized_command: str
    reason: str


def classify_bash_command(command: str) -> ClassifiedCommand:
    """Classify a bash command without changing execution behavior."""
    cpu_only_cuda = command_has_cpu_only_cuda(command)
    normalized = normalize_shell_command(command)
    segments = _split_segments(normalized)
    if not segments:
        return ClassifiedCommand(RESOURCE_LIGHT_CPU, "", normalized, "empty")

    classes: list[str] = []
    reasons: list[str] = []
    executable = ""
    for seg in segments:
        sub = _classify_segment(seg)
        if not executable and sub.executable:
            executable = sub.executable
        classes.append(sub.resource_class)
        reasons.append(sub.reason)

    if RESOURCE_HEAVY_GPU_CANDIDATE in classes:
        cls = RESOURCE_HEAVY_GPU_CANDIDATE
    elif RESOURCE_HEAVY_GPU_TRAIN in classes:
        cls = RESOURCE_HEAVY_GPU_TRAIN
    elif RESOURCE_GPU_FEATURE_EXTRACT in classes:
        cls = RESOURCE_GPU_FEATURE_EXTRACT
    elif RESOURCE_GPU_LIGHT_TRAIN in classes:
        cls = RESOURCE_GPU_LIGHT_TRAIN
    elif RESOURCE_GPU_TT_LIGHT in classes:
        cls = RESOURCE_GPU_TT_LIGHT
    elif RESOURCE_UNKNOWN_GPU_EXEC in classes:
        cls = RESOURCE_UNKNOWN_GPU_EXEC
    elif RESOURCE_HEAVY_CPU_CANDIDATE in classes:
        cls = RESOURCE_HEAVY_CPU_CANDIDATE
    elif RESOURCE_UNKNOWN_EXEC in classes:
        cls = RESOURCE_UNKNOWN_EXEC
    elif RESOURCE_LIGHT_GPU_PROBE in classes:
        cls = RESOURCE_LIGHT_GPU_PROBE
    elif classes and all(c == RESOURCE_READONLY_CPU for c in classes):
        cls = RESOURCE_READONLY_CPU
    elif RESOURCE_PURE_TT_CPU in classes:
        cls = RESOURCE_PURE_TT_CPU
    else:
        cls = RESOURCE_LIGHT_CPU
    cls, intent_reason = _apply_resource_intent(cls, command=command, normalized=normalized)
    if intent_reason:
        reasons.append(intent_reason)
    if cpu_only_cuda:
        cls = _downgrade_for_cpu_only_cuda(cls)
    return ClassifiedCommand(cls, executable, normalized, ",".join(r for r in reasons if r))


def command_has_cpu_only_cuda(command: str) -> bool:
    """Return true when the command explicitly masks CUDA from the child process."""
    tokens = _tokens(str(command or ""))
    for i, tok in enumerate(tokens):
        raw = str(tok or "")
        if raw.startswith("CUDA_VISIBLE_DEVICES="):
            value = raw.split("=", 1)[1].strip().strip("'\"").lower()
            if value in {"", "-1", "none", "cpu", "nodevfile"}:
                return True
        if raw == "unset" and i + 1 < len(tokens) and tokens[i + 1] == "CUDA_VISIBLE_DEVICES":
            return True
        if raw == "-u" and i + 1 < len(tokens) and tokens[i + 1] == "CUDA_VISIBLE_DEVICES":
            return True
        if raw == "--unset" and i + 1 < len(tokens) and tokens[i + 1] == "CUDA_VISIBLE_DEVICES":
            return True
        if raw == "--unset=CUDA_VISIBLE_DEVICES":
            return True
        if raw == "env" and i + 1 < len(tokens) and tokens[i + 1] in {"-u", "--unset"} and i + 2 < len(tokens) and tokens[i + 2] == "CUDA_VISIBLE_DEVICES":
            return True
    return False


def _downgrade_for_cpu_only_cuda(resource_class: str) -> str:
    if resource_class in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_GPU_FEATURE_EXTRACT, RESOURCE_GPU_LIGHT_TRAIN, RESOURCE_UNKNOWN_GPU_EXEC}:
        return RESOURCE_HEAVY_CPU_CANDIDATE
    if resource_class == RESOURCE_GPU_TT_LIGHT:
        return RESOURCE_PURE_TT_CPU
    if resource_class == RESOURCE_LIGHT_GPU_PROBE:
        return RESOURCE_LIGHT_CPU
    return resource_class


def _apply_resource_intent(resource_class: str, *, command: str, normalized: str) -> tuple[str, str]:
    raw_intent = _extract_resource_intent(command)
    if not raw_intent:
        return resource_class, ""
    intent_class = _RESOURCE_INTENT_CLASS.get(raw_intent)
    if not intent_class:
        return resource_class, f"resource_intent_invalid:{raw_intent}"
    if intent_class in _CPU_RESOURCE_CLASSES and _has_explicit_gpu_runtime_signal(command, normalized):
        return resource_class, f"resource_intent_conflict_gpu:{raw_intent}"
    return intent_class, f"resource_intent:{raw_intent}"


def _extract_resource_intent(command: str) -> str:
    s = str(command or "").strip()
    seen: set[str] = set()
    for _ in range(8):
        if not s or s in seen:
            break
        seen.add(s)
        tokens = _tokens(s)
        raw = _resource_intent_from_env_tokens(tokens)
        if raw:
            return raw
        changed = False
        tokens, did = _strip_leading_env(tokens)
        if did:
            s = shlex.join(tokens)
            continue
        if len(tokens) >= 3 and tokens[0] == "cd":
            try:
                idx = tokens.index("&&")
            except ValueError:
                idx = -1
            if idx >= 0 and idx + 1 < len(tokens):
                s = shlex.join(tokens[idx + 1 :])
                changed = True
                continue
        if tokens and tokens[0] == "taskset":
            s2 = _strip_taskset(tokens)
            if s2 is not None:
                s = s2
                changed = True
                continue
        if tokens and tokens[0] == "timeout":
            s2 = _strip_timeout(tokens)
            if s2 is not None:
                s = s2
                changed = True
                continue
        if tokens and tokens[0] == _ENV_TOOL:
            s2 = _strip_env_command(tokens)
            if s2 is not None:
                s = s2
                changed = True
                continue
        if len(tokens) >= 3 and tokens[0] in _SHELL_TOOLS and tokens[1] in {"-c", "-lc"}:
            s = " ".join(tokens[2:])
            changed = True
            continue
        if not changed:
            break
    return ""


def _resource_intent_from_env_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ""
    i = 0
    if tokens[0] == _ENV_TOOL:
        i = 1
        while i < len(tokens):
            tok = tokens[i]
            if tok in {"-u", "--unset"}:
                i += 2
                continue
            if tok.startswith("-u") and len(tok) > 2:
                i += 1
                continue
            if tok.startswith("--unset=") or tok.startswith("-"):
                i += 1
                continue
            break
    while i < len(tokens) and _ENV_ASSIGN_RE.match(tokens[i]):
        key, value = tokens[i].split("=", 1)
        if key in _RESOURCE_INTENT_ENV_KEYS:
            return _normalize_resource_intent(value)
        i += 1
    return ""


def _normalize_resource_intent(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(raw or "").strip().strip("'\"").lower()).strip("_")


def _has_explicit_gpu_runtime_signal(command: str, normalized: str) -> bool:
    if command_has_cpu_only_cuda(command):
        return False
    text = f"{command or ''}\n{normalized or ''}"
    if _EXPLICIT_GPU_SIGNAL_RE.search(text):
        return True
    for match in re.finditer(r"(?:^|\s)CUDA_VISIBLE_DEVICES=([^\s;&|]+)", text):
        value = match.group(1).strip().strip("'\"").lower()
        if value and value not in {"-1", "none", "cpu", "nodevfile"}:
            return True
    return False


def normalize_shell_command(command: str) -> str:
    s = (command or "").strip()
    if not s:
        return ""
    # Repeatedly unwrap common scheduler/runtime wrappers from the beginning.
    for _ in range(6):
        tokens = _tokens(s)
        if not tokens:
            return s
        changed = False
        tokens, did = _strip_leading_env(tokens)
        if did:
            s = shlex.join(tokens)
            continue
        if len(tokens) >= 3 and tokens[0] == "cd":
            try:
                idx = tokens.index("&&")
            except ValueError:
                idx = -1
            if idx >= 0 and idx + 1 < len(tokens):
                s = shlex.join(tokens[idx + 1 :])
                changed = True
                if changed:
                    continue
        if tokens[:2] == ["uv", "run"]:
            s = shlex.join(tokens[2:])
            changed = True
            continue
        if tokens and tokens[0] == "taskset":
            s2 = _strip_taskset(tokens)
            if s2 is not None:
                s = s2
                changed = True
                continue
        if tokens and tokens[0] == "timeout":
            s2 = _strip_timeout(tokens)
            if s2 is not None:
                s = s2
                changed = True
                continue
        if tokens and tokens[0] == _ENV_TOOL:
            s2 = _strip_env_command(tokens)
            if s2 is not None:
                s = s2
                changed = True
                continue
        if len(tokens) >= 3 and tokens[0] in _SHELL_TOOLS and tokens[1] in {"-c", "-lc"}:
            s = " ".join(tokens[2:])
            changed = True
            continue
        if not changed:
            break
    return s.strip()


def _classify_segment(segment: str) -> ClassifiedCommand:
    normalized = normalize_shell_command(segment)
    tokens = _tokens(normalized)
    if not tokens:
        return ClassifiedCommand(RESOURCE_LIGHT_CPU, "", normalized, "empty_segment")
    tokens, _ = _strip_leading_env(tokens)
    if not tokens:
        return ClassifiedCommand(RESOURCE_LIGHT_CPU, "", normalized, "env_only")
    exe = tokens[0]
    exe_base = exe.rsplit("/", 1)[-1]
    text = " ".join(tokens)

    if _GPU_PROBE_RE.search(text):
        return ClassifiedCommand(RESOURCE_LIGHT_GPU_PROBE, exe_base, normalized, "gpu_probe")
    if exe_base in {"torchrun", "accelerate", "deepspeed"}:
        return ClassifiedCommand(RESOURCE_HEAVY_GPU_CANDIDATE, exe_base, normalized, "launcher")
    if exe_base in _READONLY_TOOLS:
        return ClassifiedCommand(RESOURCE_READONLY_CPU, exe_base, normalized, "readonly_tool")
    if exe_base in _LIGHT_FS_SETUP_TOOLS:
        return ClassifiedCommand(RESOURCE_LIGHT_CPU, exe_base, normalized, "light_fs_setup_tool")
    if _is_pip_metadata_command(tokens):
        return ClassifiedCommand(RESOURCE_READONLY_CPU, exe_base, normalized, "pip_metadata_tool")
    inline_cpu_ml = _python_inline_cpu_ml_class(tokens, text)
    if inline_cpu_ml:
        return ClassifiedCommand(inline_cpu_ml, exe_base, normalized, "python_c_cpu_ml")
    inline_python = _python_inline_default_class(tokens)
    if inline_python and not _python_inline_should_scan_resource_keywords(text):
        return ClassifiedCommand(inline_python, exe_base, normalized, "python_c")
    if _TRAIN_WORD_RE.search(text):
        return ClassifiedCommand(RESOURCE_HEAVY_GPU_CANDIDATE, exe_base, normalized, "train_keyword")
    if _GPU_FEATURE_WORD_RE.search(text):
        return ClassifiedCommand(RESOURCE_GPU_FEATURE_EXTRACT, exe_base, normalized, "feature_extract_keyword")
    if _CPU_TT_WORD_RE.search(text):
        return ClassifiedCommand(RESOURCE_PURE_TT_CPU, exe_base, normalized, "pure_tt_keyword")
    if _GPU_TT_WORD_RE.search(text):
        return ClassifiedCommand(RESOURCE_GPU_TT_LIGHT, exe_base, normalized, "tt_keyword")
    if exe_base in _SHELL_TOOLS:
        return ClassifiedCommand(RESOURCE_UNKNOWN_EXEC, exe_base, normalized, "shell_script")
    if exe_base in _PYTHON_TOOLS:
        inline_python = _python_inline_default_class(tokens)
        if inline_python:
            reason = "python_c"
            if inline_python == RESOURCE_HEAVY_CPU_CANDIDATE:
                reason = "python_c_solution_entrypoint"
            elif inline_python == RESOURCE_UNKNOWN_EXEC:
                reason = "python_c_script_entrypoint"
            return ClassifiedCommand(inline_python, exe_base, normalized, reason)
        if _python_runs_solution_script(tokens):
            return ClassifiedCommand(
                RESOURCE_HEAVY_CPU_CANDIDATE,
                exe_base,
                normalized,
                "python_solution_script",
            )
        return ClassifiedCommand(RESOURCE_UNKNOWN_EXEC, exe_base, normalized, "python_exec")
    if exe_base in _LIGHT_CPU_TOOLS:
        return ClassifiedCommand(RESOURCE_LIGHT_CPU, exe_base, normalized, "light_tool")
    return ClassifiedCommand(RESOURCE_UNKNOWN_EXEC, exe_base, normalized, "unknown_executable")


def _tokens(s: str) -> list[str]:
    try:
        lexer = shlex.shlex(s, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError:
        try:
            return shlex.split(s, posix=True)
        except ValueError:
            return s.split()


def _strip_leading_env(tokens: list[str]) -> tuple[list[str], bool]:
    i = 0
    while i < len(tokens) and _ENV_ASSIGN_RE.match(tokens[i]):
        i += 1
    return tokens[i:], i > 0


def _strip_taskset(tokens: list[str]) -> str | None:
    if len(tokens) >= 4 and tokens[1] == "-c":
        return shlex.join(tokens[3:])
    return None


def _strip_timeout(tokens: list[str]) -> str | None:
    if len(tokens) < 3:
        return None
    i = 1
    while i < len(tokens) and tokens[i].startswith("-"):
        i += 1
        if i < len(tokens) and re.match(r"^\d", tokens[i]):
            i += 1
    if i < len(tokens) and re.match(r"^\d", tokens[i]):
        i += 1
    if i < len(tokens):
        return shlex.join(tokens[i:])
    return None


def _strip_env_command(tokens: list[str]) -> str | None:
    """Strip common ``env`` wrappers, preserving the command to classify."""
    if not tokens or tokens[0] != _ENV_TOOL:
        return None
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok in {"-u", "--unset"}:
            i += 2
            continue
        if tok.startswith("-u") and len(tok) > 2:
            i += 1
            continue
        if tok.startswith("--unset="):
            i += 1
            continue
        if _ENV_ASSIGN_RE.match(tok):
            i += 1
            continue
        if tok.startswith("-"):
            i += 1
            continue
        break
    if i < len(tokens):
        return shlex.join(tokens[i:])
    return None



def _is_pip_metadata_command(tokens: list[str]) -> bool:
    if not tokens:
        return False
    exe_base = str(tokens[0]).rsplit("/", 1)[-1]
    if exe_base in {"pip", "pip3"}:
        return len(tokens) >= 2 and tokens[1] in _PIP_METADATA_SUBCOMMANDS
    if exe_base == "uv" and len(tokens) >= 3 and tokens[1] == "pip":
        return tokens[2] in _PIP_METADATA_SUBCOMMANDS
    if exe_base in _PYTHON_TOOLS:
        if len(tokens) >= 4 and tokens[1:3] == ["-m", "pip"]:
            return tokens[3] in _PIP_METADATA_SUBCOMMANDS
        if len(tokens) >= 5 and tokens[1:4] == ["-u", "-m", "pip"]:
            return tokens[4] in _PIP_METADATA_SUBCOMMANDS
    return False

def _python_runs_solution_script(tokens: list[str]) -> bool:
    for tok in tokens[1:]:
        if tok.startswith("-"):
            continue
        return tok.rsplit("/", 1)[-1] == "solution.py"
    return False


def _python_inline_cpu_ml_class(tokens: list[str], text: str) -> str | None:
    if not tokens:
        return None
    exe_base = str(tokens[0]).rsplit("/", 1)[-1]
    if exe_base not in _PYTHON_TOOLS:
        return None
    is_inline = len(tokens) >= 3 and tokens[1] == "-c"
    is_inline = is_inline or (len(tokens) >= 4 and tokens[1] == "-u" and tokens[2] == "-c")
    if not is_inline or not _CPU_ML_INLINE_RE.search(text):
        return None
    if _GPU_ML_INLINE_RE.search(text):
        return None
    if _CPU_ML_FIT_RE.search(text):
        return RESOURCE_HEAVY_CPU_CANDIDATE
    return RESOURCE_LIGHT_CPU


def _python_inline_default_class(tokens: list[str]) -> str | None:
    if not tokens:
        return None
    exe_base = str(tokens[0]).rsplit("/", 1)[-1]
    if exe_base not in _PYTHON_TOOLS:
        return None
    text = " ".join(tokens)
    is_inline = len(tokens) >= 3 and tokens[1] == "-c"
    is_inline = is_inline or (len(tokens) >= 4 and tokens[1] == "-u" and tokens[2] == "-c")
    if not is_inline:
        return None
    if _python_inline_reads_solution_script(text):
        return RESOURCE_HEAVY_CPU_CANDIDATE
    if _python_inline_reads_script(text):
        return RESOURCE_UNKNOWN_EXEC
    return RESOURCE_LIGHT_CPU


def _python_inline_should_scan_resource_keywords(text: str) -> bool:
    if _GPU_ML_INLINE_RE.search(text):
        return True
    if _CPU_ML_FIT_RE.search(text):
        return True
    return bool(re.search(r"\b(epochs?|backward|optimizer|loss\.backward)\b", str(text or ""), re.IGNORECASE))


def _python_inline_read_entrypoints(text: str) -> list[str]:
    out: list[str] = []
    for match in re.finditer(r"(?:open|Path)\(\s*['\"]?([^'\"\)\s]+\.py)['\"]?", str(text or "")):
        out.append(match.group(1).replace("\\", "/").rsplit("/", 1)[-1])
    return out


def _python_inline_reads_script(text: str) -> bool:
    return bool(_python_inline_read_entrypoints(text))


def _python_inline_reads_solution_script(text: str) -> bool:
    return any(name == "solution.py" for name in _python_inline_read_entrypoints(text))


def _split_segments(command: str) -> list[str]:
    tokens = _tokens(command)
    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in {"&&", "||", ";", "|", ";&", ";;"}:
            if segments[-1]:
                segments.append([])
            continue
        segments[-1].append(tok)
    return [" ".join(seg).strip() for seg in segments if seg]
