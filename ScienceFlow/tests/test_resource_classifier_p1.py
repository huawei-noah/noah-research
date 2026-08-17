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

import pytest

from scienceflow.core.tools.resource_classifier import (
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_CPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_LIGHT_CPU,
    RESOURCE_LIGHT_GPU_PROBE,
    RESOURCE_PURE_TT_CPU,
    RESOURCE_READONLY_CPU,
    RESOURCE_UNKNOWN_EXEC,
    classify_bash_command,
    normalize_shell_command,
)
from scienceflow.core.tools.bash.guards import (
    background_resource_command_blocked_error,
    dangerous_delete_command_blocked_error,
    mixed_file_write_execution_blocked_error,
    truncated_resource_output_blocked_error,
)
from scienceflow.core.tools.bash_tool import (
    _normalize_cuda_visible_devices_for_task_pool,
    _parse_visible_gpu_ids,
)
from scienceflow.core.parallel_runner import ParallelRunner


@pytest.mark.parametrize(
    "cmd",
    [
        "ls -la",
        "head -20 train.csv",
        "grep -R foo . | head",
        "jq '.a' file.json",
        "cd /tmp/work && find . -maxdepth 2 -type f | wc -l",
    ],
)
def test_readonly_commands_classify_readonly_cpu(cmd: str) -> None:
    assert classify_bash_command(cmd).resource_class == RESOURCE_READONLY_CPU


@pytest.mark.parametrize(
    "cmd",
    [
        "python3 -c 'print(1)'",
        "VAR=1 python -u -c 'print(1)'",
        "uv run python -c 'print(1)'",
    ],
)
def test_python_c_classifies_light_cpu(cmd: str) -> None:
    assert classify_bash_command(cmd).resource_class == RESOURCE_LIGHT_CPU


@pytest.mark.parametrize(
    "cmd",
    [
        "python3 -c \"import pandas as pd; train = pd.read_csv('dataset/train.csv'); print(train.shape)\"",
        "python3 -c \"import csv; rows=list(csv.DictReader(open('dataset/validation.csv'))); print(len(rows))\"",
        "python3 -c \"import numpy as np; x=np.load('dataset/test_logits.npy', mmap_mode='r'); print(x.shape)\"",
    ],
)
def test_python_c_dataset_inspection_with_train_paths_stays_light_cpu(cmd: str) -> None:
    classified = classify_bash_command(cmd)

    assert classified.resource_class == RESOURCE_LIGHT_CPU
    assert classified.reason == "python_c"


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        ("SCIENCEFLOW_RESOURCE_INTENT=cpu_support python3 train.py --fold 0", RESOURCE_HEAVY_CPU_CANDIDATE),
        ("taskset -c 0-3 bash -c 'SCIENCEFLOW_RESOURCE_INTENT=cpu_support python3 train.py'", RESOURCE_HEAVY_CPU_CANDIDATE),
        ("env SCIENCEFLOW_RESOURCE_INTENT=gpu_train python3 -c 'print(1)'", RESOURCE_HEAVY_GPU_TRAIN),
        ("SCIENCEFLOW_RESOURCE_INTENT=gpu_tt python3 predict.py", RESOURCE_GPU_TT_LIGHT),
        ("SCIENCEFLOW_RESOURCE_INTENT=readonly_cpu python3 -c 'print(1)'", RESOURCE_READONLY_CPU),
    ],
)
def test_explicit_resource_intent_overrides_static_guess_when_uncontested(cmd: str, expected: str) -> None:
    classified = classify_bash_command(cmd)
    assert classified.resource_class == expected
    assert "resource_intent:" in classified.reason


def test_cpu_resource_intent_does_not_override_explicit_gpu_signal() -> None:
    classified = classify_bash_command(
        "SCIENCEFLOW_RESOURCE_INTENT=cpu_support python3 -c 'import torch; print(torch.cuda.is_available())'"
    )

    assert classified.resource_class == RESOURCE_LIGHT_GPU_PROBE
    assert "resource_intent_conflict_gpu:cpu_support" in classified.reason


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        (
            "python3 -c 'from sklearn.ensemble import HistGradientBoostingClassifier; model=HistGradientBoostingClassifier(); model.fit(X, y)'",
            RESOURCE_HEAVY_CPU_CANDIDATE,
        ),
        ("python3 -c 'import sklearn; print(sklearn.__version__)'", RESOURCE_LIGHT_CPU),
        ("python3 -c 'import xgboost as xgb; xgb.XGBClassifier(tree_method=\"hist\").fit(X, y)'", RESOURCE_HEAVY_CPU_CANDIDATE),
        ("python3 -c 'import xgboost as xgb; xgb.XGBClassifier(tree_method=\"gpu_hist\").fit(X, y)'", RESOURCE_HEAVY_GPU_CANDIDATE),
    ],
)
def test_inline_cpu_ml_commands_do_not_default_to_gpu_train(cmd: str, expected: str) -> None:
    assert classify_bash_command(cmd).resource_class == expected


def test_file_write_with_training_text_stays_readonly_cpu() -> None:
    cmd = (
        "cat > tmp/val_score.py <<'PY'\n"
        "from sklearn.ensemble import HistGradientBoostingClassifier\n"
        "model.fit(X, y)\n"
        "PY"
    )

    assert classify_bash_command(cmd).resource_class == RESOURCE_READONLY_CPU


@pytest.mark.parametrize(
    "cmd",
    [
        "mkdir -p artifacts tmp",
        "bash -lc \"mkdir -p artifacts tmp\"",
        "mkdir -p train_features",
    ],
)
def test_light_filesystem_setup_commands_classify_light_cpu(cmd: str) -> None:
    assert classify_bash_command(cmd).resource_class == RESOURCE_LIGHT_CPU


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        ("uv pip list", RESOURCE_READONLY_CPU),
        ("uv pip show torch", RESOURCE_READONLY_CPU),
        ("uv pip freeze", RESOURCE_READONLY_CPU),
        ("pip list", RESOURCE_READONLY_CPU),
        ("pip show torch", RESOURCE_READONLY_CPU),
        ("python3 -m pip list", RESOURCE_READONLY_CPU),
        ("python3 -u -m pip check", RESOURCE_READONLY_CPU),
        ("python3 -c \"print(1)\"; uv pip list | grep torch", RESOURCE_LIGHT_CPU),
    ],
)
def test_pip_metadata_commands_classify_without_unknown_exec(cmd: str, expected: str) -> None:
    assert classify_bash_command(cmd).resource_class == expected


def test_pip_install_stays_non_readonly_resource_class() -> None:
    assert classify_bash_command("uv pip install timm").resource_class != RESOURCE_READONLY_CPU


@pytest.mark.parametrize(
    "cmd",
    [
        "nvidia-smi",
        "python3 -c 'import torch; print(torch.cuda.is_available())'",
        "CUDA_VISIBLE_DEVICES=0 python -c 'import torch; print(torch.cuda.device_count())'",
    ],
)
def test_gpu_probe_classifies_light_gpu_probe(cmd: str) -> None:
    assert classify_bash_command(cmd).resource_class == RESOURCE_LIGHT_GPU_PROBE


@pytest.mark.parametrize(
    "cmd",
    [
        "torchrun --nproc_per_node=4 train.py",
        "accelerate launch train.py",
        "python3 train.py --epochs 3",
        "python3 -c 'model.fit(X, y, epochs=5)'",
        "taskset -c 0-7 bash -c 'python3 train.py --fold 0'",
    ],
)
def test_heavy_keywords_classify_heavy_gpu_candidate(cmd: str) -> None:
    assert classify_bash_command(cmd).resource_class == RESOURCE_HEAVY_GPU_CANDIDATE


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        ("python3 predict.py --tta 4", RESOURCE_GPU_TT_LIGHT),
        ("python3 extract_features.py", RESOURCE_GPU_FEATURE_EXTRACT),
        ("python3 blend_submission.py", RESOURCE_PURE_TT_CPU),
    ],
)
def test_tt_and_feature_commands_classify_new_policy_classes(cmd: str, expected: str) -> None:
    assert classify_bash_command(cmd).resource_class == expected


@pytest.mark.parametrize(
    "cmd",
    [
        "python3 solution.py",
        "python3 solution.py 2>&1 | tail -20",
        "ENSEMBLE_PREDICT=0 python3 solution.py --num_boost_round 500",
        "env -u ENSEMBLE_PREDICT python3 solution.py",
        "unset ENSEMBLE_PREDICT 2>/dev/null; env -u ENSEMBLE_PREDICT python3 solution.py",
    ],
)
def test_solution_script_classifies_heavy_cpu_candidate(cmd: str) -> None:
    assert classify_bash_command(cmd).resource_class == RESOURCE_HEAVY_CPU_CANDIDATE


@pytest.mark.parametrize(
    "cmd",
    [
        "python3 train.py 2>&1 | tail -30",
        "python3 solution.py | head -50",
        "python3 predict.py 2>&1 | tee tmp/predict_log.txt | tail -20",
        "python3 tmp/score_existing.py 2>&1 | tail -5",
        "python3 validate.py 2>&1 | head -20",
        "bash run_eval.sh | tail -10",
        "bash run_inference.sh | head -20",
        "torchrun --nproc_per_node=2 train.py | tail",
    ],
)
def test_slow_entrypoint_output_truncating_pipe_is_blocked(cmd: str) -> None:
    err = truncated_resource_output_blocked_error(cmd)

    assert err is not None
    assert "do not pipe long-running" in err


def test_plain_log_tail_is_not_blocked() -> None:
    assert truncated_resource_output_blocked_error("tail -30 tmp/train.log") is None


@pytest.mark.parametrize(
    "cmd",
    [
        "python3 -c \"from sentence_transformers import SentenceTransformer; model=SentenceTransformer('x'); model.encode(['a'])\" | tail -15",
        "python3 -c \"from predict import predict_notebooks, validate; print('run')\" 2>&1 | head -20",
    ],
)
def test_inline_long_python_output_truncating_pipe_is_blocked(cmd: str) -> None:
    err = truncated_resource_output_blocked_error(cmd)

    assert err is not None
    assert "do not pipe long-running" in err


def test_inline_print_tail_is_not_blocked() -> None:
    assert truncated_resource_output_blocked_error("python3 -c 'print(1)' 2>&1 | tail -5") is None


def test_background_resource_guard_allows_foreground_logical_and() -> None:
    assert background_resource_command_blocked_error("sleep 30 && python3 predict.py") is None
    assert background_resource_command_blocked_error("cd /tmp/work && CUDA_VISIBLE_DEVICES=0 python3 train.py") is None


def test_background_resource_guard_blocks_real_background_operator() -> None:
    err = background_resource_command_blocked_error("python3 train.py &")

    assert err is not None
    assert "background long-running resource commands" in err


def test_mixed_heredoc_write_and_resource_execution_is_blocked() -> None:
    cmd = """cat > tmp/run_full_yolo.py <<'PYEOF'
from ultralytics import YOLO
model = YOLO('yolo11l.pt')
model.train(epochs=30)
PYEOF
SCIENCEFLOW_RESOURCE_INTENT=gpu_train CUDA_VISIBLE_DEVICES=0 python3 tmp/run_full_yolo.py 2>&1"""

    err = mixed_file_write_execution_blocked_error(cmd)

    assert err is not None
    assert "Split it into two bash tool calls" in err


def test_mixed_heredoc_write_and_run_train_wrapper_is_blocked() -> None:
    cmd = """cat > tmp/run_train.py <<PYEOF
from train import main
score = main()
print(score)
PYEOF
python3 -u tmp/run_train.py 2>&1"""

    err = mixed_file_write_execution_blocked_error(cmd)

    assert err is not None
    assert "Split it into two bash tool calls" in err


def test_mixed_heredoc_write_and_same_python_script_execution_is_blocked() -> None:
    cmd = """cat > tmp/helper_script.py <<PYEOF
print("ok")
PYEOF
python3 tmp/helper_script.py"""

    err = mixed_file_write_execution_blocked_error(cmd)

    assert err is not None
    assert "Python script execution" in err


def test_pure_heredoc_file_write_is_not_blocked() -> None:
    cmd = """cat > tmp/run_full_yolo.py <<'PYEOF'
print('write only')
PYEOF"""

    assert mixed_file_write_execution_blocked_error(cmd) is None


@pytest.mark.parametrize(
    "cmd",
    [
        "bash run.sh",
        "cd /tmp/work && CUDA_VISIBLE_DEVICES=0 python3 main.py",
    ],
)
def test_ambiguous_exec_stays_unknown(cmd: str) -> None:
    assert classify_bash_command(cmd).resource_class == RESOURCE_UNKNOWN_EXEC


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        ("cd /tmp/work && python3 solution.py", "python3 solution.py"),
        ("CUDA_VISIBLE_DEVICES=0 python3 train.py", "python3 train.py"),
        ("env -u ENSEMBLE_PREDICT python3 solution.py", "python3 solution.py"),
        ("taskset -c 0-3 bash -c 'python3 train.py'", "python3 train.py"),
        ("timeout 60 bash -lc 'python3 solution.py'", "python3 solution.py"),
        ("uv run python -m pip show numpy", "python -m pip show numpy"),
    ],
)
def test_normalize_shell_command_unwraps_common_prefixes(cmd: str, expected: str) -> None:
    assert normalize_shell_command(cmd) == expected



@pytest.mark.parametrize("value", ["-1", "none", "cpu", "nodevfile"])
def test_bash_gpu_id_parser_treats_cpu_only_values_as_empty(value: str) -> None:
    assert _parse_visible_gpu_ids(value) == []




def test_cuda_visible_devices_logical_ordinals_map_to_task_physical_pool() -> None:
    value, ids, remapped, reason = _normalize_cuda_visible_devices_for_task_pool("0", ["6", "7"])

    assert value == "6"
    assert ids == ["6"]
    assert remapped is True
    assert reason == "logical_ordinal_mapped_to_task_physical_gpu"


def test_cuda_visible_devices_physical_subset_stays_physical() -> None:
    value, ids, remapped, reason = _normalize_cuda_visible_devices_for_task_pool("7", ["6", "7"])

    assert value == "7"
    assert ids == ["7"]
    assert remapped is False
    assert reason == "physical_subset"

def test_parallel_runner_treats_gpu_none_as_cpu_only(tmp_path) -> None:
    manifest = tmp_path / "tasks.yaml"
    workspace = tmp_path / "workspace"
    manifest.write_text(
        f"""
max_concurrent: 1
tasks:
  - exp_id: cpu-only-task
    run_id: cpu-only-task
    task: test task
    workspace: {workspace}
    gpu_list: none
""",
        encoding="utf-8",
    )

    runner = ParallelRunner(manifest, max_concurrent=1)
    spec = runner._tasks[0]

    assert spec.gpu_list_raw == "none"
    assert spec.gpu_list == ""
    assert spec.gpu_auto is False


def test_delete_guard_blocks_recursive_rm() -> None:
    err = dangerous_delete_command_blocked_error("rm -rf tmp/cache")

    assert err is not None
    assert "recursive rm" in err


def test_delete_guard_blocks_rm_prefix_before_resource_command() -> None:
    err = dangerous_delete_command_blocked_error(
        "rm -f tmp/submit_checkpoint.pkl 2>/dev/null; SCIENCEFLOW_RESOURCE_INTENT=gpu_train python3 predict.py"
    )

    assert err is not None
    assert "do not prefix" in err


def test_delete_guard_allows_single_file_cleanup_without_resource_command() -> None:
    assert dangerous_delete_command_blocked_error("rm -f tmp/old_checkpoint.pkl") is None
