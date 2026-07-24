from __future__ import annotations

import sys
from pathlib import Path

LOCOMO_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "locomo"
if str(LOCOMO_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(LOCOMO_SCRIPT_DIR))
SCRIPT_FILES = [
    path for path in LOCOMO_SCRIPT_DIR.iterdir() if path.is_file() and path.suffix in {".sh", ".py", ".example", ".md"}
]

from locomo_vanilla_eval import build_parser  # noqa: E402


def test_add_eval_wrapper_invokes_direct_vanilla_add_runner() -> None:
    script = (LOCOMO_SCRIPT_DIR / "start_locomo_add_eval.sh").read_text(encoding="utf-8")

    assert "locomo_vanilla_eval.py" in script
    assert "--mode" in script
    assert "add" in script
    assert "--collection-prefix" in script
    assert "--dataset" in script
    assert "--conversation-concurrency" in script
    assert "--add-concurrency" in script
    assert "LOCOMO_ADD_CONCURRENCY" in script
    assert "run_experiments.py" not in script
    assert "--base_url" not in script
    assert "--api_key" not in script


def test_add_search_eval_wrapper_invokes_direct_vanilla_add_search_runner() -> None:
    script = (LOCOMO_SCRIPT_DIR / "start_locomo_add_search_eval.sh").read_text(encoding="utf-8")
    env = (LOCOMO_SCRIPT_DIR / "locomo_eval.env.example").read_text(encoding="utf-8")

    assert "locomo_vanilla_eval.py" in script
    assert "--mode" in script
    assert "add-search" in script
    assert "--collection-prefix" in script
    assert "--top-k" in script
    assert "--add-concurrency" in script
    assert "--search-concurrency" in script
    assert "--score-concurrency" in script
    assert "${LOCOMO_TOP_K}" in script
    assert "${LOCOMO_ADD_CONCURRENCY}" in script
    assert "${LOCOMO_SEARCH_CONCURRENCY}" in script
    assert "${LOCOMO_SCORE_CONCURRENCY}" in script
    assert 'export LOCOMO_TOP_K="50"' in env
    assert 'export LOCOMO_ADD_CONCURRENCY="4"' in env
    assert 'export LOCOMO_SEARCH_CONCURRENCY="8"' in env
    assert 'export LOCOMO_SCORE_CONCURRENCY="2"' in env
    assert "run_experiments.py" not in script
    assert "--base_url" not in script
    assert "--api_key" not in script


def test_vanilla_runner_imports_vanilla_add_and_unified_search_pipeline() -> None:
    source = (LOCOMO_SCRIPT_DIR / "locomo_vanilla_eval.py").read_text(encoding="utf-8")
    legacy_flat_alias = "non" + "_schema"

    assert "VanillaAddPipeline" in source
    assert "SearchPipelineImpl" in source
    assert "AddPipelineInput" in source
    assert "SearchPipelineInput" in source
    assert 'search_pipeline="vanilla"' in source
    assert "MemoryHttpApplicationClient" not in source
    assert "run_experiments.py" not in source
    assert legacy_flat_alias not in source


def test_locomo_tracked_scripts_use_vanilla_naming() -> None:
    legacy_flat_alias = "non" + "_schema"
    offenders = [path.name for path in SCRIPT_FILES if legacy_flat_alias in path.read_text(encoding="utf-8")]

    assert offenders == []


def test_vanilla_runner_parser_defaults_to_top50() -> None:
    parser = build_parser()

    args = parser.parse_args(["--mode", "add-search", "--collection-prefix", "locomo_test", "--output-dir", "out"])

    assert args.top_k == 50
    assert args.search_recall_size == 50
    assert args.add_concurrency == 4
    assert args.search_concurrency == 8
    assert args.score_concurrency == 2


def test_obsolete_http_service_and_search_wrappers_are_removed() -> None:
    removed = [
        "start_locomo_service.sh",
        "status_locomo_service.sh",
        "tail_locomo_service.sh",
        "start_locomo_search_eval.sh",
        "status_locomo_search_eval.sh",
        "tail_locomo_search_eval.sh",
    ]

    assert [name for name in removed if (LOCOMO_SCRIPT_DIR / name).exists()] == []


def test_direct_add_and_add_search_have_matching_tail_wrappers() -> None:
    add_status = (LOCOMO_SCRIPT_DIR / "status_locomo_add_eval.sh").read_text(encoding="utf-8")
    add_tail = (LOCOMO_SCRIPT_DIR / "tail_locomo_add_eval.sh").read_text(encoding="utf-8")
    add_search_status = (LOCOMO_SCRIPT_DIR / "status_locomo_add_search_eval.sh").read_text(encoding="utf-8")
    add_search_tail = (LOCOMO_SCRIPT_DIR / "tail_locomo_add_search_eval.sh").read_text(encoding="utf-8")

    assert "add_current.env" in add_status
    assert "add_current.env" in add_tail
    assert "add_search_current.env" in add_search_status
    assert "add_search_current.env" in add_search_tail
    assert "LOCOMO_ADD_PID_FILE" in add_status
    assert "LOCOMO_ADD_LOG" in add_tail
    assert "last_error" in add_status
    assert "LOCOMO_ADD_SEARCH_PID_FILE" in add_search_status
    assert "LOCOMO_ADD_SEARCH_LOG" in add_search_tail
    assert "last_error" in add_search_status
    assert "summary.json" in add_status
    assert "summary.json" in add_search_status


def test_locomo_scripts_do_not_reference_external_absolute_paths() -> None:
    forbidden_tokens = ["/Users/", "/private/tmp", "/tmp/"]

    offenders = {
        path.name: token
        for path in SCRIPT_FILES
        for token in forbidden_tokens
        if token in path.read_text(encoding="utf-8")
    }

    assert offenders == {}


def test_env_example_uses_project_relative_paths() -> None:
    env = (LOCOMO_SCRIPT_DIR / "locomo_eval.env.example").read_text(encoding="utf-8")

    assert "MEMOS_REPO=" not in env
    assert 'export LOCOMO_EVAL_HOME=".locomo-runs"' in env
    assert 'export LOCOMO_DATASET="datasets/locomo/locomo10.json"' in env
    assert 'export LOCOMO_COLLECTION_PREFIX="locomo_full_vanilla_add_search"' in env
    assert 'export LOCOMO_CLEAR_BEFORE_ADD="true"' in env
    assert 'export UV_CACHE_DIR=".locomo-runs/uv-cache"' in env
    obsolete_variables = [
        "LOCOMO_BASE_URL",
        "LOCOMO_API_KEY",
        "LOCOMO_ADD_MAX_WORKERS",
        "LOCOMO_ADD_STAGGER_DELAY",
        "LOCOMO_REQUEST_TIMEOUT",
        "LOCOMO_SERVICE_",
    ]
    assert [name for name in obsolete_variables if name in env] == []


def test_locomo_scripts_do_not_require_external_repo_root_variable() -> None:
    offenders = [
        path.name
        for path in SCRIPT_FILES
        if path.suffix in {".sh", ".example"}
        if "MEMOS_REPO" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_start_wrappers_pass_project_relative_paths_to_runners() -> None:
    offenders = [
        path.name
        for path in LOCOMO_SCRIPT_DIR.glob("start_locomo_*_eval.sh")
        if "${REPO_ROOT}/${LOCOMO_DATASET}" in path.read_text(encoding="utf-8")
        or "${REPO_ROOT}/${LOCOMO_EVAL_HOME}" in path.read_text(encoding="utf-8")
        or "${REPO_ROOT}/${UV_CACHE_DIR}" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
