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

import asyncio
import json
import os
import signal
import time
import warnings

# Suppress jieba SyntaxWarning (Python 3.13 compatibility) before any imports
# that might trigger jieba loading (e.g., via deepcraft dependencies).
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module="jieba",
)

import click
from pathlib import Path
from typing import Any

import yaml

from scienceflow.core.agent.run_policy import AutoContinuePolicy
from scienceflow.config.settings import (
    apply_parallel_manifest_agent_overrides,
    apply_parallel_manifest_lnr_overrides,
    apply_profile_overrides,
    apply_repl_manifest_defaults,
    load_cfg,
    prep_cfg,
    Config,
)
from scienceflow.utils.env_bootstrap import bootstrap_dotenv
from scienceflow.utils.logging import setup_logging
from scienceflow.utils.workspace_git import (
    ensure_workspace_source_git,
    normalize_workspace_git_track_globs,
)

_SCIENCEFLOW_REPO_ROOT = Path(__file__).resolve().parents[1]


def _apply_task_workspace(cfg: Config, workspace: str | Path) -> None:
    cfg.task_workspace_root_dir = Path(workspace).expanduser().resolve(strict=False)


def _apply_parallel_manifest_cfg_env(cfg: Config) -> None:
    """Merge ``SCIENCEFLOW_PARALLEL_MANIFEST_CFG_JSON`` from :class:`~scienceflow.core.parallel_runner.ParallelRunner`."""
    raw = os.environ.get("SCIENCEFLOW_PARALLEL_MANIFEST_CFG_JSON")
    if not raw:
        return
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return
    if isinstance(data, dict):
        apply_repl_manifest_defaults(cfg, data)


def _bootstrap_env() -> None:
    """Load ``<repo>/.env`` then cwd ``.env`` (same order as ensemble replay).

    Set ``SCIENCEFLOW_DOTENV_OVERRIDE=1`` so repo-root ``.env`` replaces stale shell exports.
    """
    bootstrap_dotenv(repo_root=_SCIENCEFLOW_REPO_ROOT)


def _is_repl_exit_line(task: str) -> bool:
    """True for exit / quit and common variants (e.g. ``exit()`` from Python habit)."""
    t = task.strip().lower().rstrip(";").strip()
    if not t:
        return False
    if t.endswith("()"):
        t = t[:-2].strip()
    return t in ("exit", "quit", "q")


def _repl_apply_dataset_symlinks(cfg: Config) -> None:
    """Symlink cfg.input_data_dir into workspace/dataset/ (same as run/prep fast paths)."""
    from scienceflow.solver.lnr.prep_fs import (
        prepare_workspace_dataset_flat,
        resolve_workspace_dataset_source,
    )

    inp = Path(cfg.input_data_dir).expanduser().resolve(strict=False)
    if not str(inp).strip():
        return
    ws_dataset = Path(cfg.workspace_dir) / "dataset"
    if not ws_dataset.exists():
        source, _layout = resolve_workspace_dataset_source(inp)
        prepare_workspace_dataset_flat(source, ws_dataset)
    _repl_allow_input_data_root(cfg)


def _repl_allow_input_data_root(cfg: Config) -> Path | None:
    """Allow the code agent to read cfg.input_data_dir without exposing it as output."""
    inp = Path(cfg.input_data_dir).expanduser().resolve(strict=False)
    if not str(inp).strip():
        return None
    seen_pg = {
        Path(p).expanduser().resolve(strict=False)
        for p in (cfg.path_guard_extra_roots or [])
        if str(p).strip()
    }
    ir = inp.resolve()
    if ir not in seen_pg:
        cfg.path_guard_extra_roots = list(cfg.path_guard_extra_roots or [])
        cfg.path_guard_extra_roots.append(ir)
    return ir


def _repl_resolve_task_text(cfg: Config, *, manifest_task_raw: Any = None) -> tuple[str, str]:
    """Return (task_text, source) where source is 'file', 'resolved', or ''."""
    inner = Path(cfg.workspace_dir)
    for name in ("description.md", "task_desc.txt"):
        td = inner / name
        if td.is_file() or td.is_symlink():
            t = td.read_text(encoding="utf-8").strip()
            if t:
                return t, "file"
    from scienceflow.core.parallel_runner import _resolve_parallel_task_text

    exp = (cfg.exp_id or "").strip()
    if not exp:
        return "", ""
    try:
        t = _resolve_parallel_task_text(exp, manifest_task_raw).strip()
        if t:
            return t, "resolved"
    except ValueError:
        pass
    return "", ""


def _repl_materialize_task_desc(cfg: Config, text: str, source: str) -> None:
    if not text or source == "file":
        return
    out = Path(cfg.workspace_dir) / "description.md"
    if not out.exists():
        out.write_text(text, encoding="utf-8")


def _truthy_yaml(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _manifest_bool(
    defaults: dict[str, Any],
    task: dict[str, Any],
    key: str,
    fallback: bool = False,
) -> bool:
    for block in (task, defaults):
        if not isinstance(block, dict) or key not in block:
            continue
        parsed = _truthy_yaml(block.get(key))
        if parsed is not None:
            return parsed
    return fallback


def _manifest_str(defaults: dict[str, Any], task: dict[str, Any], key: str) -> str:
    for block in (task, defaults):
        if not isinstance(block, dict) or key not in block:
            continue
        value = block.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _repl_normalized_profile(value: Any) -> str:
    raw = str(value or "lite").strip().lower().replace("-", "_")
    aliases = {
        "lite": "lite",
        "legacy": "legacy",
    }
    if raw not in aliases:
        allowed = ", ".join(sorted(set(aliases)))
        raise click.ClickException(f"Invalid repl_profile {value!r}; expected one of: {allowed}")
    return aliases[raw]


def _repl_normalized_tool_preset(value: Any, *, profile: str) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if not raw:
        return "write_edit" if profile == "legacy" else "bash_write"
    aliases = {
        "bash": "bash_write",
        "bash_only": "bash_write",
        "bash_write": "bash_write",
        "write": "write_edit",
        "write_edit": "write_edit",
        "legacy": "write_edit",
    }
    if raw not in aliases:
        allowed = ", ".join(sorted(set(aliases)))
        raise click.ClickException(
            f"Invalid repl_tool_preset {value!r}; expected one of: {allowed}",
        )
    return aliases[raw]


def _repl_positive_int(value: Any, *, fallback: int, key: str) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError) as exc:
        raise click.ClickException(f"{key} must be an integer, got {value!r}") from exc
    return parsed if parsed > 0 else int(fallback)


def _repl_resolve_runtime_options(
    cfg: Config,
    *,
    auto_first_user_enabled: bool,
) -> dict[str, Any]:
    """Resolve REPL-only runtime profile."""
    profile = _repl_normalized_profile(getattr(cfg, "repl_profile", "lite"))
    tool_preset = _repl_normalized_tool_preset(
        getattr(cfg, "repl_tool_preset", ""),
        profile=profile,
    )
    max_steps = _repl_positive_int(
        getattr(cfg, "repl_max_steps", 0),
        fallback=int(getattr(cfg, "qa_max_steps", 20) or 20),
        key="repl_max_steps",
    )
    bash_max_output_chars = _repl_positive_int(
        getattr(cfg, "repl_bash_max_output_chars", 0),
        fallback=8000,
        key="repl_bash_max_output_chars",
    )
    bash_max_stream_line_chars = _repl_positive_int(
        getattr(cfg, "repl_bash_max_stream_line_chars", 0),
        fallback=2400,
        key="repl_bash_max_stream_line_chars",
    )
    bash_observation_summary = bool(
        getattr(cfg, "repl_bash_observation_summary", profile != "legacy"),
    )
    stable_system_prompt = bool(
        getattr(cfg, "repl_stable_system_prompt", profile != "legacy"),
    )
    pin_environment_context = bool(
        getattr(cfg, "repl_pin_environment_context", profile != "legacy"),
    )
    code_organization_hint = str(
        getattr(cfg, "repl_code_organization_hint", "") or "",
    ).strip()
    workspace_git_enabled = bool(
        getattr(cfg, "repl_workspace_git_enabled", profile != "legacy"),
    )
    if profile == "legacy":
        workspace_git_enabled = False
    workspace_git_track_globs = normalize_workspace_git_track_globs(
        getattr(cfg, "repl_workspace_git_track_globs", None),
    )
    workspace_git_auto_review = bool(
        getattr(cfg, "repl_workspace_git_auto_review", False),
    )
    workspace_git_auto_checkpoint = bool(
        getattr(cfg, "repl_workspace_git_auto_checkpoint", workspace_git_enabled),
    )
    if not workspace_git_enabled:
        workspace_git_auto_checkpoint = False
    pin_task_when_auto = bool(
        getattr(cfg, "repl_pin_task_description_when_auto_first_user", False),
    )
    return {
        "profile": profile,
        "tool_preset": tool_preset,
        "max_steps": max_steps,
        "stable_system_prompt": stable_system_prompt,
        "pin_environment_context": pin_environment_context,
        "code_organization_hint": code_organization_hint,
        "workspace_git_enabled": workspace_git_enabled,
        "workspace_git_track_globs": list(workspace_git_track_globs),
        "workspace_git_auto_review": workspace_git_auto_review,
        "workspace_git_auto_checkpoint": workspace_git_auto_checkpoint,
        "pin_task_description": (not auto_first_user_enabled) or pin_task_when_auto,
        "repl_bash_write_mode": tool_preset == "bash_write",
        "bash_max_output_chars": bash_max_output_chars,
        "bash_max_stream_line_chars": bash_max_stream_line_chars,
        "bash_observation_summary": bash_observation_summary,
    }


def _read_manifest_text_file(path_value: str, *, manifest_dir: Path | None) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    path = Path(raw).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if manifest_dir is not None:
            candidates.append(manifest_dir / path)
        candidates.append(_SCIENCEFLOW_REPO_ROOT / path)
        candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    tried = ", ".join(str(p) for p in candidates)
    raise click.ClickException(f"Manifest text file not found for {raw!r}; tried: {tried}")


def _repl_build_first_user_query(task_text: str) -> str:
    """Build the automatic first REPL turn without legacy solver prompt dependencies."""
    body = (task_text or "").strip()
    if body:
        return (
            "Design and implement a strong solution for this task. Work inside the current "
            "workspace, inspect the available dataset under `dataset/`, create the necessary "
            "source files, run validation, and produce the final submission artifact.\n\n"
            "## Task\n\n"
            f"{body}"
        )
    return (
        "Design and implement a strong solution for the task available in this workspace. "
        "Inspect `dataset/`, create the necessary source files, run validation, and produce "
        "the final submission artifact."
    )


def _install_cleanup_signals() -> None:
    """Register SIGINT/SIGTERM/SIGHUP handlers that sweep all live child PGIDs.

    Called once at CLI entry.  The handler sends SIGTERM to every process
    group in the live registry, waits 5 s, then sends SIGKILL to survivors,
    and finally reinstates the default handler so the process exits normally.
    """
    from scienceflow.core.subprocess_utils import _LIVE_PGIDS

    def _handler(signum: int, frame: object) -> None:
        # SIGTERM first pass
        for pgid in list(_LIVE_PGIDS):
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        # Give processes a chance to clean up (DDP destroy_process_group etc.)
        time.sleep(5)
        # SIGKILL survivors
        for pgid in list(_LIVE_PGIDS):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        # Restore default and re-raise so the process exits with correct status
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError):
            pass  # SIGHUP unavailable on some platforms


@click.group()
def main():
    """ScienceFlow - Autonomous ML Agent Framework"""
    _bootstrap_env()
    _install_cleanup_signals()


@main.command()
@click.option("--task", "-t", required=True, help="Task description")
@click.option("--config", "-c", default=None, help="Config YAML path")
@click.option(
    "--workspace",
    "-w",
    default=".",
    help="Task output root (LNR uses task_logs/ plus worker or inner workspace directories)",
)
@click.option(
    "--input-data-dir",
    "-d",
    "input_data_dir",
    default=None,
    type=click.Path(exists=False),
    help="Override config input_data_dir (read-only raw competition input)",
)
@click.option(
    "--show-preview/--no-show-preview",
    default=False,
    help="Print truncated data layout after preparing dataset",
)
@click.option(
    "--agent/--link-only",
    "run_agent",
    default=True,
    help="Run the data-prep agent after exposing input data; --link-only only symlinks.",
)
def prep(task, config, workspace, input_data_dir, show_preview, run_agent):
    """Prepare a REPL workspace dataset view, optionally via the data-prep agent."""
    cfg = load_cfg(config)
    _apply_parallel_manifest_cfg_env(cfg)
    _apply_task_workspace(cfg, workspace)
    if input_data_dir is not None:
        cfg.input_data_dir = Path(input_data_dir).expanduser().resolve(strict=False)
    apply_profile_overrides(cfg)
    apply_parallel_manifest_agent_overrides(cfg)
    prep_cfg(cfg)
    setup_logging(cfg.log_dir, interaction_log_color=cfg.scienceflow_interaction_log_color)
    if run_agent:
        _repl_allow_input_data_root(cfg)
        ws_dataset = Path(cfg.workspace_dir) / "dataset"
        if ws_dataset.is_symlink():
            ws_dataset.unlink()
        ws_dataset.mkdir(parents=True, exist_ok=True)
        from scienceflow.solver.data_prep import run_data_prep_agent

        asyncio.run(run_data_prep_agent(cfg, task, repo_root=_SCIENCEFLOW_REPO_ROOT))
        click.echo("prep agent ok")
    else:
        _repl_apply_dataset_symlinks(cfg)
    click.echo(f"prep ok: dataset={Path(cfg.workspace_dir) / 'dataset'}")
    if show_preview:
        from scienceflow.utils.data_scan import scan_data_dir

        preview = scan_data_dir(
            Path(cfg.workspace_dir) / "dataset",
            max_chars=int(getattr(cfg, "data_preview_max_chars", 32000) or 32000),
            max_items_per_dir=int(getattr(cfg, "data_preview_max_items_per_dir", 40) or 40),
            walk_budget_dirs=getattr(cfg, "data_scan_walk_budget_dirs", None),
            walk_budget_files=getattr(cfg, "data_scan_walk_budget_files", None),
            probe_binary_dirs_budget=int(getattr(cfg, "data_scan_probe_binary_dirs_budget", 8) or 8),
            preview_raw_dirs_budget=int(getattr(cfg, "data_scan_preview_raw_dirs_budget", 12) or 12),
            meta_sample_max_bytes=int(getattr(cfg, "data_scan_meta_sample_max_bytes", 50000) or 50000),
            csv_max_rows_to_scan=int(getattr(cfg, "data_scan_csv_max_rows_to_scan", 200000) or 200000),
        )
        click.echo("--- data preview ---")
        click.echo(preview)


@main.command()
@click.option("--task", "-t", required=True, help="Task description")
@click.option("--config", "-c", default=None, help="Config YAML path")
@click.option(
    "--workspace",
    "-w",
    default=".",
    help="Task output root (LNR runs create indexed worker execution dirs under it)",
)
@click.option(
    "--input-data-dir",
    "-d",
    "input_data_dir",
    default=None,
    type=click.Path(exists=False),
    help="Override config input_data_dir (read-only raw competition input)",
)
@click.option(
    "--type",
    "task_type",
    default="lnr",
    type=click.Choice(["lnr"]),
    help="REPL-native long-horizon solver.",
)
def run(
    task,
    config,
    workspace,
    input_data_dir,
    task_type,
):
    """Run a REPL-native long-horizon task."""
    cfg = load_cfg(config)
    _apply_parallel_manifest_cfg_env(cfg)
    _apply_task_workspace(cfg, workspace)
    if input_data_dir is not None:
        cfg.input_data_dir = Path(input_data_dir).expanduser().resolve(strict=False)
    if task_type == "lnr":
        setattr(
            cfg,
            "_log_dir_override",
            Path(cfg.task_workspace_root_dir).expanduser().resolve(strict=False) / "task_logs",
        )
    apply_profile_overrides(cfg)
    apply_parallel_manifest_lnr_overrides(cfg)
    apply_parallel_manifest_agent_overrides(cfg)
    prep_cfg(cfg)
    setup_logging(cfg.log_dir, interaction_log_color=cfg.scienceflow_interaction_log_color)

    from scienceflow.core.orchestrator import Orchestrator
    orch = Orchestrator(cfg)
    result = asyncio.run(orch.run(task, task_type))
    click.echo(f"Result: {result}")
    if isinstance(result, dict) and str(result.get("status") or "").lower() not in {"", "success"}:
        raise SystemExit(1)


@main.command()
@click.option("--config", "-c", default=None, help="Config YAML path")
@click.option(
    "--workspace",
    "-w",
    default=None,
    type=click.Path(exists=False),
    help="Task output root (omit when using --manifest for this command).",
)
@click.option(
    "--manifest",
    "-m",
    "manifest_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Single-task YAML: derive -w and input_data_dir like parallel (mutually exclusive with -w).",
)
@click.option(
    "--exp-id",
    "exp_id_cli",
    default=None,
    help="Override cfg.exp_id (task description lookup via tasks/**/<exp_id>/task.yaml).",
)
@click.option("--plain", is_flag=True, default=False, help="Disable rich UI (plain text mode)")
@click.option(
    "--show-code",
    type=click.IntRange(min=-1),
    default=5,
    help="Code preview lines per step: 5 (default), 0=hide, -1=full code",
)
@click.option(
    "--enable-sandbox/--no-enable-sandbox",
    default=True,
    help="Naive agent: restrict read/write/edit/grep/glob/ls to the execution workspace (default on). "
    "Use --no-enable-sandbox for arbitrary paths (e.g. external datasets).",
)
@click.option(
    "--input-data-dir",
    "-d",
    "input_data_dir",
    default=None,
    type=click.Path(exists=False),
    help="Read-only competition data root; auto-symlinked to workspace/dataset/ (same layout as run/prep).",
)
@click.option(
    "--auto-first-user/--no-auto-first-user",
    default=None,
    help="Automatically run one REPL turn using the manifest first-user query or a REPL-native task prompt.",
)
@click.option(
    "--exit-after-auto/--no-exit-after-auto",
    default=None,
    help="Exit after --auto-first-user completes instead of entering the interactive prompt.",
)
def repl(
    config,
    workspace,
    manifest_path,
    exp_id_cli,
    plain,
    show_code,
    enable_sandbox,
    input_data_dir,
    auto_first_user,
    exit_after_auto,
):
    """Interactive REPL mode."""
    from scienceflow.core.parallel_runner import (
        _manifest_input_data_dir,
        _manifest_task_exp_id,
        resolve_manifest_task_workspace,
    )

    manifest_task_raw: Any = None
    defaults: dict[str, Any] = {}
    task0: dict[str, Any] = {}
    manifest_dir: Path | None = None
    repl_first_user_query_text = ""
    link_data = False
    if manifest_path:
        if workspace is not None:
            raise click.ClickException("Use either --manifest or --workspace, not both.")
        manifest_file = Path(manifest_path).expanduser().resolve(strict=False)
        manifest_dir = manifest_file.parent
        with open(manifest_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        tasks = data.get("tasks") or []
        if len(tasks) != 1:
            raise click.ClickException("repl --manifest requires exactly one task in tasks[]")
        defaults = data.get("defaults") or {}
        task0 = tasks[0]
        exp_id = _manifest_task_exp_id(task0, 0)
        resolved_ws = resolve_manifest_task_workspace(defaults, task0, 0, exp_id)
        cfg_yaml = config
        if cfg_yaml is None:
            tcfg = task0.get("config")
            dcfg = defaults.get("config")
            cfg_yaml = tcfg if (tcfg is not None and str(tcfg).strip()) else dcfg
        cfg = load_cfg(cfg_yaml)
        apply_repl_manifest_defaults(cfg, defaults, task0)
        apply_profile_overrides(cfg)
        _apply_task_workspace(cfg, resolved_ws)
        idd_m = _manifest_input_data_dir(defaults, task0)
        if idd_m.strip():
            cfg.input_data_dir = Path(idd_m).expanduser().resolve(strict=False)
            link_data = True
        cfg.exp_id = exp_id
        manifest_task_raw = task0.get("task")
        first_user_file = _manifest_str(defaults, task0, "repl_first_user_query_file")
        if first_user_file:
            repl_first_user_query_text = _read_manifest_text_file(
                first_user_file,
                manifest_dir=manifest_dir,
            )
    else:
        ws = workspace if workspace is not None else "."
        cfg = load_cfg(config)
        apply_profile_overrides(cfg)
        _apply_task_workspace(cfg, ws)

    if input_data_dir is not None:
        cfg.input_data_dir = Path(input_data_dir).expanduser().resolve(strict=False)
        link_data = True

    if exp_id_cli is not None and str(exp_id_cli).strip():
        cfg.exp_id = str(exp_id_cli).strip()

    prep_cfg(cfg)
    cfg.scienceflow_tools_sandbox = enable_sandbox

    if link_data:
        _repl_apply_dataset_symlinks(cfg)

    td_text, td_src = _repl_resolve_task_text(cfg, manifest_task_raw=manifest_task_raw)
    _repl_materialize_task_desc(cfg, td_text, td_src)
    git_track_globs = normalize_workspace_git_track_globs(
        getattr(cfg, "repl_workspace_git_track_globs", None),
    )
    cfg.repl_workspace_git_track_globs = list(git_track_globs)
    git_result = ensure_workspace_source_git(
        cfg.workspace_dir,
        enabled=bool(getattr(cfg, "repl_workspace_git_enabled", True)),
        track_globs=git_track_globs,
        initial_commit=bool(getattr(cfg, "repl_workspace_git_initial_commit", True)),
    )
    if git_result.enabled and not git_result.ready:
        cfg.repl_workspace_git_enabled = False
        if git_result.message:
            click.echo(f"[repl-git] workspace git unavailable: {git_result.message}", err=True)
    auto_first_user_enabled = (
        _manifest_bool(
            defaults,
            task0,
            "repl_auto_first_user",
            bool(repl_first_user_query_text),
        )
        if auto_first_user is None
        else bool(auto_first_user)
    )
    exit_after_auto_enabled = (
        _manifest_bool(defaults, task0, "repl_exit_after_auto", False)
        if exit_after_auto is None
        else bool(exit_after_auto)
    )
    repl_runtime = _repl_resolve_runtime_options(
        cfg,
        auto_first_user_enabled=auto_first_user_enabled,
    )

    # File-only: avoid interleaving INFO lines with Rich REPL output
    setup_logging(
        cfg.log_dir,
        console=False,
        interaction_log_color=cfg.scienceflow_interaction_log_color,
    )

    from scienceflow.core.llm_http import aclose_llm_clients as _aclose_llm_clients
    from scienceflow.core.orchestrator import Orchestrator
    orch = Orchestrator(cfg)

    ui = None
    if not plain:
        from scienceflow.ui import RichUI
        ui = RichUI(show_code=show_code)

    if ui:
        ui.welcome()
    else:
        click.echo("ScienceFlow REPL (type exit, quit, or exit() to quit)")

    async def _close_llm(llm) -> None:
        await _aclose_llm_clients(llm)

    repl_run_id = 0
    system_prompt_override = None
    system_prompt_hook = None
    repl_teleport_mode = "off"
    llm_trace_hook = orch.make_llm_call_tracer(
        node_id="repl",
        process_id=task0.get("run_id") if isinstance(task0, dict) else None,
        detail_prefix=(
            f"mode=repl;profile={repl_runtime['profile']};"
            f"tool_preset={repl_runtime['tool_preset']};teleport={repl_teleport_mode}"
        ),
    )
    science_agent = orch.create_science_agent(
        ui=ui,
        task_description=td_text or None,
        run_policy=AutoContinuePolicy(max_text_only_retries=2),
        max_steps_override=int(repl_runtime["max_steps"]),
        system_prompt=system_prompt_override,
        system_prompt_hook=system_prompt_hook,
        append_repl_system_prompt=True,
        pin_task_description=bool(repl_runtime["pin_task_description"]),
        on_llm_call=llm_trace_hook,
        teleport_mode=repl_teleport_mode,
        repl_bash_write_mode=bool(repl_runtime["repl_bash_write_mode"]),
        stable_system_prompt=bool(repl_runtime["stable_system_prompt"]),
        pin_environment_context=bool(repl_runtime["pin_environment_context"]),
        code_organization_hint=str(repl_runtime["code_organization_hint"]),
        workspace_git_enabled=bool(repl_runtime["workspace_git_enabled"]),
        workspace_git_track_globs=repl_runtime["workspace_git_track_globs"],
        workspace_git_auto_review=bool(repl_runtime["workspace_git_auto_review"]),
        workspace_git_auto_checkpoint=bool(
            repl_runtime["workspace_git_auto_checkpoint"],
        ),
        bash_max_output_chars_override=int(repl_runtime["bash_max_output_chars"]),
        bash_max_stream_line_chars_override=int(
            repl_runtime["bash_max_stream_line_chars"],
        ),
        bash_observation_summary_override=bool(
            repl_runtime["bash_observation_summary"],
        ),
    )
    try:
        if auto_first_user_enabled:
            auto_task = repl_first_user_query_text or _repl_build_first_user_query(td_text)
            msg = "file first-user query" if repl_first_user_query_text else "REPL first-user query"
            click.echo(f"[repl-auto] running {msg}")
            repl_run_id += 1
            science_agent.set_repl_session_run_index(repl_run_id)
            asyncio.run(science_agent.run(auto_task))
            if exit_after_auto_enabled:
                return

        while True:
            if ui:
                task = ui.prompt_input()
                if task is None:
                    break
            else:
                try:
                    task = click.prompt("scienceflow", prompt_suffix="> ")
                except (EOFError, KeyboardInterrupt):
                    break

            if _is_repl_exit_line(task):
                break

            stripped = task.strip()
            if stripped == "/compact":
                out = asyncio.run(science_agent.compact())
                click.echo(out)
                continue
            if stripped == "/files":
                click.echo(science_agent.file_state_summary)
                continue

            repl_run_id += 1
            science_agent.set_repl_session_run_index(repl_run_id)
            asyncio.run(science_agent.run(task))
    finally:
        asyncio.run(_close_llm(science_agent.llm))

    if ui:
        ui.goodbye()


@main.command()
@click.option("--manifest", "-m", required=True, help="Task manifest YAML path")
@click.option("--max-concurrent", "-j", default=None, type=int, help="Override max concurrent tasks")
@click.option(
    "--log-dir",
    default=None,
    help="Legacy external subprocess log directory; by default logs stay in each task workspace.",
)
def parallel(manifest, max_concurrent, log_dir):
    """Run multiple tasks in parallel with per-task CPU/GPU isolation."""
    from scienceflow.core.parallel_runner import ParallelRunner

    try:
        runner = ParallelRunner(manifest, max_concurrent=max_concurrent, log_dir=log_dir)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Launching {len(runner._tasks)} tasks (max_concurrent={runner._max_concurrent})")

    results = asyncio.run(runner.run_all())
    click.echo("\n" + runner.format_summary(results))

    failed = [r for r in results if r.status not in ("success", "skipped")]
    if failed:
        raise SystemExit(1)


@main.command("monitor")
@click.option(
    "--log-dir",
    "-l",
    default=None,
    type=click.Path(exists=False),
    help="Log directory containing monitor_state.json (single task)",
)
@click.option(
    "--manifest",
    "-m",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Parallel manifest YAML (multi-task: one row per task)",
)
@click.option(
    "--refresh",
    "-r",
    default=2.0,
    show_default=True,
    help="Refresh interval in seconds",
)
def monitor_cmd(log_dir, manifest, refresh):
    """Watch REPL-native long-horizon run(s) via a Rich Live dashboard."""
    from scienceflow.ui.monitor import MonitorDashboard, parse_monitor_manifest

    if bool(log_dir) == bool(manifest):
        raise click.UsageError("Specify exactly one of --log-dir or --manifest.")

    if manifest:
        entries = parse_monitor_manifest(manifest)
        if not entries:
            raise click.ClickException("Manifest has no tasks with workspace paths.")
        dashboard = MonitorDashboard(state_file=None, refresh_interval=refresh, task_entries=entries)
    else:
        state_file = Path(log_dir) / "monitor_state.json"
        dashboard = MonitorDashboard(state_file, refresh)
    dashboard.run()


@main.command("monitor-trace")
@click.option(
    "--manifest",
    "-m",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Parallel manifest YAML to visualize.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(dir_okay=False),
    help="HTML file to write.",
)
@click.option(
    "--cache",
    default=None,
    type=click.Path(dir_okay=False),
    help="Optional JSON cache path. Defaults to monitor_trace_data.json next to output.",
)
@click.option(
    "--refresh",
    "-r",
    default=60.0,
    show_default=True,
    help="Refresh interval in seconds.",
)
@click.option("--once", is_flag=True, help="Render once and exit.")
def monitor_trace_cmd(manifest, output, cache, refresh, once):
    """Write a self-refreshing HTML trend view from monitor task logs."""
    from scienceflow.ui.monitor_trace import run_monitor_trace

    run_monitor_trace(
        manifest_path=manifest,
        output_path=output,
        cache_path=cache,
        refresh_sec=refresh,
        once=once,
    )


@main.command("resource-summary")
@click.argument("root", type=click.Path(exists=True))
@click.option("--max-files", default=256, show_default=True, help="Maximum resource_events.jsonl files to scan.")
@click.option("--max-events-per-file", default=None, type=int, help="Optional cap for events read per file.")
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON instead of a table.")
def resource_summary_cmd(root, max_files, max_events_per_file, json_output):
    """Print a run-level summary from resource_events.jsonl files."""
    from scienceflow.solver.lnr.resource_runtime.unified_store import (
        format_resource_run_summary,
        summarize_resource_run,
    )

    summary = summarize_resource_run(root, max_files=max_files, max_events_per_file=max_events_per_file)
    if json_output:
        click.echo(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        click.echo(format_resource_run_summary(summary))


@main.command("replay-prepare")
@click.option(
    "--source",
    "-s",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Saved run or bad-case directory to copy.",
)
@click.option(
    "--out",
    "-o",
    "output",
    required=True,
    type=click.Path(exists=False, file_okay=False),
    help="New isolated replay directory to create.",
)
@click.option(
    "--message-index",
    default=None,
    type=int,
    help="Zero-based memory record index containing the assistant tool call.",
)
@click.option(
    "--tool-call-id",
    default=None,
    help="Tool call id to replay. Mutually exclusive with --message-index.",
)
@click.option(
    "--memory",
    default=None,
    type=click.Path(exists=False, dir_okay=False),
    help="Optional short_term.json or long_term.jsonl source path, relative to copied root or absolute under source.",
)
@click.option(
    "--manifest-name",
    default="replay.yaml",
    show_default=True,
    help="Manifest filename under the copied root to patch.",
)
@click.option(
    "--patch-manifest/--no-patch-manifest",
    default=True,
    show_default=True,
    help="Patch workspace_base in the copied manifest to point at <out>/run.",
)
def replay_prepare_cmd(
    source,
    output,
    message_index,
    tool_call_id,
    memory,
    manifest_name,
    patch_manifest,
):
    """Copy a saved LNR run and truncate memory to replay one pending tool call."""
    from scienceflow.solver.lnr.replay_prepare import ReplayPrepareError, prepare_lnr_replay

    try:
        result = prepare_lnr_replay(
            source,
            output,
            message_index=message_index,
            tool_call_id=tool_call_id,
            memory_path=memory,
            manifest_name=manifest_name,
            patch_manifest=patch_manifest,
        )
    except ReplayPrepareError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
