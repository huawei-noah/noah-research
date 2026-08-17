# Scripts

`scripts/` only keeps the currently maintained entry scripts and a few reusable
manifests; new temporary experiment manifests should not be committed here.

## Canonical Manifests

| File | Purpose |
| --- | --- |
| [`prep.yaml`](prep.yaml) | Single-task data-prep agent manifest |
| [`lnr.yaml`](lnr.yaml) | Single-task LNR run manifest; defaults to the nomad2018 CPU-only 1h validation |
| [`lnr_nomad2018_3seed_cpu.yaml`](lnr_nomad2018_3seed_cpu.yaml) | nomad2018 3-seed CPU-only 1h validation manifest |
| [`lnr_two_tasks.yaml`](lnr_two_tasks.yaml) | Example LNR manifest running two tasks in parallel |
| [`repl.yaml`](repl.yaml) | Compact REPL example manifest |

## Utility Scripts

| File | Purpose |
| --- | --- |
| [`lnr_kill_resume.sh`](lnr_kill_resume.sh) | Kill by workspace, then continue with `--type lnr --resume` |
| [`monitor_lnr.sh`](monitor_lnr.sh) | Launcher for the LNR manifest monitoring UI; read-only monitoring, does not start tasks |

## Task Package Evaluators

`scripts/` no longer keeps Python evaluator wrappers. When adding a task, place the
system-side evaluator at `tasks/<category>/<suite>/<task>/evaluator.py` or
`tasks/<category>/<task>/evaluator.py`, and use the `task_package` backend in the
manifest.

Historical opt-solver command manifests are kept only as archived records; legacy
evaluator wrappers are no longer maintained. To reproduce an experiment, migrate the
old manifest to a `task_package` evaluator first. Data preparation runs through the
data-prep agent of `scienceflow prep`, and task-shape rules live in `.scienceflow/skills/`.

## Examples

```bash
uv run python -m scienceflow.cli parallel -m scripts/prep.yaml -j 1

# For tasks that need prep output, set later LNR input_data_dir to:
# ./workspaces/manual_data_prep/<run_id>/<exp_id>/dataset

uv run python -m scienceflow.cli parallel -m scripts/lnr.yaml -j 1

uv run python -m scienceflow.cli parallel -m scripts/lnr_nomad2018_3seed_cpu.yaml -j 3

./scripts/monitor_lnr.sh scripts/lnr.yaml 5

./scripts/lnr_kill_resume.sh /path/to/<run_id>/<exp_id> \
  --config scienceflow/config/default.yaml \
  --input-data-dir ./data/mlebench_all_data/<exp_id>/prepared/public

uv run python -m scienceflow.cli parallel -m scripts/lnr_two_tasks.yaml -j 2
```

## Monitor Entry

`monitor_lnr.sh` is the monitoring entry for LNR run status, not a task execution
entry. It takes a manifest path, invokes the current `scienceflow.cli monitor`,
reads the LNR logs from the task directory the manifest points to, and refreshes
the display.

```bash
./scripts/monitor_lnr.sh <manifest.yaml> [refresh_sec]
```

- `<manifest.yaml>`: an LNR manifest that has already started or is about to be monitored.
- `[refresh_sec]`: optional refresh interval, default `5` seconds.
- The script only reads task logs/resource events/stage CSVs; it never starts, stops, or modifies tasks.
- Experiments are still launched with `uv run python -m scienceflow.cli parallel -m ...`.

## Configuration Notes

- A `scripts/` manifest only records run deltas: tasks, workspace, resource isolation,
  budget, and the LNR/REPL parameters you actually need to override. Stable values such
  as REPL profile, bash tool, workspace git, stage capture, estra, resource monitor,
  and log level come from `scienceflow/config/default.yaml`.
- Resource management in canonical manifests only exposes `lnr.resource_control_mode`;
  the default is `resource_smart_llm`. Do not expand low-level switches such as
  arbiter/advisory/GPU share at the script top level.
- The default REPL profile is `lite`; do not use the old `codex_like` naming.
- For multi-task experiments, put temporary manifests under
  `mlebench_workspaces/<date>_run/<run_name>/`; do not commit them to `scripts/`.
- `parallel` writes outer subprocess logs to each task workspace's
  `task_logs/parallel_subprocess.log` by default; do not use the repo-root
  `parallel_logs/` anymore.
- Evaluators for new tasks do not belong in `scripts/`. With the task package backend,
  the evaluator code is visible to the system but not to the agent; the agent only sees
  the task description, the data view, and the artifact contract.
- Data preparation is no longer exposed to top-level runs through a standalone
  deterministic split script; use the data-prep agent of `scienceflow prep`, and keep
  task-shape rules in `.scienceflow/skills/`.
- `prep.yaml` uses `phase: prep`, and the output directory is organized as
  `<workspace_base>/<run_id>/<exp_id>/dataset`. A later LNR run can point
  `input_data_dir` at this `dataset/`, or directly at an already trusted
  `prepared/public`.
- CPU isolation is declared per task with `cpu_list`, and `lnr.omp_threads_cap` should
  match the cores available to a single worker. For example, with 2 workers at 8 cores
  each: task-level `cpu_list: "0-15"`, `lnr.num_workers: 2`, `lnr.omp_threads_cap: 8`.
- GPU tasks declare `gpu_list: "0"` explicitly and keep the task-level
  `lnr.resource_gpu_pool` consistent, e.g. `["0"]`. CPU-only tasks use
  `gpu_list: "cpu"` and set `lnr.resource_gpu_pool: []`.
- The default nomad2018 validation setup uses
  `prepared/dataset_split/Deep`, `wall_clock_budget_sec: 3600`, 2 workers,
  8 CPUs per worker, CPU-only, and no skills.
- When two tasks run in parallel, `-j` should equal the number of concurrently running
  tasks; the worker count inside a task is controlled by task-level `lnr.num_workers`.

## Rules

- New temporary experiment manifests are not committed to `scripts/`; put them under
  `mlebench_workspaces/<date>_run/`.
- Prefer placing plotting and analysis scripts in `.codex/skills/`, an external
  experiment workspace, or a task-specific directory.
- If a script does not serve the current LNR tracking, resume, REPL, monitor, or
  two-task parallel entry points, do not put it back into `scripts/`.
