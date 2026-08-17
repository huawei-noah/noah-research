#!/usr/bin/env bash
# Kill ScienceFlow CLI processes bound to one LNR workspace, then resume it.
#
# Usage:
#   ./scripts/lnr_kill_resume.sh /path/to/<run_id>/<exp_id> \
#       --config scienceflow/config/default.yaml \
#       --input-data-dir ./data/mlebench_all_data/<exp_id>/prepared/public
#   # Or pass a data-prep output directory:
#   #   --input-data-dir /path/to/data_prep/<run_id>/<exp_id>/dataset
#
# Options:
#   --dry-run          Print matching processes and resume command only.
#   --kill-only        Kill matching processes and do not resume.
#   --resume-only      Resume without killing first.
#   --resume-step N    Pass through to scienceflow run --resume-step.
#   --python CMD       Python command, default: python3. Example: "uv run python".
set -euo pipefail

if [[ $# -lt 1 ]] || [[ "${1:-}" == -* ]]; then
  echo "usage: $0 <workspace_dir> [--config PATH] [--input-data-dir PATH] [--resume-step N] [--dry-run|--kill-only|--resume-only] [--python CMD]" >&2
  exit 1
fi

if [[ ! -d "$1" ]]; then
  echo "error: workspace not a directory: $1" >&2
  exit 1
fi

WORKSPACE="$(cd "$1" && pwd)"
shift

CONFIG="scienceflow/config/default.yaml"
INPUT_DATA_DIR=""
RESUME_STEP=""
MODE=""
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:?}"
      shift 2
      ;;
    --input-data-dir)
      INPUT_DATA_DIR="${2:?}"
      shift 2
      ;;
    --resume-step)
      RESUME_STEP="${2:?}"
      shift 2
      ;;
    --dry-run|--kill-only|--resume-only)
      MODE="$1"
      shift
      ;;
    --python)
      PYTHON_BIN="${2:?}"
      shift 2
      ;;
    --python=*)
      PYTHON_BIN="${1#--python=}"
      shift
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCIENCEFLOW_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCIENCEFLOW_ROOT"

EXP_ID="$(basename "$WORKSPACE")"
mapfile -t TASK_MANIFESTS < <(find tasks -type f -path "*/${EXP_ID}/task.yaml" | sort)
if [[ "${#TASK_MANIFESTS[@]}" -ne 1 ]]; then
  echo "error: expected one task package at tasks/**/${EXP_ID}/task.yaml, found ${#TASK_MANIFESTS[@]}" >&2
  printf 'candidate: %s\n' "${TASK_MANIFESTS[@]}" >&2
  echo "hint: workspace basename should match exp_id, or pass a workspace ending in <exp_id>" >&2
  exit 1
fi
TASK_FILE="$(dirname "${TASK_MANIFESTS[0]}")/description_lite.md"
if [[ ! -f "$TASK_FILE" ]]; then
  echo "error: task description not found beside ${TASK_MANIFESTS[0]}" >&2
  exit 1
fi

CONFIG_ABS="$CONFIG"
if [[ "$CONFIG_ABS" != /* ]]; then
  CONFIG_ABS="$SCIENCEFLOW_ROOT/$CONFIG"
fi

echo "[info] workspace=$WORKSPACE"
echo "[info] exp_id=$EXP_ID"
echo "[info] config=$CONFIG_ABS"

kill_matches() {
  local pid cmdline found=0
  for pid in $(pgrep -f "scienceflow.cli" 2>/dev/null || true); do
    [[ -z "$pid" ]] && continue
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    if [[ "$cmdline" != *"--workspace"* ]] || [[ "$cmdline" != *"$WORKSPACE"* ]]; then
      continue
    fi
    found=1
    echo "[kill] pid=$pid scienceflow.cli workspace match"
    if [[ "$MODE" != "--dry-run" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done

  if [[ "$found" -eq 0 ]]; then
    echo "[info] no scienceflow.cli process matched this workspace"
    return 0
  fi
  if [[ "$MODE" == "--dry-run" ]]; then
    return 0
  fi

  sleep 1
  for pid in $(pgrep -f "scienceflow.cli" 2>/dev/null || true); do
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    if [[ "$cmdline" == *"--workspace"*"$WORKSPACE"* ]] || [[ "$cmdline" == *"--workspace $WORKSPACE"* ]]; then
      echo "[warn] pid=$pid still running, SIGKILL" >&2
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  echo "[info] kill pass done"
}

run_resume() {
  local task_body
  task_body="$(cat "$SCIENCEFLOW_ROOT/$TASK_FILE")"

  local -a python_cmd cmd
  read -r -a python_cmd <<< "$PYTHON_BIN"
  cmd=(
    "${python_cmd[@]}" -m scienceflow.cli run
    --task "$task_body"
    --workspace "$WORKSPACE"
    --type lnr
    --config "$CONFIG_ABS"
    --resume
  )
  if [[ -n "$INPUT_DATA_DIR" ]]; then
    cmd+=(--input-data-dir "$INPUT_DATA_DIR")
  fi
  if [[ -n "$RESUME_STEP" ]]; then
    cmd+=(--resume-step "$RESUME_STEP")
  fi

  echo "[info] exec: ${PYTHON_BIN} -m scienceflow.cli run --type lnr --resume --workspace $WORKSPACE"
  exec "${cmd[@]}"
}

case "${MODE:-}" in
  "")
    kill_matches
    run_resume
    ;;
  "--dry-run")
    kill_matches
    echo "[dry-run] would resume with: ${PYTHON_BIN} -m scienceflow.cli run --type lnr --resume --workspace $WORKSPACE"
    ;;
  "--kill-only")
    MODE=""
    kill_matches
    ;;
  "--resume-only")
    run_resume
    ;;
  *)
    echo "internal error: MODE=$MODE" >&2
    exit 1
    ;;
esac
