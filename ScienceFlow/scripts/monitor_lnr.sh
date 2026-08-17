#!/usr/bin/env bash
# Monitor an LNR manifest with the current ScienceFlow CLI.
set -euo pipefail

if [[ $# -lt 1 ]] || [[ "${1:-}" == -* ]]; then
  echo "usage: $0 <manifest.yaml> [refresh_sec]" >&2
  exit 1
fi

MANIFEST="$1"
REFRESH_SEC="${2:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCIENCEFLOW_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCIENCEFLOW_ROOT"

uv run python -m scienceflow.cli monitor --manifest "$MANIFEST" --refresh "$REFRESH_SEC"
