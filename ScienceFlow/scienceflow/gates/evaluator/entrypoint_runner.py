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

"""Subprocess runner for task-package evaluator entrypoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args(argv)

    try:
        config = json.loads(args.config_json)
        result = _call_entrypoint(
            entrypoint=Path(args.entrypoint),
            function_name=args.function,
            artifact_path=Path(args.artifact),
            workspace_dir=Path(args.workspace),
            task_dir=Path(args.task_dir),
            dataset_dir=Path(args.dataset_dir),
            config=config,
        )
    except Exception as exc:  # noqa: BLE001 - this is a process boundary.
        print(f"task evaluator error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _call_entrypoint(
    *,
    entrypoint: Path,
    function_name: str,
    artifact_path: Path,
    workspace_dir: Path,
    task_dir: Path,
    dataset_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    module = _load_module(entrypoint)
    fn = getattr(module, function_name, None)
    if not callable(fn):
        raise ValueError(f"entrypoint {entrypoint}:{function_name} is not callable")
    result = fn(
        artifact_path=artifact_path,
        workspace_dir=workspace_dir,
        task_dir=task_dir,
        dataset_dir=dataset_dir,
        config=config,
    )
    if not isinstance(result, dict):
        raise ValueError("task evaluator must return a JSON object")
    return result


def _load_module(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"entrypoint not found: {path}")
    module_name = f"_scienceflow_task_eval_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load evaluator module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
