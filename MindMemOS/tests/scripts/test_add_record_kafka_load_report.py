from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[2] / ".deploy" / "add_record_kafka_load.py"
    spec = importlib.util.spec_from_file_location("add_record_kafka_load", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_levels_and_render_report() -> None:
    module = _load_module()

    assert module._parse_levels("200,500,20000") == [200, 500, 20000]

    args = argparse.Namespace(
        env="dev_load_test",
        mode="both",
        total=20000,
        keys=200,
        flush=True,
        ensure_schema=True,
    )
    result = module.LevelResult(
        concurrency=200,
        total=20000,
        ok=20000,
        errors=0,
        delivery_errors=0,
        enqueue_wall=10.0,
        flush_wall=2.0,
        record_stats={"mean": 1.0, "p50": 1.0, "p90": 2.0, "p99": 3.0, "max": 4.0},
        enqueue_stats={"mean": 0.5, "p50": 0.5, "p90": 1.0, "p99": 1.5, "max": 2.0},
    )

    report = module._render_report(args, [result])

    assert "# add_record + Kafka 投递压测报告" in report
    assert "| 200 | 20000 | 20000 | 0 | 0 | 10.00 | 2000.0 | 2.00 | 1666.7 | 3.0 | 1.5 |" in report
