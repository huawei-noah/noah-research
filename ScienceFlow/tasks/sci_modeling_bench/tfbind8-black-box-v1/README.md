# SciModelingBench TFBind8 task package

This package adapts the official SciModelingBench TFBind8 black-box optimization
task to ScienceFlow's `task_package` evaluator. It is a `scientific_design` task,
but its provider is `sci_modeling_bench`; it is not an official SciDesignBench
manifest item.

Prepare the pinned public protocol view:

```bash
uv sync --extra scientific-design
```

```bash
uv run python \
  tasks/sci_modeling_bench/tfbind8-black-box-v1/prepare_data.py
```

Run the example:

```bash
uv run python -m scienceflow.cli parallel \
  -m scienceflow/config/examples/tasks_sci_modeling_bench_tfbind8_example.yaml -j 1
```

The preparation output separates `public/`, which is used as `input_data_dir`,
from `validation/`, which contains a deterministic visible-data baseline for
smoke tests. The authoritative evaluator loads the complete table at the pinned
Hugging Face revision through `sci-modeling-bench==0.2.0`.

The complete table is public upstream. A leakage-resistant experiment must run
workers without external network access and without access to the evaluator's
Hugging Face cache. This adapter validates the unified evaluator contract; it
does not by itself provide process or cache isolation.
