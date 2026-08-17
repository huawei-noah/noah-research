# ScienceFlow Config Examples

This directory contains copy-and-edit examples only. Runtime defaults are loaded from
`scienceflow/config/default.yaml`, which includes stable blocks from `scienceflow/config/defaults/`.

Use these examples as templates:

- `lnr_config_overlay_example.yaml`: a real config overlay. It includes the repository default config and overrides only frequently tuned LNR fields.
- `tasks_lnr_parallel_example.yaml`: current LNR parallel manifest using `workspace_base + run_id + exp_id`.
- `tasks_example.yaml`: compact multi-task LNR manifest with model/key pool examples.
- `tasks_unified_evaluator_example.yaml`: one sequential launch manifest covering Nomad2018, an opt_solver task, and SciModelingBench TFBind8 through the shared task-package evaluator and gate.
- `tasks_sci_modeling_bench_tfbind8_example.yaml`: official SciModelingBench TFBind8 candidate-batch task using the shared authoritative evaluator and scientific-design gate.
- `tasks_prep_example.yaml`: optional legacy prep manifest. Current LNR runs usually point directly at `prepared/dataset_split/Deep` or another single dataset root.

Run the TFBind8 scientific-design example from the repository root:

```bash
uv sync --extra scientific-design
uv run python tasks/sci_modeling_bench/tfbind8-black-box-v1/prepare_data.py
uv run python -m scienceflow.cli parallel \
  -m scienceflow/config/examples/tasks_sci_modeling_bench_tfbind8_example.yaml -j 1
```

The generated `cache/sci_modeling_bench/tfbind8-v1/public` directory contains only the protocol-visible observations. The complete public upstream table is loaded by the trusted evaluator; isolation from worker network and Hugging Face cache is still required for leakage-resistant runs.

Resource control modes in new examples:

- `off`: disable resource control.
- `resource_smart_policy`: deterministic runtime/queue/policy arbiter without independent LLM resource judges.
- `resource_smart_llm`: default full smart monitoring with admission/arbiter LLM, main-agent advisory, and GPU share.

Path prefix convention in examples:

```text
./data/mlebench_all_data/<competition>/prepared/dataset_split/Deep
./workspaces/<date>_run/<run_name>
```

Replace `<competition>`, `<date>_run`, CPU/GPU ranges, and model/key settings for your machine.
