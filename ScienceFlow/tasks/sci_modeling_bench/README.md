# SciModelingBench

ScienceFlow supports 12 scientific modeling and design tasks through the
[SciModelingBench](https://github.com/xukp20/sci-modeling-bench) task API and
the frozen datasets published on
[Hugging Face](https://huggingface.co/datasets/sci-modeling-bench/design-bench).
Each task exposes observations and a submission contract to the agent, while a
system-side evaluator keeps candidate outcomes hidden and scores ordered
candidate batches.

## Tasks

`N` is the submitted batch size and `K` is the leading subset summarized by the
task metric. Query limits below are per worker when using the included
SciModelingBench profile.

| Task id | Protocol | Primary metric | N / K | Queries |
|---|---|---:|---:|---:|
| `sci-modeling-bench-tfbind8` | Black-box optimization | `best_k_mean` | 32 / 5 | 10 |
| `sci-modeling-bench-tfbind10-pho4` | Black-box optimization | `normalized_enrichment` | 128 / 16 | 10 |
| `sci-modeling-bench-superconductor` | Candidate-pool ranking | `global_ndcg` | 32 / 5 | 10 |
| `sci-modeling-bench-utr-mrl` | Candidate-pool ranking | `normalized_enrichment` | 128 / 16 | 5 |
| `sci-modeling-bench-gfp` | Candidate-pool ranking | `normalized_enrichment` | 128 / 16 | 10 |
| `sci-modeling-bench-hopper-controller` | Candidate-pool ranking | `global_ndcg` | 32 / 5 | 5 |
| `sci-modeling-bench-drugmatrix-{mchc,mch,creatinine,sodium,chloride,phosphorus}` | Candidate-pool ranking | `global_ndcg` | 16 / 5 | 4 |

Black-box tasks accept valid designs from the declared domain. Ranking tasks
provide a candidate view with hidden labels and ask the agent to order a subset.
All tasks write an ordered batch to `artifacts/submission.json`.

## Prepare and run

Install the optional benchmark dependencies:

```bash
uv sync --extra scientific-design
```

Prepare the disclosure-scoped TFBind8 input and run the example manifest:

```bash
uv run python tasks/sci_modeling_bench/_shared/prepare_data.py \
  --task-id sci-modeling-bench-tfbind8
uv run python -m scienceflow.cli parallel \
  -m scienceflow/config/examples/tasks_sci_modeling_bench_tfbind8_example.yaml -j 1
```

Preparation writes the agent-visible input under
`cache/sci_modeling_bench/tfbind8/public`. Use `--output-dir` to choose another
location. Canonical manifests for all tasks are under
`scienceflow/config/runs/sci_modeling_bench/`.

## Evaluation profile

The opt-in `scienceflow/config/sci_modeling_bench.yaml` profile uses independent
worker query histories, stops after an accepted submission exhausts the query
budget, exposes the current runtime budget, and preserves structured experiment
state between research stages. It disables global merge and materializes the
best eligible Stage as the final artifact. General ScienceFlow defaults are
unchanged unless this profile is selected.

Prepared `public/` files are the agent input. Hidden outcomes, evaluator state,
and derived lookup caches remain system-side. Evaluator events and Stage
snapshots are stored in each run workspace for inspection.
