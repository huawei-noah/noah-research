---
name: data_prep
category: data_processing
tags: [data-preparation, flat-dataset, validation]
task_types: ["*"]
phase: [data_preparation]
priority: 100
executable: true
max_retries: 4
context_vars: [input_data_dir, workspace_dir, seed, task_desc, prep_val_fraction, prep_retry_note_section]
---

# Dataset Preparation for Flat LNR Workspaces

{prep_retry_note_section}

Generate a Python script that prepares a single workspace data root at `dataset/`. Do **not** create `dataset/Deep/` or `dataset/Shallow/`.

## Task

{task_desc}

## Original dataset path (read-only)

```
{input_data_dir}
```

Bind this once in code, for example `input_base = Path(r"{input_data_dir}")`. Use it only for reads.

## Workspace output root

```
{workspace_dir}
```

The process cwd is the workspace root. Create all prepared data under `dataset/` relative to this cwd.

## Required Layout

- `dataset/train*` or equivalent training metadata/data needed by the task.
- `dataset/test*`, `dataset/sample_submission*`, `description.md`, and other non-training metadata when present.
- Optional local validation files at `dataset/validation*` when a holdout can be created safely.
- Optional hidden labels for self-eval as `dataset/mask_validation_label.*` when validation labels must not be visible in validation features.
- Large media/binary folders should be symlinked into `dataset/`, not copied.

## Validation Holdout

If the raw task has public labels, build the primary validation split from the public labeled pool. When both public `train` and public labeled `validation` are available, merge them first and then re-split into `dataset/train*` and `dataset/validation*`. For classification, segmentation, detection, or sparse-positive tasks, stratify by label presence and target density when possible. Never include held-out groups or rows in the remaining train files.

Do not blindly preserve an official/public validation split as the primary validation set when public labels are available for re-splitting. You may keep it only as clearly named auxiliary reference data if needed, but the main `dataset/validation*` should be the rebuilt holdout.

## Train/Validation Split Protocol

When creating validation from training data, create one reproducible train/validation split. Do not create `Deep/`, `Shallow/`, or CV fold directories.

1. Prefer group-aware splitting before row-level splitting. Infer group keys from stable ids, source filenames, patient/study ids, sequences, videos, or paired variants. All rows from the same group must stay entirely in either train or validation.
2. Prefer task-like validation. If test rows are generated from train groups differently than raw train rows, build validation rows to mimic the test row construction instead of blindly preserving the raw train row distribution.
3. For small, imbalanced, multilabel, or medical datasets, use a stratified group holdout when possible. Keep the validation fraction large enough to contain rare labels, and report any missing rare labels.
4. When groups differ by device, geography, source, site, class family, or acquisition route, prefer coverage-balanced validation over a pure random holdout. Represent rare or high-tail groups when possible, and report which groups remain uncovered.
5. Write the resulting labeled files as `dataset/train.csv` and `dataset/validation.csv` or task-equivalent names when the original schemas require them.
6. Write `dataset/split_manifest.json` with seed, split type, group key, target columns, train/validation row counts, label distribution, and any task-specific sampling rule.
7. Write `dataset/split_report.md` with a concise human-readable summary of train/validation distributions and known limitations.

For calibration, stacking, blending, thresholding, or post-processing, do not train and evaluate on the same validation predictions. If such steps are needed, create them only from the training side or clearly mark the validation score as unsafe for final model selection. Treat `same_validation_meta_fit`, proxy metrics, and unknown split protocols as invalid for final model selection.

Pattern-based guardrails:

- Source-variant image tasks: when several rows/files are variants of the same source item, group by source id or filename stem and keep all variants together. If the test set samples one candidate per source, build validation with one sampled candidate per source and match the target balance described by the task or sample layout.
- Segmentation, detection, and sparse-positive media tasks: if labels are masks, boxes, or per-record targets, split from the full public labeled pool and stratify by positive record presence plus target density/area buckets. Avoid inheriting an official validation set whose positive rate or target density may not match the task distribution.
- Medical, video, audio, time-series, or sequence tasks: group by patient, study, exam, series, video, recording, or sequence id rather than frame/slice/row id. Balance multilabel targets when possible, and report rare-label counts because a few positives can dominate the metric.
- Generic tabular classification/regression: use stratified splitting for classification targets when possible; for regression, stratify by target quantile bins if the sample size supports it.
- If labeled training groups overlap public sample/final-test identifiers, exclude those overlapping groups from train and validation tuning. Use analogous non-overlap training groups for validation coverage instead of tuning on final keys.
- If no reliable group key or target-like validation construction can be inferred, create a conservative stratified holdout, mark the split validity as `medium` or `low` in `split_manifest.json`, and explain the limitation in `split_report.md`.

## Output Rules

1. Write only inside `{workspace_dir}/dataset`.
2. Do not modify `{input_data_dir}`.
3. Do not create Deep/Shallow split directories.
4. Preserve schemas, IDs, row order, and relative media paths.
5. Use symlinks for large folders and bulky media.
6. Print a short summary including row counts and created files.
