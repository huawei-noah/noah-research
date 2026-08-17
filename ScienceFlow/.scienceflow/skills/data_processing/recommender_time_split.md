---
name: recommender_time_split
description: Build a time-window validation split for transaction recommendation datasets.
category: data_processing
tags: [data-preparation, recommender, transactions, time-split, validation]
task_types: ["*"]
phase: [data_preparation]
priority: 110
executable: false
context_vars: [input_data_dir, workspace_dir, seed, task_desc]
---

# Transaction recommender time split

Use this when the dataset contains a transaction log such as
`transactions_train.csv` with columns like `t_dat`, `customer_id`, and
`article_id`, plus item/customer metadata and a `sample_submission.csv`.

Do not use a random row split for this layout. Recommendation test rows usually
represent future customer behavior, so validation should hold out the latest
time window and train only on earlier transactions.

## Procedure

1. Read the raw input from the configured input data directory or the exposed
   `dataset/` view. Do not modify the raw input.
2. Parse `t_dat` as dates and find the maximum transaction date.
3. Use the last 7 days as validation by default:
   `validation_start = max_date - 7 days`. If the task description explicitly
   defines another public/private evaluation window, match that window instead.
4. Write earlier rows to `dataset/transactions_train.csv`.
5. Write held-out rows to `dataset/validation.csv`.
6. Group held-out rows by `customer_id` and write
   `dataset/validation_answers.csv` with columns:
   `customer_id,prediction`, where `prediction` is a space-separated list of
   article ids purchased by that customer in validation order.
7. Write `dataset/validation_sample_submission.csv` with the same customers and
   empty or popular-item baseline predictions, preserving the submission schema
   when practical.
8. Copy or symlink metadata files needed for training and prediction. Symlink
   large media folders such as `images/`.
9. Write `dataset/split_manifest.json` and `dataset/split_report.md`.

## Validation checks

- Normalize article ids to string tokens before writing. If ids look numeric or
  float-like, strip a trailing `.0` and left-pad to the width used by
  `articles.csv` when applicable.
- Ensure every article id in `validation_answers.csv` exists in item metadata
  when item metadata is available.
- Ensure train dates are strictly before the validation window and validation
  dates are inside the held-out window.
- Report train/validation row counts, customer counts, article counts, date
  ranges, and any cold-start customers or articles.

If the data layout is only superficially similar and the task metric is not a
future-interaction recommendation metric, fall back to `data_prep` instead of
forcing this split.
