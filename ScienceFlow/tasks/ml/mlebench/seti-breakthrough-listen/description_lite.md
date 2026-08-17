# Seti Breakthrough Listen — Lite Task Description

## Task description
Kaggle competition `seti-breakthrough-listen`.

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** Predict targets for `seti-breakthrough-listen` test set per official rules.

## Target metric (evaluation)
Area under the ROC curve (ROC AUC) between predicted probabilities and observed targets; higher is better. Report raw validation ROC AUC as `Final Validation Score` and set `lower_is_better=false`.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** `id,target`; one row per test key, matching `sample_submission.csv`.
```
id,target
0cee567456cd304,0.5
```

## Dataset and construction
- **train/** - a training set of cadence snippet files stored in `numpy` `float16` format (v1.20.1), one file per cadence snippet `id`, with corresponding labels found in the `train_labels.csv` file. Each file has … *(see full `description.md`.)*
- **test/** - the test set cadence snippet files; you must predict whether or not the cadence contains a "needle", which is the `target` for this competition
- **sample_submission.csv** - a sample submission file in the correct format
- **train_labels** - targets corresponding (by `id`) to the cadence snippet files found in the `train/` folder
