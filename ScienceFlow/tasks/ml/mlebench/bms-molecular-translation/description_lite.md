# Bms Molecular Translation — Lite Task Description

## Task description
In a technology-forward world, sometimes the best and easiest tools are still pen and paper. Organic chemists frequently draw out molecular work with the Skeletal formula, a structural notation used for centuries. Recent publications are also annotated with machine-readable chemical descriptions (InChI), but there are …

## Task objective
- **Input:** Each test row / notebook identified by `image_id` (and any features in released `test` / `public` data).
- **Output:** For each `image_id` in the test set, you must predict the InChi string of the molecule in the corresponding image.

## Target metric (evaluation)
Submissions are evaluated on the mean Levenshtein distance between the InChi strings you submit and the ground truth InChi values.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** image_id,InChI; one row per test key (see `sample_submission.csv`).
```
image_id,InChI
00000d2a601c,InChI=1S/H2O/h1H2
00001f7fc849,InChI=1S/H2O/h1H2
```

## Dataset and construction
- **train/** - the training images, arranged in a 3-level folder structure by `image_id`
- **test/** - the test images, arranged in the same folder structure as `train/`
- **train_labels.csv** - ground truth InChi labels for the training images
- **sample_submission.csv** - a sample submission file in the correct format
