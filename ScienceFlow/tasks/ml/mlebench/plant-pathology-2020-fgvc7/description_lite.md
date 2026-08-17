# Plant Pathology 2020 Fgvc7 — Lite Task Description

## Task description
Objectives of 'Plant Pathology Challenge' are to train a model using images of training dataset to 1) Accurately classify a given image from testing dataset into different diseased category or a healthy leaf; 2) Accurately distinguish between many diseases, sometimes more than one on a single leaf; 3) Deal with rare …

## Task objective
- **Input:** Each test row / notebook identified by `image_id` (and any features in released `test` / `public` data).
- **Output:** For each image_id in the test set, you must predict a probability for each target variable.

## Target metric (evaluation)
Submissions are evaluated on mean column-wise ROC AUC. In other words, the score is the average of the individual AUCs of each predicted column.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** image_id,healthy,multiple_diseases,rust,scab; one row per test key (see `sample_submission.csv`).
```
image_id,healthy,multiple_diseases,rust,scab
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
```

## Dataset and construction
- `image_id`: the foreign key
- healthy: one of the target labels
- multiple_diseases: one of the target labels
- rust: one of the target labels
- scab: one of the target labels
