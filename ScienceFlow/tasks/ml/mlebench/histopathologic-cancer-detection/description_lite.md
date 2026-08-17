# Histopathologic Cancer Detection — Lite Task Description

## Task description
In this competition, you must create an algorithm to identify metastatic cancer in small image patches taken from larger digital pathology scans. The data for this competition is a slightly modified version of the PatchCamelyon (PCam) benchmark dataset (the original PCam dataset contains duplicate images due to its …

## Task objective
- **Input:** Each test row / notebook identified by `id` (and any features in released `test` / `public` data).
- **Output:** For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue.

## Target metric (evaluation)
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** id,label; one row per test key (see `sample_submission.csv`).
```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
```

## Dataset and construction
- In this dataset, you are provided with a large number of small pathology images to classify.
- Files are named with an image `id`.
- The `train_labels.csv` file provides the ground truth for the images in the `train` folder.
- You are predicting the labels for the images in the `test` folder.
