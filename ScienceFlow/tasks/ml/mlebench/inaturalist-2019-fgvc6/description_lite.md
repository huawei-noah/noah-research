# Inaturalist 2019 Fgvc6 — Lite Task Description

## Task description
As part of the FGVC6 workshop at CVPR 2019 we are conducting the iNat Challenge 2019 large scale species classification competition, sponsored by Microsoft. It is estimated that the natural world contains several million species of plants and animals. Without expert knowledge, many of these species are extremely …

## Task objective
- **Input:** Each test row / notebook identified by `image` (and any features in released `test` / `public` data).
- **Output:** For each image in the test set, you must predict 1 category label.

## Target metric (evaluation)
We use top-1 classification error as the metric for this competition. For each image, an algorithm will produce 1 label. If the predicted label matches the ground truth label then the error for that image is 0, otherwise it is 1. The final score is the error averaged across all …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** id,predicted; one row per test key (see `sample_submission.csv`).
```
id,predicted
268243,71 108 339 341 560
268244,333 729 838 418 785
```

## Dataset and construction
- **train_val2019.tar.gz** - Contains the training and validation images in a directory structure following {iconic category name}/{category name}/{image id}.jpg .
- **train2019.json** - Contains the training annotations.
- **val2019.json** - Contains the validation annotations.
- **test2019.tar.gz** - Contains a single directory of test images.
