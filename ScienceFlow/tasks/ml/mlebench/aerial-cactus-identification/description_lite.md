# Aerial Cactus Identification — Lite Task Description

## Task description
To assess the impact of climate change on Earth's flora and fauna, it is vital to quantify how human activities such as logging, mining, and agriculture are impacting our protected natural areas. Researchers in Mexico have created the VIGIA project, which aims to build a system for autonomous surveillance of protected …

## Task objective
- **Input:** Each test row / notebook identified by `ID` (and any features in released `test` / `public` data).
- **Output:** For each ID in the test set, you must predict a probability for the `has_cactus` variable.

## Target metric (evaluation)
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Kernels-only**; see kernel rules.
- **Schema:** id,has_cactus; one row per test key (see `sample_submission.csv`).
```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
```

## Dataset and construction
- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format
