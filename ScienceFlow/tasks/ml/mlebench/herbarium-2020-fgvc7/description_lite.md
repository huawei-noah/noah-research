# Herbarium 2020 Fgvc7 — Lite Task Description

## Task description
The Herbarium 2020 FGVC7 Challenge is to identify vascular plant species from a large, long-tailed collection herbarium specimens provided by the New York Botanical Garden (NYBG).

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** The Herbarium 2020 FGVC7 Challenge is to identify vascular plant species from a large, long-tailed collection herbarium specimens provided by the New York Botanical Garden (NYBG).

## Target metric (evaluation)
Submissions are evaluated using the macro F1 score. F1 is calculated as follows: *(formula in full description)* where: *(formula in full description)* In "macro" F1 a separate F1 score is calculated for each `species` value and then averaged.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** `Id,Predicted`; one row per test key, matching `sample_submission.csv`.
```
Id,Predicted
0,0
```

## Dataset and construction
- ### Data Overview The training and test set contain images of herbarium specimens, from over 32,000 species of vascular plants.
- Each image contains exactly one specimen.
- The text and barcode labels on the specimen images have been blurred to remove category information in the image.
- The data has been approximately split 80%/20% for training/test.
