# Herbarium 2021 Fgvc8 — Lite Task Description

## Task description
*The Herbarium 2021: Half-Earth Challenge* is to identify vascular plant specimens provided by the New York Botanical Garden (NY), Bishop Museum (BPBM), Naturalis Biodiversity Center (NL), Queensland Herbarium (BRI), and Auckland War Memorial Museum (AK).

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** *The Herbarium 2021: Half-Earth Challenge* is to identify vascular plant specimens provided by the New York Botanical Garden (NY), Bishop Museum (BPBM), Naturalis Biodiversity Center (NL), Queensland Herbarium (BRI), and …

## Target metric (evaluation)
Submissions are evaluated using the macro F1 score. The F1 score is given by *(formula in full description)* where: *(formula in full description)* In "macro" F1 a separate F1 score is calculated for each `species` value and then averaged.

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
- ### Data Overview The training and test set contain images of herbarium specimens from nearly 65,000 species of vascular plants.
- Each image contains exactly one specimen.
- The text labels on the specimen images have been blurred to remove category information in the image.
- The data has been approximately split 80%/20% for training/test.
