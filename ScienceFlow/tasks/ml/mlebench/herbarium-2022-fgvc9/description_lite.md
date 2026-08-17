# Herbarium 2022 Fgvc9 — Lite Task Description

## Task description
*The Herbarium 2022: Flora of North America* is a part of a project of the New York Botanical Garden funded by the National Science Foundation to build tools to identify novel plant species around the world. The dataset strives to represent all known vascular plant taxa in North America, using images gathered from 60 …

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** *The Herbarium 2022: Flora of North America* is a part of a project of the New York Botanical Garden funded by the National Science Foundation to build tools to identify novel plant species around the world. The dataset …

## Target metric (evaluation)
Submissions are evaluated using the macro F1 score. The F1 score is given by where: In "macro" F1 a separate F1 score is calculated for each `species` value and then averaged. #Submission Format For each image `Id`, you should predict the corresponding image label (`category_id`) …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** `Id,Predicted`; one row per test key, matching `sample_submission.csv`.
```
Id,Predicted
0,42
```

## Dataset and construction
- ## Data Overview The training and test sets contain images of herbarium specimens from 15,501 species of vascular plants.
- Each image contains exactly one specimen.
- The text labels on the specimen images have been blurred to remove category information in the image.
- The data has been approximately split 80%/20% for training/test.
