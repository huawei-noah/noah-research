# Petfinder Pawpularity Score — Lite Task Description

## Task description
Currently, PetFinder.my uses a basic Cuteness Meter to rank pet photos. It analyzes picture composition and other factors compared to the performance of thousands of pet profiles. While this basic tool is helpful, it's still in an experimental stage and the algorithm could be improved.

## Task objective
- **Input:** Each test row / notebook identified by `Id` (and any features in released `test` / `public` data).
- **Output:** For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`.

## Target metric (evaluation)
Root mean squared error (RMSE) between predicted and observed Pawpularity; lower is better. Report raw validation RMSE as `Final Validation Score` and set `lower_is_better=true`.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** Id, Pawpularity; one row per test key (see `sample_submission.csv`).
```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
```

## Dataset and construction
- The **Pawpularity Score** is derived from each pet profile's page view statistics at the listing pages, using an algorithm that normalizes the traffic data across different pages, platforms (web & mobile) and various … *(see full `description.md`.)*
- Duplicate clicks, crawler bot accesses and sponsored profiles are excluded from the analysis.
- We have included optional **Photo Metadata**, manually labeling each photo for key visual quality and composition parameters.
- These labels are **not used** for deriving our Pawpularity score, but it may be beneficial for better understanding the content and co-relating them to a photo's attractiveness. Our end goal is to deploy AI solutions … *(see full `description.md`.)*
