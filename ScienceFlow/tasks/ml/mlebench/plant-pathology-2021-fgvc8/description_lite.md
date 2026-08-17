# Plant Pathology 2021 Fgvc8 — Lite Task Description

## Task description
The main objective of the competition is to develop machine learning-based models to accurately classify a given leaf image from the test dataset to a particular disease category, and to identify an individual disease from multiple disease symptoms on a single leaf image.

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** The main objective of the competition is to develop machine learning-based models to accurately classify a given leaf image from the test dataset to a particular disease category, and to identify an individual disease …

## Target metric (evaluation)
The evaluation metric for this competition is Mean F1-Score.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** `image,labels`; one row per test key, matching `sample_submission.csv`.
```
image,labels
ca6a50c5d2adb8ae.jpg,healthy
```

## Dataset and construction
- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the … *(see full `description.md`.)*
- `image`
- `labels`
