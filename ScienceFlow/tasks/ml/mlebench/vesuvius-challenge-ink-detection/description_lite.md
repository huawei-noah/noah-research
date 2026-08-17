# Vesuvius Challenge Ink Detection — Lite Task Description

## Task description
The ink used in the Herculaneum scrolls does not show up readily in X-ray scans. But we have found that machine learning models can detect it. Luckily, we have ground truth data. Since the discovery of the Herculaneum Papyri almost 300 years ago, people have tried opening them, often with disastrous results. Many …

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** For each test fragment, predict a binary ink mask encoded as run-length pairs in `submission.csv`.

## Target metric (evaluation)
We evaluate how well your output image matches our reference image using a modified version of the Sørensen--Dice coefficient, where instead of using the F1 score, we are using the F0.5 score. Higher is better. The F0.5 score weights precision higher than recall.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** `Id,Predicted`; one row per test key, matching `sample_submission.csv`.
```
Id,Predicted
a,1 1 5 1
```

## Dataset and construction
- **[train/test]/[fragment_id]/surface_volume/[image_id].tif** slices from the 3d x-ray surface volume. Each file contains a greyscale slice in the z-direction. Each fragment contains 65 slices. Combined this image stack … *(see full `description.md`.)*
- **[train/test]/[fragment_id]/mask.png** --- a binary mask of which pixels contain data.
- **train/[fragment_id]/inklabels.png** --- a binary mask of the ink vs no-ink labels.
- **train/[fragment_id]/inklabels_rle.csv** --- a run-length-encoded version of the labels, generated using this script. This is the same format as you should make your submission in.
