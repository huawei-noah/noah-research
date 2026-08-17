# Freesound Audio Tagging 2019 — Lite Task Description

## Task description
To win this competition, Kagglers will develop an algorithm to tag audio data automatically using a diverse vocabulary of 80 categories.

## Task objective
- **Input:** Each test row / notebook identified by `fname` (and any features in released `test` / `public` data).
- **Output:** For each `fname` in the test set, you must predict the probability of each label.

## Target metric (evaluation)
The primary competition metric will be label-weighted label-ranking average precision (*lwlrap*, pronounced "Lol wrap").

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Kernels-only**; see kernel rules.
- **Schema:** fname, … (remaining columns per sample_submission.csv); one row per test key (see `sample_submission.csv`).
```
fname,Accelerating_and_revving_and_vroom,...Zipper_(clothing)
000ccb97.wav,0.1,....,0.3
0012633b.wav,0.0,...,0.8
```

## Dataset and construction
- Freesound Dataset (FSD): a dataset being collected at the MTG-UPF based on Freesound content organized with the AudioSet Ontology
- The soundtracks of a pool of Flickr videos taken from the … *(see full `description.md`.)*
