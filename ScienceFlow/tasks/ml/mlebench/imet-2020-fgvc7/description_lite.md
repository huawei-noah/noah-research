# Imet 2020 Fgvc7 — Lite Task Description

## Task description
The Metropolitan Museum of Art in New York, also known as The Met, has a diverse collection of over 1.5M objects of which over 200K have been digitized with imagery. Can you help find the significant attributes to identify a specific work of art? Help advance this research in this notebook competition.

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** The Metropolitan Museum of Art in New York, also known as The Met, has a diverse collection of over 1.5M objects of which over 200K have been digitized with imagery. Can you help find the significant attributes to …

## Target metric (evaluation)
Submissions will be evaluated based on their micro averaged F1 score.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Kernels-only**; see kernel rules.
- **Schema:** id,attribute_ids; one row per test key (see `sample_submission.csv`).
```
id,attribute_ids
00011f01965f141f5d1eea6592fa9862,0 1 2
00014abc91ed3e4bf1663fde8136fe80,0 1 2
```

## Dataset and construction
- **train.csv** gives the `attribute_ids` for the train images in **/train**
- **/test** contains the test images. You must predict the `attribute_ids` for these images.
- **sample_submission.csv** contains a submission in the correct format
- **labels.csv** provides descriptions of the attributes
