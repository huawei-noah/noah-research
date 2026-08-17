# Jigsaw Unintended Bias In Toxicity Classification — Lite Task Description

## Task description
Can you help detect toxic comments ― *and* minimize unintended model bias? That's your challenge in this competition.

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** Can you help detect toxic comments ― *and* minimize unintended model bias? That's your challenge in this competition.

## Target metric (evaluation)
### Competition Evaluation This competition will use a newly developed metric that combines several submetrics to balance overall performance with various aspects of unintended bias. First, we'll define each submetric. ### Overall AUC This is the ROC-AUC for the full evaluation …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Kernels-only**; see kernel rules.
- **Schema:** id,prediction; one row per test key (see `sample_submission.csv`).
```
id,prediction
7000000,0.0
7000001,0.0
```

## Dataset and construction
- See prepared `public/` and full `description.md` for file layout.
