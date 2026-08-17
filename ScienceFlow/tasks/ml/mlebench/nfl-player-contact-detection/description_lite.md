# Nfl Player Contact Detection — Lite Task Description

## Task description
Kaggle competition `nfl-player-contact-detection`.

## Task objective
- **Input:** Tracking, helmet detection, video metadata, and labels in prepared `public/`; labels only for train.
- **Output:** Predict contact events for every `contact_id` in `sample_submission.csv`.

## Target metric (evaluation)
Matthews Correlation Coefficient (MCC), higher is better.

## Validation and feature route hints
- Validate with GroupKFold by `game_play` or `game_key`/`play_id`; never use random row or `contact_id` splits as final selection evidence.
- Start with tracking, helmet boxes, and video metadata; if using raw videos, prefer pretrained frame/video embeddings over training video models from scratch.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** `contact_id,contact`; one row per required contact pair from `sample_submission.csv`.
- **Important:** `contact` must be a binary 0/1 label for scoring, not a continuous probability column. Tune a threshold before writing the final submission.
```
contact_id,contact
58168_003392_0_38590_43854,0
58168_003392_0_38590_41257,1
```

## Dataset and construction
- See prepared `public/` and full `description.md` for file layout.
