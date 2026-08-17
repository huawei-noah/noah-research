# Lmsys Chatbot Arena — Lite Task Description

## Task description
We utilized a large dataset collected from Chatbot Arena, where users chat with two anonymous LLMs and choose the answer they prefer. Your task in this competition is to predict which response a user will prefer in these head-to-head battles.

## Task objective
- **Input:** Each test row / notebook identified by `id` (and any features in released `test` / `public` data).
- **Output:** For each id in the test set, you must predict the probability for each target class.

## Target metric (evaluation)
Submissions are evaluated on the log loss between the predicted probabilities and the ground truth values (with "eps=auto").

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** id, … (remaining columns per sample_submission.csv); one row per test key (see `sample_submission.csv`).
```
id,winner_model_a,winner_model_b,winner_tie
 136060,0.33,0,33,0.33
 211333,0.33,0,33,0.33
```

## Dataset and construction
- `id` - A unique identifier for the row.
- `model_[a/b]` - The identity of model_[a/b]. Included in train.csv but not test.csv.
- `prompt` - The prompt that was given as an input (to both models).
- `response_[a/b]` - The response from model_[a/b] to the given prompt.
