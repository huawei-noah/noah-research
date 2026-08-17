# Learning Agency Lab Automated Essay Scoring 2 — Lite Task Description

## Task description
Essay writing is an important method to evaluate student learning and performance. It is also time-consuming for educators to grade by hand. Automated Writing Evaluation (AWE) systems can score essays to supplement an educator’s other efforts. AWEs also allow students to receive regular and timely feedback on their …

## Task objective
- **Input:** Each test row / notebook identified by `essay_id` (and any features in released `test` / `public` data).
- **Output:** For each `essay_id` in the test set, you must predict the corresponding `score` (described on the Data page).

## Target metric (evaluation)
Submissions are scored based on the quadratic weighted kappa, which measures the agreement between two outcomes. This metric typically varies from 0 (random agreement) to 1 (complete agreement). In the event that there is less agreement than expected by chance, the metric may go …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** essay_id,score; one row per test key (see `sample_submission.csv`).
```
essay_id,score
000d118,3
000fe60,3
```

## Dataset and construction
- **train.csv** - Essays and scores to be used as training data.
- `essay_id` - The unique ID of the essay
- `full_text` - The full essay response
- `score` - Holistic score of the essay on a 1-6 scale
