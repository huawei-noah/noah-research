# Tensorflow2 Question Answering — Lite Task Description

## Task description
In this competition, your goal is to predict short and long answer responses to real questions about Wikipedia articles. The dataset is provided by Google's Natural Questions, but contains its own unique private test set. A visualization of examples shows long and---where available---short answers. In addition to …

## Task objective
- **Input:** Each test example is identified by `example_id` in `simplified-nq-test.jsonl`; `sample_submission.csv` contains two rows per base example, suffixed as `<example_id>_long` and `<example_id>_short`.
- **Output:** For each submission row, predict a `PredictionString`: a `start_token:end_token` span, `YES`/`NO` for short answers only, or a blank string when no answer should be predicted.

## Target metric (evaluation)
Submissions are evaluated using micro F1 between the predicted and expected answers; higher is better. Predicted long and short answers must match exactly the token indices of one of the ground truth labels (or match YES/NO if the question has a yes/no short answer). There may be up to five …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Kernels-only**; see kernel rules.
- **Schema:** `example_id,PredictionString`; one `_long` row and one `_short` row for every base test `example_id` (see `sample_submission.csv`). Leave `PredictionString` blank for no answer.
```
example_id,PredictionString
-7853356005143141653_long,6:18
-7853356005143141653_short,YES
-545833482873225036_long,105:200
-545833482873225036_short,
```

## Dataset and construction
- A long answer would be a longer section of text that answers the question - several sentences or a paragraph.
- A short answer might be a sentence or phrase, or even in some cases a YES/NO. The short answers are always contained within / a subset of one of the plausible long answers.
- A given article can (and very often will) allow for both long *and* short answers, depending on the question.
- **simplified-nq-train.jsonl** - the training data, in newline-delimited JSON format.
