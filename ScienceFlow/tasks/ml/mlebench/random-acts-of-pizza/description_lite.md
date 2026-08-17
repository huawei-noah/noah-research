# Random Acts Of Pizza — Lite Task Description

## Task description
For each request in the test set, you should predict a real-valued probability that it resulted in a pizza. The file should contain a header and have the following format:

## Task objective
- **Input:** Each test row / notebook identified by `request` (and any features in released `test` / `public` data).
- **Output:** For each request in the test set, you should predict a real-valued probability that it resulted in a pizza.

## Target metric (evaluation)
Submissions are evaluated on area under the ROC curve between the predicted probability that a request will get pizza and the observed outcomes.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** request_id,requester_received_pizza; one row per test key (see `sample_submission.csv`).
```
request_id,requester_received_pizza
t3_i8iy4,0
t3_1mfqi0,0
```

## Dataset and construction
- See, fork, and run a random forest benchmark model through Kaggle Scripts This dataset includes 5671 requests collected from the Reddit community Random Acts of Pizza between December 8, 2010 and …
- All requests ask for the same thing: a free pizza.
- The outcome of each request -- whether its author received a pizza or not -- is known.
- Meta-data includes information such as: time of the request, activity of the requester, community-age of the requester, etc.
