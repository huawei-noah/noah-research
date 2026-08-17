# Spaceship Titanic — Lite Task Description

## Task description
Binary classification: from passenger records recovered from the damaged ship computer, predict whether each passenger was **transported** to an alternate dimension by the anomaly.

## Task objective
- **Input:** Each test passenger `PassengerId` and released tabular features (see `test.csv`).
- **Output:** Boolean `Transported` per `PassengerId` (see `sample_submission.csv`).

## Target metric (evaluation)
**Classification accuracy** — fraction of test rows whose `Transported` prediction matches the hidden label.

## Brief background
Getting-started tabular playground; ignore narrative fluff for modeling.

## Submission
- **File:** `submission.csv`.
- **Schema:** `PassengerId,Transported`; one row per test passenger.
```
PassengerId,Transported
0013_01,False
0018_01,False
```

## Dataset and construction
- **train.csv:** ~8700 passengers with features + `Transported` label.
- **test.csv:** passengers to predict (no `Transported`).
- **sample_submission.csv:** required row IDs and column template.
