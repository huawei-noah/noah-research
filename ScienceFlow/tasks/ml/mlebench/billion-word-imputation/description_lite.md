# Billion Word Imputation — Lite Task Description

## Task description
This competition uses the billion-word benchmark corpus provided by Chelba et al. for language modeling. Rather than ask participants to create a classic language model and evaluate sentence probabilities -- a task which is difficult to faithfully score in Kaggle's supervised ML setting -- we have introduced a …

## Task objective
- **Input:** In the prepared workspace, `dataset/train.txt` contains complete sentences, `dataset/validation.txt` contains complete held-out validation sentences, and `dataset/test_v2.txt` contains rows with one missing word to impute.
- **Output:** For each test `id`, submit the completed sentence.

## Target metric (evaluation)
Official metric: mean character-level Levenshtein distance between the submitted completed sentences and the original sentences; lower is better.

Report `Final Validation Score` as the raw mean Levenshtein distance in the same units and direction as the leaderboard metric, e.g. total edit distance divided by validation row count. Do not use a normalized accuracy/proxy such as `1 - distance / characters` as the primary score; if computed, label it as auxiliary only.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** id,sentence; one row per test key. Preserve the exact IDs from `test_v2.txt` (they are zero-based in this prepared dataset) rather than regenerating them. In this prepared split, `test_v2.txt` has the same header/order as the submission template.
```
"id","sentence"
0,"<completed sentence for test id 0>"
1,"<completed sentence for test id 1>"
```

## Dataset and construction
- **dataset/train.txt** - training sentences, complete English sentences.
- **dataset/validation.txt** - held-out complete validation sentences; skip sentences with fewer than three whitespace-delimited tokens, remove exactly one non-first/non-last token from each retained sentence, and score with raw mean character-level Levenshtein distance.
- **dataset/test_v2.txt** - the test/submission-template rows where one word has been removed.
