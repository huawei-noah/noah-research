# Whale Categorization Playground — Lite Task Description

## Task description
To aid whale conservation efforts, scientists use photo surveillance systems to monitor ocean activity. They use the shape of whales’ tails and unique markings found in footage to identify what species of whale they’re analyzing and meticulously log whale pod dynamics and movements. For the past 40 years, most of this …

## Task objective
- **Input:** Each test row / notebook identified by `Image` (and any features in released `test` / `public` data).
- **Output:** For each `Image` in the test set, you may predict up to 5 labels for the whale `Id`.

## Target metric (evaluation)
Submissions are evaluated according to the Mean Average Precision @ 5 (MAP@5): *(formula in full description)* where `U` is the number of images, `P(k)` is the precision at cutoff `k`, and `n` is the number predictions per image.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** Image,Id; one row per test key (see `sample_submission.csv`).
```
Image,Id
00029b3a.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
0003c693.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
```

## Dataset and construction
- **train.zip** - a folder containing the training images
- **train.csv** - maps the training `Image` to the appropriate whale `Id`. Whales that are not predicted to have a label identified in the training data should be labeled as `new_whale`.
- **test.zip** - a folder containing the test images to predict the whale `Id`
- **sample_submission.csv** - a sample submission file in the correct format
