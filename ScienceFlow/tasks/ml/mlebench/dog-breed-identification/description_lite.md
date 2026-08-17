# Dog Breed Identification — Lite Task Description

## Task description
In this playground competition, you are provided a strictly canine subset of ImageNet in order to practice fine-grained image categorization. How well you can tell your Norfolk Terriers from your Norwich Terriers? With 120 breeds of dogs and a limited number training images per class, you might find the problem more …

## Task objective
- **Input:** Each test row / notebook identified by `image` (and any features in released `test` / `public` data).
- **Output:** For each image in the test set, you must predict a probability for each of the different breeds.

## Target metric (evaluation)
Submissions are evaluated on Multi Class Log Loss between the predicted probability and the observed target.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** id, … (remaining columns per sample_submission.csv); one row per test key (see `sample_submission.csv`).
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset and construction
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set
