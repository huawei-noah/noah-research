# Kuzushiji Recognition — Lite Task Description

## Task description
Imagine the history contained in a thousand years of books. What stories are in those books? What knowledge can we learn from the world before our time? What was the weather like 500 years ago? What happened when Mt. Fuji erupted? How can one fold 100 cranes using only one piece of paper? The answers to these questions …

## Task objective
- **Input:** Each test row / notebook identified by `image` (and any features in released `test` / `public` data).
- **Output:** For each image in the test set, you must locate and identify all of the kuzushiji characters.

## Target metric (evaluation)
Submissions are evaluated on a modified version of the F1 Score. To score a true positive, you must provide center point coordinates that are within the ground truth bounding box and a matching label. The ground truth bounding boxes are defined in the format `{label X Y Width …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** image_id,labels; one row per test key (see `sample_submission.csv`).
```
image_id,labels
image_id,{label X Y} {...}
```

## Dataset and construction
- Some images only contain illustrations, in which case no labels should be submitted for the page.
- Kuzushiji text is written such that annotations are placed between the columns of the main text, usually in a slightly smaller font. See the characters with no bounding boxes in this image for examples. The annotation … *(see full `description.md`.)*
- You can occasionally see through especially thin paper and read characters from the opposite side of the page. Those characters should also be ignored.
- **train.csv** - the training set labels and bounding boxes.
