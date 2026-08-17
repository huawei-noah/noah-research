# Denoising Dirty Documents — Lite Task Description

## Task description
Optical Character Recognition (OCR) is the process of getting type or handwritten documents into a digitized format. If you've read a classic novel on a digital reading device or had your doctor pull up old healthcare records via the hospital computer system, you've probably benefited from OCR. OCR makes previously …

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** Optical Character Recognition (OCR) is the process of getting type or handwritten documents into a digitized format. If you've read a classic novel on a digital reading device or had your doctor pull up old healthcare …

## Target metric (evaluation)
Submissions are evaluated on the root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** id,value; one row per test key (see `sample_submission.csv`).
- **Value scale:** `value` must be a normalized grayscale intensity in `[0, 1]`, matching the cleaned target pixels used by the grader. Do not submit `0..255` pixel values.
```
id,value
1_1_1,1.0
1_2_1,0.998
```

## Dataset and construction
- You are provided two sets of images, train and test.
- These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts.
- The training set includes the test without the noise (train_cleaned).
- You must create an algorithm to clean the images in the test set.
