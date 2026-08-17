# Dogs Vs Cats Redux Kernels Edition — Lite Task Description

## Task description
In 2013, we hosted one of our favorite for-fun competitions: Dogs vs. Cats. Much has since changed in the machine learning landscape, particularly in deep learning and image analysis. Back then, a tensor flow was the diffusion of the creamer in a bored mathematician's cup of coffee. Now, even the cucumber farmers are …

## Task objective
- **Input:** Each test row / notebook identified by `image` (and any features in released `test` / `public` data).
- **Output:** For each image in the test set, you must submit a probability that image is a dog.

## Target metric (evaluation)
Submissions are scored on the log loss: *(formula in full description)* where - n is the number of images in the test set - \\( \hat{y}_i \\) is the predicted probability of the image being a dog - \\( y_i \\) is 1 if the image is a dog, 0 if cat - \\( log() \\) is the natural …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** id,label; one row per test key (see `sample_submission.csv`).
```
id,label
1,0.5
2,0.5
```

## Dataset and construction
- The train folder contains 25,000 images of dogs and cats.
- Each image in this folder has the label as part of the filename.
- The test folder contains 12,500 images, named according to a numeric id.
- For each image in the test set, you should predict a probability that the image is a dog (1 = dog, 0 = cat).
