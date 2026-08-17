# Google Research Identify Contrails Reduce Global Warming — Lite Task Description

## Task description
Contrails are clouds of ice crystals that form in aircraft engine exhaust. They can contribute to global warming by trapping heat in the atmosphere. Researchers have developed models to predict when contrails will form and how much warming they will cause. However, they need to validate these models with satellite …

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** Contrails are clouds of ice crystals that form in aircraft engine exhaust. They can contribute to global warming by trapping heat in the atmosphere. Researchers have developed models to predict when contrails will form …

## Target metric (evaluation)
This competition is evaluated on the **global** Dice coefficient. The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by: *(formula in full description)* where X is the …

**Metric fidelity requirement:** validation, threshold search, and reported `Final Validation Score` must match the official global Dice calculation:
- Aggregate all predicted and true mask pixels across every validation record, then compute one Dice score: `2 * intersection_sum / (predicted_positive_sum + true_positive_sum)`.
- Do **not** optimize or report mean per-image Dice, smoothed Dice, or empty-image-rewarded Dice as the primary validation score.
- Threshold selection and model selection must maximize this same global Dice. Per-image Dice may be logged only as a diagnostic.

## Brief background
Contrail avoidance is potentially one of the most scalable, cost-effective sustainability solutions available to airlines today. Contrails, short for ‘condensation trails’, are …

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Required columns:** `record_id,encoded_pixels`; one row per test record. The prepared sample may include `height,width`, but the grader only requires `record_id` and `encoded_pixels`.
- **Important:** `encoded_pixels` must be a run-length string, or `-` for an empty mask. Do not leave it as NaN/blank.
```
record_id,encoded_pixels
1000834164244036115,1 1 5 1
1002653297254493116,-
```

## Dataset and construction
- Contrails must contain at least 10 pixels
- At some time in their life, Contrails must be at least 3x longer than they are wide
- Contrails must either appear suddenly or enter from the sides of the image
- Contrails should be visible in at least two image
