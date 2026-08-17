# Uw Madison Gi Tract Image Segmentation — Lite Task Description

## Task description
In 2019, an estimated 5 million people were diagnosed with a cancer of the gastro-intestinal tract worldwide. Of these patients, about half are eligible for radiation therapy, usually delivered over 10-15 minutes a day for 1-6 weeks. Radiation oncologists try to deliver high doses of radiation using X-ray beams pointed …

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** For each `(id, class)` test row, predict an RLE segmentation mask for `large_bowel`, `small_bowel`, and `stomach`; use an empty string when no mask is predicted.

## Target metric (evaluation)
This competition is evaluated on the mean Dice coefficient and 3D Hausdorff distance. The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by: *(formula in full description)* …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** id,class,predicted; one row per test key (see `sample_submission.csv`).
```
id,class,predicted
case123_day20_slice_0001,large_bowel,1 1 5 2
case123_day20_slice_0001,small_bowel,
case123_day20_slice_0001,stomach,
```

## Dataset and construction
- `train.csv` - IDs, organ class, and RLE masks for all training objects.
- `test.csv` - IDs and organ class rows for the held-out test slices.
- `sample_submission.csv` - a sample submission file in the correct format.
- `train/` and `test/` - folders of `case*/case*_day*/scans/*.png` slice images.
- Image filenames encode slice number, height, width, and pixel spacing: `slice_####_H_W_px_py.png`.
- `id` - unique slice identifier in the form `case{case}_day{day}_slice_{slice}`.
- Local split files under `dataset_split/Deep` keep train/validation/test ids disjoint; train and test may share patient cases but not exact case-day slices.
