# Hubmap Kidney Segmentation — Lite Task Description

## Task description
Your challenge is to detect functional tissue units (FTUs) across different tissue preparation pipelines. An FTU is defined as a "three-dimensional block of cells centered around a capillary, such that each cell in this block is within diffusion distance from any other cell in the same block" (de Bono, 2013). The goal …

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** For each test image ID, predict an RLE-encoded segmentation mask.

## Target metric (evaluation)
Mean Dice coefficient across test images; higher is better. Report the raw mean Dice as `Final Validation Score` and set `lower_is_better=false`.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Kernels-only**; see kernel rules.
- **Schema:** id,predicted; one row per test key (see `sample_submission.csv`).
```
id,predicted
8242609fa,1 1
0486052bb,1 1
095bf7a1f,1 1
```

## Dataset and construction
- A `type` (`Feature`) and object type `id` (`PathAnnotationObject`). Note that these fields are the same between all files and do not offer signal.
- A `geometry` containing a `Polygon` with `coordinates` for the feature's enclosing volume
- Additional `properties`, including the name and color of the feature in the image.
- The `IsLocked` field is the same across file types (locked for glomerulus, unlocked for anatomical structure) and is not signal-bearing.
