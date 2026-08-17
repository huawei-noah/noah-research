# Rsna Breast Cancer Detection — Lite Task Description

## Task description
The goal of this competition is to identify breast cancer. You'll train your model with screening mammograms obtained from regular screening.

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** The goal of this competition is to identify breast cancer. You'll train your model with screening mammograms obtained from regular screening.

## Target metric (evaluation)
Submissions are evaluated using the probabilistic F1 score (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. You can find a Python implementation here. With pX as the probabilistic version of X: *(formula in full …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** `prediction_id,cancer`; one row per test key, matching `sample_submission.csv`.
```
prediction_id,cancer
10116_L,0.02057604809879753
```

## Dataset and construction
- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.


## Local split caveats
- Image files are DICOM: `train_images/{patient_id}/{image_id}.dcm` and `test_images/{patient_id}/{image_id}.dcm`. The prepared Deep/Shallow splits symlink these image directories from `public/`.
- The provided validation split is image-level, not a strict patient-level holdout. Deep validation patients all also appear in train; Shallow also has substantial patient overlap. Treat local validation as a fast route signal, not final generalization proof.
- Build any serious CV by grouping on `patient_id` and checking performance by site, laterality, view, age/machine, and cancer prevalence. Avoid patient-ID memorization or metadata-only shortcuts.
- `test.csv` has multiple image rows per `prediction_id` (`patient_id_laterality`), while `sample_submission.csv` has one row per `prediction_id`. Aggregate image-level probabilities by `prediction_id` before writing `submission.csv`.
- Cancer prevalence is about 2%; thresholding and pF1 calibration are unstable. Tune thresholds only on grouped validation, and report both image-level and prediction_id-level behavior.

## Implementation note: DICOM pre-caching (MANDATORY)

Training images are DICOM files. **Do NOT call `pydicom.dcmread` + `pixel_array` inside `Dataset.__getitem__`** — real-time DICOM decode is extremely CPU-heavy and will starve the GPU even with high `num_workers`.

Instead, add a **one-time pre-caching step before the DataLoader** that converts all DICOM files to PNG:

```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
import pydicom, numpy as np
from PIL import Image

def _convert_one(args):
    dcm_path, out_path, img_size = args
    out = Path(out_path)
    if out.exists():
        return
    arr = pydicom.dcmread(dcm_path).pixel_array.astype(np.float32)
    arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255).astype(np.uint8)
    Image.fromarray(arr).resize((img_size, img_size)).save(out)

def cache_dicom_to_png(dicom_paths, cache_dir: Path, img_size: int = 512):
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, str(cache_dir / (Path(p).stem + ".png")), img_size) for p in dicom_paths]
    workers = min(len(jobs), os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_convert_one, jobs))

# Call this ONCE before building the Dataset/DataLoader:
cache_dir = Path("cache_png")
cache_dicom_to_png(all_dicom_paths, cache_dir)

# Then __getitem__ just does:
#   img = Image.open(cache_dir / (stem + ".png"))  ← fast
```
