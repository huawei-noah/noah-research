# Cdiscount Image Classification Challenge — Lite Task Description

## Task description
Classify each Cdiscount product from one or more product images. This is an
extreme multiclass problem with 5,270 leaf categories and JPEG images embedded
inside product-level BSON records.

## Task objective
- **Input:** Each test product identified by `_id`, with one or more product images.
- **Output:** For each *_id* in the test set, you must predict a *category_id*.

## Target metric (evaluation)
### Goal The goal of this competition is to predict the category of a product based on its image(s). Note that a product can have one or several images associated. For every product `_id` in the test set, you should predict the correct `category_id`.

### Metric
Categorization accuracy: the percentage of test products assigned the correct `category_id`. **Higher is better.**

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** _id,category_id; one row per test key (see `sample_submission.csv`).
```
_id,category_id
2,1000000055
5,1000016018
```

## BSON loading (MANDATORY — avoid OOM)

Do **not** use `bson.decode_all(f.read())` on multi-GB files. Stream records, e.g.:

Use **PyMongo’s** `bson` module (`pip` package `pymongo`). There is **no** `bson.loads` — decode one document with **`bson.decode(head + body)`** (full BSON document bytes).

```python
import struct
from bson import decode

def iter_bson_products(path):
    with open(path, "rb") as f:
        while True:
            head = f.read(4)
            if not head:
                break
            size = struct.unpack("<i", head)[0]
            body = f.read(size - 4)
            if len(body) < size - 4:
                break
            doc = decode(head + body)
            yield doc
```

Use a small shard or `train_example.bson` first; scale with multiprocessing only after the pipeline is stable.

## Efficiency and scaling guidance

- Establish a comparable validation score on a representative, category-covered subset before running full validation or test inference.
- Keep early train and validation scales balanced. Do not combine a tiny training subset with the full validation set.
- Avoid full multinomial sklearn `LogisticRegression` on high-dimensional features with thousands of classes. For large feature sets, use a mini-batch GPU-trained linear head or small MLP.
- Stream BSON records and save features in resumable chunks instead of retaining the full image or feature collection in CPU/GPU memory.
- Early experiments should produce a validation metric within roughly 15-20 minutes. If they do not, reduce scope or change route before scaling.

## Suggested model order

- **Fast baseline:** pretrained EfficientNet-B0/B2 on a category-covered subset, with a GPU-trained classification head.
- **Primary route:** pretrained `tf_efficientnet_b4`, then fine-tune at least the final backbone blocks after the data pipeline is stable. Average logits or features across all images of each product.
- **Complementary route:** pretrained ConvNeXt-Tiny to provide architectural diversity once the same streaming pipeline works.
- **Feature fallback:** DINOv2 ViT-S/14 with a GPU linear/MLP head can validate data and sampling quickly, but a frozen feature model should not be treated as the final medal route without evidence.

## Dataset and construction

The prepared `dataset_split_v2` flat view is authoritative for this run. It
contains 5,726,646 labeled training products, 636,260 labeled validation
products, and 706,990 unlabeled test products. `sample_submission.csv` contains
the exact required test IDs in submission order.

The provided train/validation split is a deterministic category-stratified
90/10 product-level holdout with seed 42. Train and validation IDs are disjoint,
and all 5,270 categories occur in both splits. Use this split for comparable
local validation; do not merge validation records back into training or create
a different primary holdout.

- **train.bson** - Labeled training products with `_id`, `category_id`, and one or more images.
- **validation.bson** - Labeled holdout products with the same schema as `train.bson`.
- **train_example.bson** - The first 100 records of the prepared `train.bson`, for fast pipeline checks.
- **test.bson** - Unlabeled products; predict one category for every `_id` in `sample_submission.csv`.
- **category_names.csv** - The French level1/level2/level3 hierarchy for all 5,270 leaf `category_id` values.
- **split_manifest.json / split_report.md** - Reproducibility metadata, counts, hashes, and known split limitations.
