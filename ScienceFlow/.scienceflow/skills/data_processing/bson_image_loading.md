---
name: bson_image_loading
category: data_processing
tags:
  - bson
  - image-classification
  - data-loading
  - memory-efficient
  - large-dataset
task_types: ["*"]
phase: [analysis, drafting, debugging]
priority: 90
has_post_process: false
---

# BSON image data loading (memory-efficient)

## When to use

- Competition data ships as **MongoDB-style BSON** streams (`train.bson`, `test.bson`, …) with JPEG bytes under `imgs[*].picture`.
- Files are **multi-GB**; loading every record into Python lists will **OOM** (e.g. ~700k test products × several KB each ≈ **several GB** in RAM).
- You need **PyTorch `DataLoader`** without duplicating the whole file in memory.

## Anti-patterns (causes OOM or 1h+ stalls)

1. **`bson.decode_file_iter(f)` then `self.data.append(record)`** — stores **all** image bytes in RAM.
2. **`max_samples=None` on test** while still appending full records — still loads **every** image into a list.
3. **Two full scans** of `train.bson` — one for label vocabulary, one for `Dataset` — doubles I/O time.

## Pattern 1: single-pass index (`build_bson_index`)

BSON on disk is a **concatenation of documents**. Each document starts with a **little-endian int32 length** (bytes, **including** the 4 length bytes). Read that length, skip/read exactly `length` bytes, decode **one** document, record **file offset** + metadata only.

- Store **only** `(offset, _id, category_id or -1)` in `numpy` arrays or a compact table — **not** `picture` bytes.
- In the same pass, collect **unique `category_id`** for `category_to_idx` (train) or skip labels (test).
- Optionally **persist** the index (`np.savez` / `.npy`) so restarts do not rescan multi-GB files.

```python
import struct
from pathlib import Path

import bson
import numpy as np


def iter_bson_documents(path: str | Path):
    """Yield (file_offset, record) for each top-level document in a BSON file."""
    path = Path(path)
    with path.open("rb") as f:
        while True:
            offset = f.tell()
            len_buf = f.read(4)
            if len(len_buf) < 4:
                break
            (doc_len,) = struct.unpack("<i", len_buf)
            if doc_len < 5:  # malformed
                break
            rest = f.read(doc_len - 4)
            if len(rest) != doc_len - 4:
                break
            raw_doc = len_buf + rest
            yield offset, bson.decode(raw_doc)


def build_bson_index(
    bson_path: str | Path,
    is_train: bool,
    max_records: int | None = None,
    cache_npz: str | Path | None = None,
):
    """
    One scan: offsets + ids + optional category ids; build category set for train.
    Returns:
      offsets: np.ndarray int64
      ids: np.ndarray int64
      cat_ids: np.ndarray int64  # -1 if missing (test)
      category_to_idx, idx_to_category (train only; None for test)
    """
    bson_path = Path(bson_path)
    if cache_npz and Path(cache_npz).is_file():
        z = np.load(cache_npz, allow_pickle=True)
        offsets = z["offsets"]
        ids = z["ids"]
        cat_ids = z["cat_ids"]
        if is_train and "category_to_idx" in z.files:
            category_to_idx = z["category_to_idx"].item()
            idx_to_category = z["idx_to_category"].item()
            return offsets, ids, cat_ids, category_to_idx, idx_to_category
        return offsets, ids, cat_ids, None, None

    offsets_list, ids_list, cat_list = [], [], []
    categories = set()

    for i, (off, rec) in enumerate(iter_bson_documents(bson_path)):
        if max_records is not None and i >= max_records:
            break
        offsets_list.append(off)
        ids_list.append(int(rec["_id"]))
        if is_train:
            cid = int(rec["category_id"])
            cat_list.append(cid)
            categories.add(cid)
        else:
            cat_list.append(-1)

    offsets = np.asarray(offsets_list, dtype=np.int64)
    ids = np.asarray(ids_list, dtype=np.int64)
    cat_ids = np.asarray(cat_list, dtype=np.int64)

    category_to_idx = idx_to_category = None
    if is_train:
        sorted_cats = sorted(categories)
        category_to_idx = {c: j for j, c in enumerate(sorted_cats)}
        idx_to_category = {j: c for c, j in category_to_idx.items()}
        if cache_npz:
            np.savez(
                cache_npz,
                offsets=offsets,
                ids=ids,
                cat_ids=cat_ids,
                category_to_idx=category_to_idx,
                idx_to_category=idx_to_category,
            )
        return offsets, ids, cat_ids, category_to_idx, idx_to_category

    if cache_npz:
        np.savez(cache_npz, offsets=offsets, ids=ids, cat_ids=cat_ids)
    return offsets, ids, cat_ids, None, None
```

## Pattern 2: lazy `Dataset` (`BSONDatasetFast`)

- **`__len__`**: number of index rows (same as `offsets.shape[0]`).
- **`__getitem__(idx)`**: `seek(offsets[idx])`, read one document length + payload, `bson.decode`, decode JPEG from `record["imgs"][0]["picture"]` (or aggregate multi-image logic as required).
- **Do not** cache full `record` lists in the instance.

**`DataLoader` + `num_workers > 0`:** each worker must use its **own** file handle. Safe options:

1. **Open the file inside `__getitem__`** (simple; some overhead — often acceptable vs OOM).
2. **`threading.local()`** file object for reuse in the same worker process after lazy init.
3. **`worker_init_fn`** that opens one handle per worker (advanced).

```python
import io
import struct
from pathlib import Path

import bson
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def decode_one_at_offset(path: Path, offset: int) -> dict:
    with path.open("rb") as f:
        f.seek(offset)
        len_buf = f.read(4)
        if len(len_buf) < 4:
            raise EOFError("short read")
        (doc_len,) = struct.unpack("<i", len_buf)
        rest = f.read(doc_len - 4)
        return bson.decode(len_buf + rest)


class BSONImageDataset(Dataset):
    """Lazy BSON image dataset: O(1) RAM vs number of products (index only)."""

    def __init__(
        self,
        bson_path: str | Path,
        offsets: np.ndarray,
        category_to_idx: dict | None,
        cat_ids: np.ndarray,
        is_test: bool,
        transform=None,
    ):
        self.bson_path = Path(bson_path)
        self.offsets = offsets
        self.category_to_idx = category_to_idx
        self.cat_ids = cat_ids
        self.is_test = is_test
        self.transform = transform

    def __len__(self):
        return int(self.offsets.shape[0])

    def __getitem__(self, idx: int):
        off = int(self.offsets[idx])
        rec = decode_one_at_offset(self.bson_path, off)
        pic = rec["imgs"][0]["picture"]
        img = Image.open(io.BytesIO(pic)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img, int(rec["_id"])
        label = self.category_to_idx[int(rec["category_id"])]
        return img, label
```

## Notes

- **Test set size:** index can cover **all** test IDs without loading all images at once; only **one** decoded image lives in RAM per batch.
- **Subset training:** pass `max_records` when building the train index, or slice `offsets[:N]` — do **not** load full BSON into a list first.
- **Shuffle:** shuffle **index positions** (`np.random.permutation`) or use `Subset` — not raw file order requirement beyond your sampling strategy.
- **Validation BSON:** same index pattern; ensure label column exists like train.

## Self-check

- [ ] No list/dict that grows with **every** product’s raw `imgs` bytes in memory.
- [ ] At most **one** full-document decode per `__getitem__` (or batched read if you implement a custom batch loader).
- [ ] Train **category mapping** built in the **same** scan as offsets when possible.
- [ ] `num_workers > 0` tested — no shared seek across processes without per-worker handles.
