---
name: image_task_guide
category: data_processing
tags: [image, resolution, sampling, augmentation, preprocessing]
task_types: ["*"]
phase: [analysis, drafting]
priority: 87
has_post_process: false
---

# Image tasks: resolution, sampling, and preprocessing

## Resolution by task type

| Task family | Guidance |
|-------------|----------|
| Steganalysis, IQA, fine texture, many medical | Do **not** downsample below **~50%** of native size if possible; **minimum 224×224** for CNNs unless description says otherwise. |
| Generic classification (object salient) | **128–224** short side is often acceptable. |
| Detection / segmentation | Prefer long edge **≥ 640** (or task default). |

**Rule of thumb:** if originals are **N×N**, avoid shrinking below `max(224, N // 2)` for subtle-signal tasks.

## Training data sampling (GPU)

- Use **stratified** sampling when labels are imbalanced.
- Prefer **full training set** when runtime allows; GPU should handle batching.
- If data is **huge (>100k samples)** and time is tight, subset but keep **≥ 50%** of rows unless you measured throughput and cannot finish otherwise.
- **Time estimate:** time one epoch (or N batches) on a small slice, extrapolate to full data before fixing `train_subset`.

## Preprocessing

- Standard ImageNet normalization is a good default for pretrained CNNs.
- **Train/val/test** must share the same **eval** preprocessing for inference; augmentation only on train.

## Anti-patterns

- Resizing to **64×64** or below for steganalysis / subtle artifacts — destroys the signal.
- Setting `train_subset` without checking CSV row counts or wall-clock budget.
- Different resize/normalize between validation and test submission paths.

## Self-check

- [ ] Chosen resolution matches task (texture vs coarse object).
- [ ] Sampling preserves class ratios where required.
- [ ] Val/test preprocessing matches exported inference path.
