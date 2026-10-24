# DeepLPF: code upgrade, a runnable entry point, a module split, and packaging

25 commits against `DeepLPF/`. Nothing outside that directory is touched;
`LICENSE` is not modified. I am one of the authors of this code.

```
cd DeepLPF
pip install -r requirements.txt
python -m pytest test_deeplpf.py      # 45 tests, CPU only, no dataset needed
```

## The one commit that changes the model's output

`model: fuse the two scaling maps around their neutral element`. Each map is a
scaling factor, neutral at 1, and they fuse as `1 + (s_g - 1) + (s_e - 1)`, so
a neutral pair leaves the image alone.

The branch ships a checkpoint trained under it, at **24.18 dB / 0.917 SSIM**
over 500 FiveK test images on the reconstructed split, and a second checkpoint
at **23.83 dB / 0.916 SSIM** on the original DPE split, so both of the
protocols this README distinguishes have a checkpoint trained under it.

## Everything else

**Model** (four commits, gradients and geometry only): the binarisation's
straight-through estimator reaches autograd; the graduated filter's branch
selection is differentiable, so its inversion indicator trains; ellipse 2's
three channels share one geometry; and the graduated filter's central line has
its own intercept, with the lower offset expressed as a fraction of the upper,
which gives each of its fifteen line-geometry outputs a path to the loss and
removes the two clamps that bounded them against each other. That last commit
also drops four modules `UNet.__init__` built and never used, whose ten
state-dict entries stopped the shipped checkpoint loading with `strict=True`.

**Loss:** MS-SSIM's top-scale term enters the product once.

**Runnable:** the README's commands run as written, on CPU, MPS or CUDA,
including single-image inference. Image ids match with or without a dash,
greyscale input loads, and inference over photographs with no retouched target
writes enhanced images and skips the metrics.

**Structure:** `model.py`'s five classes split into per-filter modules and
`main()` into `cli`/`train`/`inference`, with every name re-exported and the
module hierarchy untouched, so state-dict keys are unchanged. The filter heads
evaluate their three instances as a batch (dispatched aten ops 4847 → 1965 on
one step; an operation count, not a wall-clock claim).

**Optional, all off by default:** `--learn_filter_count`, `--colour_head`,
`--batch_size` with `--crop_size`, and `--compile` / `--cuda_graphs` / `--tf32`
/ `--amp bf16`.

**README:** rewritten around the shipped checkpoint and the `deeplpf enhance`
entry point, with the architecture, filter and gallery figures and the training
curve.

**Packaging:** `pip install -e .` and `deeplpf enhance photo.jpg`.
`requirements.txt` gives lower bounds that install on current Python, and
scikit-image's `multichannel`/`channel_axis` rename is handled by inspecting
the installed signature.

**Licence:** `LICENSE` is MIT and is not touched. The last commit makes the
file headers and the README describe it as MIT, matching the LICENSE already
shipped. If MIT is not what you intended, drop that commit and change `LICENSE`
instead.

## Verification

Run on CPU (Python 3.11, torch 2.12, scikit-image 0.26, numpy 2.4).

- 45 tests pass, nine of them covering the model and loss updates.
- The refactor, batching and dead-code commits are bitwise identical to the
  tree before them: 90 state-dict keys in the same order, forward and loss
  max difference 0, gradients 3.7e-09.
- End to end on a synthetic dataset: training, checkpointing, evaluation, and
  inference over untargeted images.

**Not verified:** the retrained figure is one 1000-epoch run, not a per-commit
ablation; nothing ran on a GPU, so `--cuda_graphs`, `--compile`, `--tf32` and
`--amp` are unexercised; and neither optional capability is shown to improve
accuracy.

---

*`PULL_REQUEST.md` is the PR body, not repository content, and is not committed
on the branch.*
