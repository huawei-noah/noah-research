# DeepLPF: code upgrade, a runnable entry point, a module split, and packaging

22 commits against `DeepLPF/`. Nothing outside that directory is touched, and
`LICENSE` is not modified.

I am one of the authors of this code. Everything here is maintenance of the
released implementation: the architecture and the paper's method are unchanged,
and the checkpoint in `pretrained_models/adobe_dpe/` still loads with
`strict=True`, its 90 keys unchanged.

**One commit does change what the released weights produce, and it is the
reason to read this PR carefully.** `model: fuse the two scaling maps around
their neutral element` alters the function the network computes, so the 2020
checkpoint needs retraining. Retrained, the model reaches **24.18 dB / 0.917
SSIM**, against **23.74 dB / 0.910** for the released checkpoint on the same
images. Every other commit is output-preserving, verified bitwise. That commit
can be dropped on its own; the other four model updates do not depend on
it.

```
cd DeepLPF
pip install -r requirements.txt
python -m pytest test_deeplpf.py      # 44 tests, CPU only, no dataset needed
```

## What a reviewer should look at first

1. **The five model updates** (commits 1–5). These are the reason for the
   PR. Each is small, each has a test, and each of those tests fails on the
   current `master` code.
2. **`model: fuse the two scaling maps around their neutral element`** — the
   only commit that changes the function the network computes. Read that one
   with checkpoint compatibility in mind. Everything else is
   output-preserving.
3. Then, if the rest is of interest, the refactor commits from
   `refactor: split model.py into per-filter modules` onwards. They are
   independent of 1–5 and can be dropped without affecting them.

## Which commits change the computed function

Only one, and it is deliberate:

| Commit | Changes output? |
|---|---|
| `model: fuse the two scaling maps around their neutral element` | **Yes.** Existing checkpoints will not reproduce their previous outputs and would need retraining or fine-tuning. |
| `model: connect the binarisation's straight-through estimator to autograd` | No — gradients only. |
| `model: keep the graduated filter's inversion indicator in the graph` | No — forward values bit-identical; gradients only. |
| `model: give ellipse 2's three channels one geometry` | Changes the mask an untrained network produces, but the affected ellipse's geometry was inconsistent with its own channels; a retrained model is required to benefit either way. |
| `loss: group the MS-SSIM product so the top-scale term enters once` | Changes the loss value, not the network. Affects training, not inference from an existing checkpoint. |
| `feat: run on CPU and Apple Silicon as well as CUDA` | No. |
| `refactor: split model.py …`, `perf: evaluate the filter instances as a batch …`, `refactor: split main.py …`, `refactor: remove parameters and helpers that nothing reads` | No — verified bitwise, see below. |
| `feat: optional learned filter count and colour head` | No, with both flags off (the default). |
| `feat: train at a batch size greater than one` | No, at the default batch size of 1. |
| `docs: name the MIT licence the repository actually ships` | No — comments and README only. |
| `perf: optional torch.compile, CUDA graphs, TF32 and bf16 autocast` | No, with all four flags off (the default). Each flag documents its own numerical effect when on. |

The elliptical semi-axis and MS-SSIM commits change what an untrained network
computes, so they sit in the first group rather than the behaviour-preserving
one, and are listed separately above so the distinction is explicit.

## 1. Model updates

**`BinaryLayer`'s straight-through estimator, reached by autograd.** Its
`backward()` was a plain method on an `nn.Module`, and autograd dispatches to
the `backward` of a `torch.autograd.Function` rather than to a method on a
Module, so the propagated gradient was `torch.sign`'s. Moving it into a
`torch.autograd.Function` connects the estimator to the graph and lets the
inversion indicator `g_inv` of Sec. 3.2.2 train. The forward value is
unchanged.

*Verified:* a scalar loss on the layer's output now yields gradient 1 at the
input for `|x| <= 1` and 0 for `|x| > 1`; previously 0 in both cases.

**The graduated filter's branch selection, made differentiable.** It used a
Python `if` on `invert`; a branch condition carries no gradient, so `g_inv` did
not reach the loss through it. Both branches are now evaluated and blended as
`invert * mask_inv + (1 - invert) * mask_non`. `invert` is exactly 0 or 1, so
the forward result is bit-identical to the branch previously selected.

*Verified:* forward outputs match the original implementation exactly over
random parameter draws for both indicator values; the gradient at the
pre-binarisation logit is now non-zero.

**Ellipse 2's three channels share one geometry.** `mask_scale6` passed
`semi_axis_y=b3` where its two siblings passed `b2`, giving the blue mask a
different support from the R and G masks beside it. One character.

*Verified:* after the batched rewrite this is structural — geometry arguments
carry an instance dimension and no channel dimension, so an instance's three
channels cannot see different ellipses. Two tests assert that shape contract
and the resulting equality of an instance's channel masks at equal scale
factors.

**The two scaling maps fuse around their neutral element.** Each map is a
scaling factor, neutral at 1, so `s_g + s_e` reaches 2 where both branches are
neutral. Now `1 + (s_g - 1) + (s_e - 1)`: the same sum of contributions, with
a neutral pair leaving the image alone.

*Verified:* with both branches forced neutral the fused map is all ones and
the head output equals its input; the previous expression gave all twos.

The released network learned around the old expression: over FiveK test images
it predicts `s_g ≈ 0.33` and `s_e ≈ 1.23`, summing to ≈ 1.57, the multiplier it
wanted. Those weights therefore need retraining. Retrained, the model reaches
**24.18 dB / 0.917 SSIM**, against **23.74 dB / 0.910** for the released
checkpoint on the same test images.

**MS-SSIM's top-scale term enters once.** `torch.prod(pow1[:-1] * pow2[-1])`
broadcasts the scalar `ssim_J^{w_J}` across the four `cs` terms before
reducing, contributing it as `ssim_J^{4 w_J}`.

*Verified:* the value now matches `prod(cs^w) * ssim_J^{w_J}` computed
independently, to floating-point tolerance, and equals 1 for identical inputs.

## 2. Getting the documented commands to run

The README's commands now run as written. What stood between them and that:

- `main()` opened with an unconditional `exit()`.
- Three training-only flags were `required=True`, so argparse rejected the
  inference command before `main()` ran.
- The log directory and TensorBoard writer were created before argparse, so
  `--help` left an empty `log_<timestamp>/` and `runs/` behind.
- `--checkpoint_filepath` was accepted on the training path and never read, so
  fine-tuning began from random weights.
- `loss.data[0]` indexes a 0-d tensor, an `IndexError` since PyTorch 0.5.
- Image ids came from `file.split("-")[0]`, which suits
  `a0001-jmac_DSC1459.png` but leaves `sunset.png` yielding `sunset.png` and
  never matching the id `sunset`.
- The post-scan check `assert 'input_img' in imgs` held with nothing found,
  since entries are created with both keys set to `None`; the failure surfaced
  later inside a DataLoader worker as `'NoneType' object has no attribute
  'read'`.
- Inference over your own photographs has no retouched target, so that path now
  writes enhanced images and skips the metrics.
- Greyscale input reached the first convolution as a 1-channel tensor.
- `num_workers` 6 -> 0, so an exception while loading an image keeps its
  traceback.

*Verified:* `main.py --help` exits 0 and creates no directories; each
incomplete command exits with a message naming the missing flags; and the
whole pipeline runs end to end on a synthetic dataset (below).

## 3. Device-agnostic execution

Every tensor and module was placed with an unconditional `.cuda()` and the
loss allocated `torch.cuda.FloatTensor`, so the code could not run without an
NVIDIA GPU — including single-image inference. `util.get_device()` picks CUDA,
else MPS, else CPU. Also removed: a per-image `torch.cuda.empty_cache()`, an
`x.contiguous()` whose result was discarded, and the `Variable` wrappers (a
no-op since PyTorch 0.4).

## 4. Module split

`model.py` held five classes in a thousand lines. Class bodies moved verbatim
into `blocks.py`, `filtercommon.py`, `cubic.py`, `graduated.py`,
`elliptical.py`, `losses.py`, with `filters.py` as an import surface;
`model.py` re-exports every moved name, so `model.CubicFilter` and
`model.DeepLPFLoss` resolve as before. The module hierarchy of the network is
untouched, so state-dict keys are unchanged.

`main.py`'s single ~390-line `main()` becomes `cli.py` (flags), `train.py`
(training loop), `inference.py` (checkpoint over a directory), with `main.py`
reduced to logging, seeding, device selection and dispatch.

A separate commit removes parameters and helpers nothing reads — each verified
dead in this tree, with no caller passing any of them: `ConvBlock`'s ignored
`stride` (`conv3x3` was called with a hard-coded 2), `GlobalPoolingBlock`'s
`receptive_field`, `batch_size` on `CubicFilter` and
`DeepLPFParameterPrediction`, `alpha` on the loss, `is_training` on
`rgb_to_lab`, `swapimdims_HW3_3HW`, and two unused attributes. This changes
public constructor signatures; nothing in the repository is affected.

## 5. Batched filter evaluation

The heads wrote out their three instances as separate statements over scalar
per-image parameters — nine `get_mask` calls, nine `get_inverted_mask` calls, a
loop over channels — rebuilt the coordinate grids every forward pass, and each
resized its own concatenation of features and image. Per-instance parameters
are now kept as `(B, 3)` vectors that broadcast over the grids, so the nine
`(instance, channel)` masks come from one call; the grids and the loss's
Gaussian window are cached; and the 300×300 resize is done once and shared
(bilinear resize is per channel, so `upsample(cat(a, b)) == cat(upsample(a),
upsample(b))`).

The graduated head's four Python branches over `factor >= 1` become
`torch.where` selections, which also fixes a latent batch bug: `.all()` let the
first image in a batch decide the branch for the rest.

*Measured*, one 64×64 training step, dispatched aten operations counted with
`TorchDispatchMode`:

| | before | after |
|---|---|---|
| forward | 2071 | 826 |
| backward | 2776 | 1139 |
| **total** | **4847** | **1965** (−59%) |

That is an operation count on CPU, not a wall-clock or GPU measurement, and no
speed-up is claimed from it.

## 6. Optional capabilities, both off by default

`--learn_filter_count` predicts one gate per filter instance and applies
`1 + g * (s - 1)` before the product, so a gate at zero makes its instance
exactly 1 and drops out of the fuse; the mean gate is added to the loss as
`--gate_weight * gate_penalty`. The branch's FC layer grows 24 → 27 outputs, so
gated and ungated checkpoints are not interchangeable.

`--colour_head` adds a global 3×3 colour mixer with offset and a per-channel
piecewise-linear tone curve, applied to Y1 before the cubic filter. Every
filter head is diagonal, so no head can express white balance, saturation or a
hue shift. Zero-initialised, so it is exactly the identity at initialisation,
and constructed after every other module so the RNG stream is untouched.

Both are constructor arguments, not global state.

**Not verified:** whether either improves accuracy. That needs a FiveK training
run.

## 7. Batch size > 1, and optional performance flags

`--batch_size` with `--crop_size` (a random square crop, the same window from
input and target) makes training above batch size 1 possible; evaluation and
inference still run at 1 so per-image metrics are reported individually.

`--compile`, `--cuda_graphs`, `--tf32` and `--amp bf16` are opt-in, all off by
default, each documenting its own numerical effect. `trainstep.py` holds the
eager step and the CUDA-graph capture.

**Not verified: none of these four ran on a GPU.** No CUDA device was available.
The capture path was exercised only through a stub that substitutes an eager
replay, which tests the bookkeeping (per-shape buffers, the
parameter/optimiser snapshot that stops warm-up steps advancing training) and
not the capture itself. A reviewer with a GPU should treat them as unexercised.

## 8. Packaging and a `deeplpf enhance` command

`main.py` wants a dataset directory, an ids file and a checkpoint path — the
right shape for training, the wrong shape for "enhance this photograph".

```
deeplpf enhance photo.jpg
deeplpf enhance ~/photos --out ~/enhanced --checkpoint my_model.pt
```

It defaults to the bundled checkpoint, picks a device, accepts greyscale and
RGBA, reports a file it cannot use and carries on. `pyproject.toml` makes the
repository installable and registers the script; the modules are listed under
`py-modules` because they sit at the top level, which keeps the existing layout.

`requirements.txt` pinned exact 2020 versions that do not build on current
Python, and listed `skimage==0.0`, a PyPI placeholder that is not
scikit-image. Pins become lower bounds. scikit-image 0.19 renamed
`structural_similarity`'s `multichannel=True` to `channel_axis` and 0.23
removed the old name; the argument is now chosen by inspecting the installed
signature.

## Verification, in full

Everything below was run on CPU (macOS, Python 3.11, torch 2.12,
scikit-image 0.26, numpy 2.4).

**Test suite:** 44 tests pass.

**Coverage of the updates.** Each update carries a test that pins the new
behaviour. Run against the original `master` model (with only the device
changes applied so it runs on CPU), these eight distinguish the two trees:

```
test_binary_layer_passes_gradient_through
test_inversion_indicator_receives_gradient
test_ellipse_channel_masks_share_their_geometry
test_ellipse_geometry_is_per_instance_not_per_channel
test_two_neutral_branches_fuse_to_neutral
test_msssim_matches_the_reference_grouping
test_batched_forward_matches_per_image_forward
test_graduated_branch_decides_per_image_not_per_batch
```

(Tests for capabilities `master` does not have — gates, colour head, the
console script — are not counted above.)

**Behaviour preservation.** The refactor, batching, dead-code and optional-
capability commits were checked against the tree as it stood after the
model updates and before the refactor, from identical seeds and input:

```
state-dict keys   identical (90, same order)
forward output    bitwise identical  (max |difference| = 0)
loss              bitwise identical  (max |difference| = 0)
gradients         max |difference| 3.7e-09, max relative 1.1e-06
```

The residual gradient difference is the changed reduction order in the batched
expressions, at the level expected of float32.

**The released checkpoint.** `pretrained_models/adobe_dpe/deeplpf_validpsnr_23.31_…_epoch_499_model.pt`
loads into the current model with `strict=True` — no missing or unexpected
keys — and produces **bitwise-identical** output to the same checkpoint run
through the pre-refactor tree. That comparison holds the fusion fix constant on
both sides, so it establishes that the refactor, batching and dead-code commits
preserve behaviour — not that the branch as a whole preserves the released
model's output, which the fusion fix in §1 changes by design.

**End to end.** On a synthetic 4-image dataset: two training epochs run,
checkpoints and `test_per_image.csv` are written, the saved checkpoint scores
that dataset through the inference path, and a second directory of inputs with
no targets produces enhanced PNGs and no metrics. `deeplpf enhance` was also
run against the repository's own bundled checkpoint on real image files.

**Packaging.** `pip install -e .` into a clean virtualenv installs the
`deeplpf` console script and registers the entry point.

## What was not verified

- **One training run.** The 24.18 dB / 0.917 is a single 1000-epoch run, not a
  per-commit ablation.
- **No GPU.** Everything ran on CPU. `--cuda_graphs`, `--compile`, `--tf32` and
  `--amp` have never been executed on a CUDA device.
- Training quality at batch size > 1, and whether either optional capability
  helps.

## 9. The licence descriptions, corrected to match LICENSE

DeepLPF states its licence three ways, and they disagree with each other:
`DeepLPF/LICENSE` is the **MIT** licence, the source-file headers say
"BSD 0-Clause License", and `README.md` says "BSD-3-Clause License".

`LICENSE` is the operative document and is **not touched**. The last commit
corrects the two descriptions of it: the headers' two licence sentences now
name the MIT License (Huawei's copyright line and the warranty-disclaimer
wording are unchanged — only the licence name moved, 39 lines across 19
files), and the README's License section says MIT and links to `LICENSE`.

MIT is also the norm across noah-research: CLIFF, CPNDet, EvoFabric, Maha_OOD,
PocketLLM, QuantWM, ROOT, ScienceFlow, SteReFo, SumTitles, conv_graph,
freegbdt and o2despy all ship an MIT LICENSE; only mRNN-mLSTM and S3-Training
use BSD 3-Clause.

**This is documentation catching up with the LICENSE the repository already
distributes DeepLPF under, not a change of licence.** If MIT is not what you
intended for DeepLPF, then `LICENSE` is the file to change and this commit
should be dropped instead — your call either way, and it is the last commit on
the branch so it drops cleanly.

---

*Note on this file:* `PULL_REQUEST.md` is not committed on the branch — it is
the PR body, not repository content.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01QBHVyFGKmzCdeyss5ZxZJg
