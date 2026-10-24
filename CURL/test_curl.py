# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""CPU-only test suite for CURL.

Runs without a dataset and without a GPU:

    python3 -m pytest test_curl.py

The colour-space and curve tests pin down the numerics; the network tests pin
down shapes, determinism and the checkpoint contract. Together they are the
safety net for refactoring: anything that moves a pixel shows up here.
"""
import glob
import os

import numpy as np
import pytest
import torch

os.environ.setdefault('CURL_DEVICE', 'cpu')

import device  # noqa: E402

# Every test runs on CPU regardless of what the machine has, so results are
# comparable across machines and CI.
device.DEVICE = torch.device('cpu')

import data  # noqa: E402
import model  # noqa: E402
import rgb_ted  # noqa: E402
from util import ImageProcessing  # noqa: E402

SEED = 0


def _rand_image(height=32, width=48, seed=SEED):
    """A deterministic HxWx3 image in [0, 1] as a CHW tensor."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(3, height, width, generator=g)


# ---------------------------------------------------------------- colour spaces


def test_rgb_to_lab_roundtrip():
    img = _rand_image()
    lab = ImageProcessing.rgb_to_lab(img.clone())
    back = ImageProcessing.lab_to_rgb(lab)
    assert torch.allclose(back, img, atol=2e-3), \
        'max error %.5f' % (back - img).abs().max()


def test_rgb_to_lab_output_range():
    """The conversion normalises L to [0, 1] and a/b to about [0, 1]."""
    lab = ImageProcessing.rgb_to_lab(_rand_image())
    assert lab.min() >= -0.05 and lab.max() <= 1.05


def test_rgb_to_hsv_roundtrip():
    img = _rand_image()
    hsv = ImageProcessing.rgb_to_hsv(img.clone())
    back = ImageProcessing.hsv_to_rgb(hsv)
    assert torch.allclose(back, img, atol=2e-3), \
        'max error %.5f' % (back - img).abs().max()


def test_hsv_roundtrip_on_structured_pixels():
    """Round-trip the pixels uniform random data never produces.

    torch.rand draws three independent floats, so two channels are never
    exactly equal and none is ever exactly zero. Those are the cases where the
    hue branches can overlap, and a random-input test samples only the region
    where the conversion already works.
    """
    cases = {
        'pure primaries': [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        'secondaries': [(1, 1, 0), (0, 1, 1), (1, 0, 1)],
        'two channels equal': [(v, v, 0.5) for v in (0.0, 0.25, 0.6, 1.0)],
        'one channel zero': [(v, 0.0, 0.5) for v in (0.0, 0.3, 1.0)],
        'greys': [(v, v, v) for v in (0.0, 0.5, 1.0)],
    }
    for name, pixels in cases.items():
        img = torch.tensor(pixels, dtype=torch.float32).T.reshape(3, 1, -1)
        back = ImageProcessing.hsv_to_rgb(ImageProcessing.rgb_to_hsv(img.clone()))
        err = (back - img).abs().max().item() * 255
        assert err < 0.01, '%s: round trip off by %.2f 8-bit levels' % (name, err)


def test_hsv_roundtrip_on_a_real_photograph():
    """Natural images carry ties and zeroed channels that random input does not."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adobe5k_dpe',
                        'curl_example_test_input', 'a4723-_DGW7894.png')
    if not os.path.exists(path):
        pytest.skip('bundled example absent')
    from PIL import Image
    img = torch.from_numpy(np.array(Image.open(path).convert('RGB'))
                           ).float().permute(2, 0, 1) / 255.
    back = ImageProcessing.hsv_to_rgb(ImageProcessing.rgb_to_hsv(img.clone()))
    err = (back - img).abs().max().item() * 255
    assert err < 0.01, 'round trip off by %.2f 8-bit levels on a photograph' % err


def test_rgb_to_hsv_known_colours():
    """Primaries have known hue, and full saturation and value."""
    px = torch.tensor([[[1.0, 0.0, 0.0]],      # red   -> hue 0
                       [[0.0, 1.0, 0.0]],      # green -> hue 1/3
                       [[0.0, 0.0, 1.0]]])     # blue  -> hue 2/3
    hsv = ImageProcessing.rgb_to_hsv(px.permute(2, 0, 1))
    hue = hsv[0].flatten()
    assert torch.allclose(hue, torch.tensor([0.0, 1 / 3, 2 / 3]), atol=1e-3)
    assert torch.allclose(hsv[1].flatten(), torch.ones(3), atol=1e-3)
    assert torch.allclose(hsv[2].flatten(), torch.ones(3), atol=1e-3)


def test_greyscale_has_zero_saturation():
    grey = torch.full((3, 8, 8), 0.4)
    hsv = ImageProcessing.rgb_to_hsv(grey)
    assert hsv[1].abs().max() < 1e-4


# ----------------------------------------------------------------------- curves


def _flat_knots(n, value=0.0):
    """Knots of a curve that scales every pixel by exp(value) ** 1."""
    return torch.full((n,), value)


def test_apply_curve_identity_knots():
    """A curve whose knots are all 1 leaves the channel alone."""
    img = torch.rand(8, 8, 3, generator=torch.Generator().manual_seed(1)) * 0.8
    knots = torch.ones(16)
    reg = torch.zeros(1)
    out, _ = ImageProcessing.apply_curve(img.clone(), knots, reg,
                                         channel_in=0, channel_out=0)
    assert torch.allclose(out[:, :, 0], img[:, :, 0], atol=1e-5)


def test_apply_curve_constant_scale():
    """Flat knots at value k scale the output channel by k."""
    img = torch.full((4, 4, 3), 0.5)
    reg = torch.zeros(1)
    out, _ = ImageProcessing.apply_curve(img.clone(), _flat_knots(16, 0.5), reg,
                                         channel_in=0, channel_out=1)
    assert torch.allclose(out[:, :, 1], torch.full((4, 4), 0.25), atol=1e-5)


def test_apply_curve_leaves_other_channels_alone():
    img = torch.rand(6, 6, 3, generator=torch.Generator().manual_seed(2))
    reg = torch.zeros(1)
    out, _ = ImageProcessing.apply_curve(img.clone(), _flat_knots(16, 0.7), reg,
                                         channel_in=0, channel_out=0)
    assert torch.allclose(out[:, :, 1], img[:, :, 1])
    assert torch.allclose(out[:, :, 2], img[:, :, 2])


def test_apply_curve_output_is_clamped():
    img = torch.full((4, 4, 3), 0.9)
    reg = torch.zeros(1)
    out, _ = ImageProcessing.apply_curve(img.clone(), _flat_knots(16, 10.0), reg,
                                         channel_in=0, channel_out=0)
    assert out.max() <= 1.0 and out.min() >= 0.0


def test_smoothness_regulariser_zero_for_straight_curve():
    """Evenly spaced knots have no slope change, so no penalty."""
    img = torch.rand(4, 4, 3, generator=torch.Generator().manual_seed(3))
    knots = torch.linspace(1.0, 2.0, 16)
    reg = torch.zeros(1)
    _, reg = ImageProcessing.apply_curve(img, knots, reg,
                                         channel_in=0, channel_out=0)
    assert reg.abs().item() < 1e-5


def test_smoothness_regulariser_positive_for_kinked_curve():
    img = torch.rand(4, 4, 3, generator=torch.Generator().manual_seed(4))
    knots = torch.tensor([0.0, 1.0, 0.0, 1.0] * 4)
    reg = torch.zeros(1)
    _, reg = ImageProcessing.apply_curve(img, knots, reg,
                                         channel_in=0, channel_out=0)
    assert reg.item() > 1.0


def test_curve_knots_receive_gradient():
    """The predicted knots must be trainable through the image loss.

    The curve heads exist to be learnt; if the knots are detached from the
    image path they can only be moved by the smoothness regulariser, whose
    weight in the loss is 1e-6.
    """
    img = torch.rand(8, 8, 3, generator=torch.Generator().manual_seed(5))
    knots = torch.ones(16, requires_grad=True)
    reg = torch.zeros(1)
    out, _ = ImageProcessing.apply_curve(img, knots, reg,
                                         channel_in=0, channel_out=0)
    out.sum().backward()
    assert knots.grad is not None, 'no gradient reached the knots at all'
    assert knots.grad.abs().sum() > 0, 'knot gradient is identically zero'


def test_curve_passes_through_its_knots():
    """The defining property: at x = k/steps the curve's value is C[k].

    Without the hinge every segment ramps across the whole range and the
    fifteen of them sum to one straight line, so only two knots survive.
    """
    torch.manual_seed(6)
    knots = torch.exp(torch.randn(16) * 0.15)
    steps = 15
    x = torch.arange(16).float() / steps
    img = torch.zeros(1, 16, 3)
    img[0, :, 0] = x        # channel 0 drives the curve
    img[0, :, 1] = 0.001    # channel 1 receives it, far from the clamp
    out, _ = ImageProcessing.apply_curve(img.clone(), knots, torch.zeros(1),
                                         channel_in=0, channel_out=1)
    assert torch.allclose(out[0, :, 1] / 0.001, knots, atol=1e-4)


def test_a_knot_only_affects_its_own_neighbourhood():
    """Moving one knot must be a local edit, not a global one."""
    torch.manual_seed(7)
    base = torch.ones(16)
    moved = base.clone()
    moved[8] = 2.0
    img = torch.zeros(1, 16, 3)
    img[0, :, 0] = torch.arange(16).float() / 15
    img[0, :, 1] = 0.001
    a, _ = ImageProcessing.apply_curve(img.clone(), base, torch.zeros(1), 0, 1)
    b, _ = ImageProcessing.apply_curve(img.clone(), moved, torch.zeros(1), 0, 1)
    delta = (b[0, :, 1] - a[0, :, 1]).abs() / 0.001
    assert delta[8] > 0.5, 'the moved knot did not move its own sample'
    assert delta[:7].max() < 1e-4, 'knot 8 changed the curve below knot 7'
    assert delta[10:].max() < 1e-4, 'knot 8 changed the curve above knot 9'


def test_every_curve_segment_is_used():
    """A curve with 16 knots has 15 segments and all of them must apply.

    Two curves that differ only in their last knot must produce different
    output on a pixel that falls in the last segment.
    """
    img = torch.full((4, 4, 3), 0.99)  # lands in the final segment
    a = torch.ones(16)
    b = torch.ones(16)
    b[-1] = 5.0
    out_a, _ = ImageProcessing.apply_curve(img.clone(), a, torch.zeros(1), 0, 0)
    out_b, _ = ImageProcessing.apply_curve(img.clone(), b, torch.zeros(1), 0, 0)
    assert not torch.allclose(out_a, out_b), \
        'the last curve segment has no effect on the output'


# ------------------------------------------------------------------ the network


def test_ted_output_shape():
    torch.manual_seed(SEED)
    net = rgb_ted.TEDModel()
    out = net(_rand_image(64, 64).unsqueeze(0))
    assert out.shape[0] == 1 and out.shape[2:] == (64, 64)


def test_curlnet_forward_shape_and_range():
    torch.manual_seed(SEED)
    net = model.CURLNet().eval()
    img = _rand_image(64, 64).unsqueeze(0)
    with torch.no_grad():
        out, reg = net(img)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert torch.isfinite(reg).all()


def test_curlnet_rejects_images_below_the_minimum_size():
    """The pooling depth sets a floor of 48 px on each edge.

    Anything smaller collapses a feature map to zero and raises inside
    max_pool2d, so the constraint is documented here rather than discovered.
    """
    torch.manual_seed(SEED)
    net = model.CURLNet().eval()
    with torch.no_grad():
        net(torch.rand(1, 3, 48, 48))  # the floor: must not raise
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            net(torch.rand(1, 3, 32, 32))


@pytest.mark.parametrize('height,width', [(48, 64), (63, 65), (64, 64), (65, 63)])
def test_curlnet_preserves_shape_for_odd_sizes(height, width):
    """The padding blocks in rgb_ted exist for non-square, odd sizes."""
    torch.manual_seed(SEED)
    net = model.CURLNet().eval()
    img = torch.rand(1, 3, height, width)
    with torch.no_grad():
        out, _ = net(img)
    assert out.shape == img.shape


def test_curlnet_is_deterministic():
    torch.manual_seed(SEED)
    net = model.CURLNet().eval()
    img = _rand_image(64, 64).unsqueeze(0)
    with torch.no_grad():
        a, _ = net(img)
        b, _ = net(img)
    assert torch.equal(a, b)


def test_curlnet_output_is_finite():
    torch.manual_seed(SEED)
    net = model.CURLNet().eval()
    for fill in (0.0, 1.0):  # pure black and pure white stress the conversions
        img = torch.full((1, 3, 64, 64), fill)
        with torch.no_grad():
            out, reg = net(img)
        assert torch.isfinite(out).all(), 'non-finite output for fill %s' % fill
        assert torch.isfinite(reg).all()


# Constructed in rgb_ted.TED.__init__ and never referenced in its forward, so
# they are optimised as no-ops. They are present in the shipped checkpoint, so
# deleting them is a state-dict-breaking change and a separate decision.
UNUSED_MODULES = ('tednet.ted.conv1', 'tednet.ted.conv2', 'tednet.ted.conv3',
                  'tednet.ted.local_net')


def test_curve_layer_trains_on_the_image_loss():
    """Regression test for the detached-curve bug.

    The curve heads and their conv blocks predict the knots. If they are cut
    off from the image path they can only be moved by the smoothness
    regulariser, which enters the loss at 1e-6 and whose optimum is a straight
    line - so the curve half of CURL would never learn the enhancement.
    """
    torch.manual_seed(SEED)
    net = model.CURLNet()
    out, _ = net(_rand_image(64, 64).unsqueeze(0))
    out.sum().backward()  # image loss only, deliberately no regulariser
    dead = [name for name, p in net.named_parameters()
            if name.startswith('curllayer.')
            and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not dead, 'curve-layer parameters with no image-loss gradient: %s' % dead


def test_backbone_trains_apart_from_the_known_unused_modules():
    torch.manual_seed(SEED)
    net = model.CURLNet()
    out, reg = net(_rand_image(64, 64).unsqueeze(0))
    (out.sum() + reg.sum()).backward()
    dead = [name for name, p in net.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    unexpected = [n for n in dead
                  if not any(n.startswith(m + '.') for m in UNUSED_MODULES)]
    assert not unexpected, 'parameters with no gradient: %s' % unexpected[:10]


# ------------------------------------------------------------------------ loss


def test_loss_is_small_for_identical_images():
    torch.manual_seed(SEED)
    criterion = model.CURLLoss(ssim_window_size=5)
    img = _rand_image(64, 64).unsqueeze(0)
    loss = criterion(img, img.clone(), torch.zeros(1))
    assert loss.item() < 1e-2, 'loss %.5f on identical images' % loss.item()


def test_loss_increases_with_error():
    torch.manual_seed(SEED)
    criterion = model.CURLLoss(ssim_window_size=5)
    target = _rand_image(64, 64).unsqueeze(0)
    near = torch.clamp(target + 0.02, 0, 1)
    far = torch.clamp(target + 0.20, 0, 1)
    assert (criterion(near, target, torch.zeros(1)).item()
            < criterion(far, target, torch.zeros(1)).item())


def test_loss_is_differentiable():
    torch.manual_seed(SEED)
    criterion = model.CURLLoss(ssim_window_size=5)
    pred = _rand_image(64, 64).unsqueeze(0).requires_grad_(True)
    target = _rand_image(64, 64, seed=SEED + 1).unsqueeze(0)
    criterion(pred, target, torch.zeros(1)).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0


# ------------------------------------------------------------------ orientation


def test_square_target_is_not_transposed():
    """The old test compared input height to target width, true for squares."""
    inp = torch.rand(3, 32, 32)
    target = torch.rand(3, 32, 32)
    assert torch.equal(data.match_orientation(inp, target), target)


def test_transposed_target_is_corrected():
    inp = torch.rand(3, 20, 40)
    target = torch.rand(3, 40, 20)
    out = data.match_orientation(inp, target)
    assert out.shape == inp.shape
    assert torch.equal(out, target.permute(0, 2, 1))


def test_matching_target_is_left_alone():
    inp = torch.rand(3, 20, 40)
    target = torch.rand(3, 20, 40)
    assert torch.equal(data.match_orientation(inp, target), target)


# --------------------------------------------------------------------- metrics


def test_psnr_of_identical_images_is_finite_and_large():
    """An exact match must not return inf, which would poison the average."""
    img = _rand_image().unsqueeze(0).numpy()
    psnr = ImageProcessing.compute_psnr(img, img.copy(), 1.0)
    assert np.isfinite(psnr) and psnr > 90


def test_psnr_matches_the_definition():
    a = np.zeros((1, 3, 8, 8), dtype=np.float32)
    b = np.full((1, 3, 8, 8), 0.1, dtype=np.float32)
    expected = 10 * np.log10(1.0 / (0.1 ** 2))
    assert abs(ImageProcessing.compute_psnr(a, b, 1.0) - expected) < 1e-3


def test_ssim_uses_a_fixed_data_range():
    """SSIM must not depend on the contrast of the particular target.

    With data_range taken from the image, a low-contrast pair scored
    differently from a high-contrast one, so the reported number was not
    comparable between images or with published figures.
    """
    torch.manual_seed(9)
    a = _rand_image(64, 64).unsqueeze(0).numpy()
    b = np.clip(a + 0.05, 0, 1)
    full = ImageProcessing.compute_ssim(a, b)
    # the same pair, compressed into a narrow band around mid grey
    low = ImageProcessing.compute_ssim(a * 0.2 + 0.4, b * 0.2 + 0.4)
    assert low > full, 'a lower-contrast pair should not score worse'


def test_ssim_of_identical_images_is_one():
    img = _rand_image(64, 64).unsqueeze(0).numpy()
    assert abs(ImageProcessing.compute_ssim(img, img.copy()) - 1.0) < 1e-4


def test_loss_is_a_scalar():
    """The loss must be a scalar the training loop can log and save.

    It was shape (1, 1), and the checkpoint path unpacked it with .tolist()[0].
    """
    torch.manual_seed(SEED)
    criterion = model.CURLLoss(ssim_window_size=5)
    img = _rand_image(64, 64).unsqueeze(0)
    loss = criterion(img, img.clone(), torch.zeros(1))
    assert loss.numel() == 1
    assert isinstance(float(loss), float)


def test_modules_re_export_the_old_names():
    """util.ImageProcessing still resolves, for code written against it."""
    import util
    for name in ('rgb_to_lab', 'lab_to_rgb', 'apply_curve', 'compute_psnr'):
        assert callable(getattr(util.ImageProcessing, name))
        assert callable(getattr(util, name))


# ------------------------------------------------------------------ checkpoint


# Whichever checkpoint is shipped, rather than one named here: a name pinned in
# the test goes stale the moment a different checkpoint is shipped, and the test
# then skips instead of failing.
def _shipped_checkpoints():
    pattern = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'pretrained_models', '*', '*.pt')
    return sorted(glob.glob(pattern))


@pytest.mark.skipif(not _shipped_checkpoints(), reason='no checkpoint shipped')
@pytest.mark.parametrize('checkpoint_path', _shipped_checkpoints() or [None])
def test_shipped_checkpoint_loads_into_the_current_network(checkpoint_path):
    """The state-dict contract: every key the checkpoint has, the net wants."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint)
    net = model.CURLNet()
    missing, unexpected = net.load_state_dict(state, strict=False)
    assert not unexpected, 'checkpoint has keys the network does not: %s' % unexpected[:5]
    assert not missing, 'network has parameters the checkpoint does not: %s' % missing[:5]
