# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
'''
Tests for the DeepLPF implementation.

Run with:  python -m pytest test_deeplpf.py

These are CPU-only and use synthetic images, so they need no dataset and no
GPU. They cover the pieces where a mistake is silent: the differentiability of
the binarised inversion indicator, the fusion of the two scaling maps, the
MS-SSIM product, the geometry shared by one ellipse's three channel masks, and
the data loader's file matching.
'''
import os

import numpy as np
import pytest
import torch
from PIL import Image

import data
import model
import trainstep
import util


# ---------------------------------------------------------------- binarisation

def test_binary_layer_forward_is_sign():
    layer = model.BinaryLayer()
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert torch.equal(layer(x), torch.sign(x))


def test_binary_layer_passes_gradient_through():
    """The straight-through estimator must actually be called by autograd."""
    layer = model.BinaryLayer()
    x = torch.tensor([-0.5, 0.5], requires_grad=True)
    layer(x).sum().backward()
    # torch.sign's own gradient is zero; the STE's is one inside |x| <= 1.
    assert torch.equal(x.grad, torch.ones(2))


def test_binary_layer_zeroes_saturated_gradient():
    layer = model.BinaryLayer()
    x = torch.tensor([-2.0, 2.0], requires_grad=True)
    layer(x).sum().backward()
    assert torch.equal(x.grad, torch.zeros(2))


# ------------------------------------------------------------ graduated filter

def _reference_inverted_mask(factor, invert, d1, d2, max_scale, top_line):
    """The original branch-selecting implementation, for a forward comparison."""
    if (invert == 1).all():
        if (factor >= 1).all():
            diff = ((factor - 1)) / 2 + 1
            mask = torch.clamp(factor + ((diff - factor) / d1) * top_line
                               + ((1 - diff) / d2) * top_line, min=1, max=max_scale)
        else:
            diff = ((1 - factor)) / 2 + factor
            mask = torch.clamp(factor + ((diff - factor) / d1) * top_line
                               + ((1 - diff) / d2) * top_line, min=0, max=1)
    else:
        if (factor >= 1).all():
            diff = ((factor - 1)) / 2 + 1
            mask = torch.clamp(1 + ((diff - factor) / d1) * top_line
                               + ((factor - diff) / d2) * top_line, min=1, max=max_scale)
        else:
            diff = ((1 - factor)) / 2 + factor
            mask = torch.clamp(1 + ((diff - 1) / d1) * top_line
                               + ((factor - diff) / d2) * top_line, min=0, max=1)
    return torch.clamp(mask.unsqueeze(0), 0, max_scale)


@pytest.mark.parametrize("invert_value", [0.0, 1.0])
@pytest.mark.parametrize("factor_value", [0.4, 1.6])
def test_inverted_mask_matches_branching_implementation(invert_value, factor_value):
    torch.manual_seed(0)
    filt = model.GraduatedFilter()
    factor = torch.tensor(factor_value)
    invert = torch.tensor(invert_value)
    d1, d2 = torch.tensor(0.4), torch.tensor(0.3)
    top_line = torch.rand(8, 8)

    got = filt.get_inverted_mask(factor, invert, d1, d2, 2, top_line)
    want = _reference_inverted_mask(factor, invert, d1, d2, 2, top_line)
    assert torch.allclose(got, want, atol=0, rtol=0)


def test_inversion_indicator_receives_gradient():
    """A gradient must reach the logit the indicator is binarised from."""
    torch.manual_seed(0)
    filt = model.GraduatedFilter()
    logit = torch.tensor(0.7, requires_grad=True)
    invert = (filt.bin_layer(logit) + 1) / 2
    mask = filt.get_inverted_mask(torch.tensor(1.5), invert,
                                  torch.tensor(0.4), torch.tensor(0.3), 2,
                                  torch.rand(8, 8))
    mask.sum().backward()
    assert logit.grad is not None and logit.grad.abs() > 0


# ----------------------------------------------------------- elliptical filter

def test_ellipse_channel_masks_share_their_geometry():
    """Each ellipse's three channel masks must differ only in scale factor.

    The nine (instance, channel) masks are produced by one batched get_mask
    call. The geometry arguments carry an instance dimension and no channel
    dimension, so all three channels of an instance necessarily see the same
    ellipse; only scale_factor is per channel. Assert exactly that shape
    contract, which is what a per-channel geometry argument would violate.
    """
    torch.manual_seed(0)
    filt = model.EllipticalFilter()
    calls = []
    real_get_mask = filt.get_mask

    def recording_get_mask(x_axis, y_axis, **kwargs):
        calls.append(kwargs)
        return real_get_mask(x_axis, y_axis, **kwargs)

    filt.get_mask = recording_get_mask
    filt.get_elliptical_mask(torch.rand(1, 61, 32, 32), torch.rand(1, 3, 32, 32))

    assert len(calls) == 1
    kwargs = calls[0]
    # (batch, instance, channel, H, W): geometry is broadcast over channels.
    for key in ("shift_x", "shift_y", "semi_axis_x", "semi_axis_y"):
        assert kwargs[key].shape == (1, 3, 1, 1, 1), key
    assert kwargs["scale_factor"].shape == (1, 3, 3, 1, 1)
    # The semi-axes belong to the same ellipse index as the centre.
    assert kwargs["semi_axis_y"].shape[1] == kwargs["shift_x"].shape[1]


def test_ellipse_geometry_is_per_instance_not_per_channel():
    """Two channels of one instance must produce the same mask when their
    scale factors are equal, which fails if they use different semi-axes."""
    torch.manual_seed(0)
    filt = model.EllipticalFilter()
    grid_h = grid_w = 16
    x_axis = torch.arange(grid_h).view(-1, 1).repeat(1, grid_w) / grid_h
    y_axis = torch.arange(grid_w).repeat(grid_h, 1) / grid_w

    geom = dict(shift_x=torch.rand(1, 3, 1, 1, 1),
                shift_y=torch.rand(1, 3, 1, 1, 1),
                semi_axis_x=torch.rand(1, 3, 1, 1, 1) + 0.1,
                semi_axis_y=torch.rand(1, 3, 1, 1, 1) + 0.1,
                alpha=torch.rand(1, 3, 1, grid_h, grid_w),
                radius=torch.rand(1, 3, 1, 1, 1) + 0.1)
    scale = torch.full((1, 3, 3, 1, 1), 1.5)
    mask = filt.get_mask(x_axis, y_axis, scale_factor=scale, **geom)
    for instance in range(3):
        assert torch.equal(mask[0, instance, 0], mask[0, instance, 1])
        assert torch.equal(mask[0, instance, 0], mask[0, instance, 2])


# ------------------------------------------------------------------- the fuse

def test_two_neutral_branches_fuse_to_neutral():
    """S = 1 + (s_g - 1) + (s_e - 1) leaves a neutral image untouched."""
    torch.manual_seed(0)
    head = model.DeepLPFParameterPrediction()
    neutral = torch.ones(1, 3, 16, 16)
    head.graduated_filter.mask_from_input = lambda feat, img: neutral
    head.elliptical_filter.mask_from_input = lambda feat, img: neutral
    cubic = torch.full((1, 3, 16, 16), 0.25)
    head.cubic_filter.mask_from_input = lambda feat, img: cubic

    x = torch.zeros(1, 64, 16, 16)
    out = head(x)
    # img (Y1) is zero here, so the output is exactly the fused Y3 = S * Y2.
    assert torch.allclose(out, cubic)


# ------------------------------------------------------------------- MS-SSIM

def test_msssim_of_identical_images_is_one():
    loss = model.DeepLPFLoss()
    img = torch.rand(1, 1, 64, 64)
    assert loss.compute_msssim(img, img).item() == pytest.approx(1.0, abs=1e-4)


def test_msssim_matches_the_reference_grouping():
    torch.manual_seed(0)
    loss = model.DeepLPFLoss()
    a, b = torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64)

    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    x, y = a.clone(), b.clone()
    ssims, mcs = [], []
    for _ in range(weights.numel()):
        ssim_val, cs = loss.compute_ssim(x, y)
        ssims.append(ssim_val)
        mcs.append(cs)
        x = torch.nn.functional.avg_pool2d(x, (2, 2))
        y = torch.nn.functional.avg_pool2d(y, (2, 2))
    ssims = (torch.stack(ssims) + 1) / 2
    mcs = (torch.stack(mcs) + 1) / 2
    want = torch.prod(mcs[:-1] ** weights[:-1]) * (ssims[-1] ** weights[-1])

    assert loss.compute_msssim(a, b).item() == pytest.approx(want.item(), rel=1e-5)


# ------------------------------------------------------- colour space, metrics

def test_rgb_to_lab_against_skimage():
    from skimage.color import rgb2lab
    torch.manual_seed(0)
    img = torch.rand(3, 16, 16)
    got = util.ImageProcessing.rgb_to_lab(img.clone())

    # rgb_to_lab returns CxHxW with each channel rescaled to [0, 1].
    got = got.permute(1, 2, 0).detach().numpy()
    got = np.stack([got[..., 0] * 100.0,
                    (got[..., 1] * 2 - 1) * 110.0,
                    (got[..., 2] * 2 - 1) * 110.0], axis=-1)
    want = rgb2lab(img.permute(1, 2, 0).numpy())
    assert np.allclose(got, want, atol=1e-2)


def test_compute_psnr_against_skimage():
    from skimage.metrics import peak_signal_noise_ratio
    rng = np.random.default_rng(0)
    a = rng.random((1, 3, 32, 32)).astype(np.float32)
    b = rng.random((1, 3, 32, 32)).astype(np.float32)
    got = util.ImageProcessing.compute_psnr(a, b, 1.0)
    want = peak_signal_noise_ratio(a[0], b[0], data_range=1.0)
    assert got == pytest.approx(want, rel=1e-4)


# ---------------------------------------------------------------- data loading

def _write_image(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array).save(path)


def test_ids_match_filenames_with_and_without_a_dash(tmp_path):
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    _write_image(str(tmp_path / "input" / "a0001-jmac_DSC1459.png"), pixels)
    _write_image(str(tmp_path / "output" / "a0001-jmac_DSC1459.png"), pixels)
    _write_image(str(tmp_path / "input" / "sunset.png"), pixels)
    _write_image(str(tmp_path / "output" / "sunset.png"), pixels)
    ids = tmp_path / "ids.txt"
    ids.write_text("a0001\nsunset\n")

    loader = data.Adobe5kDataLoader(str(tmp_path), str(ids))
    assert len(loader.load_data()) == 2


def test_missing_target_is_reported_but_optional(tmp_path):
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    _write_image(str(tmp_path / "input" / "sunset.png"), pixels)
    ids = tmp_path / "ids.txt"
    ids.write_text("sunset\n")

    with pytest.raises(FileNotFoundError):
        data.Adobe5kDataLoader(str(tmp_path), str(ids)).load_data()

    loaded = data.Adobe5kDataLoader(str(tmp_path), str(ids)).load_data(
        require_output=False)
    assert loaded[0]['output_img'] is None

    dataset = data.Dataset(data_dict=loaded, normaliser=255, is_inference=True)
    sample = dataset[0]
    assert 'output_img' not in sample
    assert sample['input_img'].shape == (3, 8, 8)


def test_no_matching_ids_raises(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("nothing_here\n")
    with pytest.raises(FileNotFoundError):
        data.Adobe5kDataLoader(str(tmp_path), str(ids)).load_data()


@pytest.mark.parametrize("mode,shape", [("L", (8, 8)), ("RGBA", (8, 8, 4))])
def test_greyscale_and_rgba_images_load_as_three_channels(tmp_path, mode, shape):
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, shape, dtype=np.uint8)
    path = str(tmp_path / ("img_%s.png" % mode))
    Image.fromarray(pixels, mode=mode).save(path)
    img = util.ImageProcessing.load_image(path, normaliser=255)
    assert img.shape == (8, 8, 3)


# ------------------------------------------------------------------ end to end

def test_forward_pass_runs_on_cpu():
    torch.manual_seed(0)
    net = model.DeepLPFNet()
    net.eval()
    img = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        out = net(img)
    assert out.shape == img.shape
    assert torch.isfinite(out).all()
    assert out.min() >= 0 and out.max() <= 1


def test_backward_pass_produces_finite_gradients():
    torch.manual_seed(0)
    net = model.DeepLPFNet()
    criterion = model.DeepLPFLoss(ssim_window_size=5)
    img = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    loss = criterion(torch.clamp(net(img), 0, 1), target)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_get_device_returns_a_device():
    assert isinstance(util.get_device(), torch.device)


# --------------------------------------------------- optional capabilities

def test_defaults_leave_the_architecture_untouched():
    """Both optional features off must give the released architecture."""
    torch.manual_seed(0)
    plain = model.DeepLPFNet()
    torch.manual_seed(0)
    explicit = model.DeepLPFNet(learn_filter_count=False, colour_knots=None)
    assert list(plain.state_dict()) == list(explicit.state_dict())
    assert not hasattr(plain.deeplpfnet, 'colour_head')
    # 24 predicted parameters per branch, no gates.
    assert plain.deeplpfnet.graduated_filter.fc_graduated.out_features == 24
    assert plain.deeplpfnet.elliptical_filter.fc_elliptical.out_features == 24
    for a, b in zip(plain.state_dict().values(), explicit.state_dict().values()):
        assert torch.equal(a, b)


def test_gates_add_three_outputs_per_branch_and_a_penalty():
    torch.manual_seed(0)
    net = model.DeepLPFNet(learn_filter_count=True)
    assert net.deeplpfnet.graduated_filter.fc_graduated.out_features == 27
    assert net.deeplpfnet.elliptical_filter.fc_elliptical.out_features == 27
    net.eval()
    with torch.no_grad():
        net(torch.rand(1, 3, 48, 48))
    assert 0.0 <= net.gate_penalty.item() <= 1.0


def test_gate_penalty_is_differentiable():
    torch.manual_seed(0)
    net = model.DeepLPFNet(learn_filter_count=True)
    net(torch.rand(1, 3, 48, 48))
    net.gate_penalty.backward()
    grad = net.deeplpfnet.graduated_filter.fc_graduated.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_a_zero_gate_removes_its_instance_from_the_product():
    from filtercommon import _apply_gates
    torch.manual_seed(0)
    mask = torch.rand(1, 3, 3, 4, 4) * 2
    gates = torch.tensor([[0.0, 1.0, 0.0]])
    gated = _apply_gates(mask, gates)
    assert torch.equal(gated[:, 0], torch.ones_like(mask[:, 0]))
    assert torch.equal(gated[:, 2], torch.ones_like(mask[:, 2]))
    assert torch.equal(gated[:, 1], mask[:, 1])


def test_colour_head_is_the_identity_at_initialisation():
    torch.manual_seed(0)
    head = model.ColourHead(64, knots=16)
    img = torch.rand(1, 3, 16, 16)
    context = torch.rand(1, 64, 16, 16)
    assert torch.allclose(head(context, img), img, atol=1e-6)


def test_colour_head_off_gives_the_same_weights_as_on_at_the_same_seed():
    """ColourHead is constructed last, so it must not disturb the RNG stream."""
    torch.manual_seed(0)
    without = model.DeepLPFNet()
    torch.manual_seed(0)
    with_head = model.DeepLPFNet(colour_knots=16)
    shared = without.state_dict()
    other = with_head.state_dict()
    for key in shared:
        assert torch.equal(shared[key], other[key]), key


def test_colour_head_network_starts_from_the_unmodified_output():
    torch.manual_seed(0)
    without = model.DeepLPFNet()
    torch.manual_seed(0)
    with_head = model.DeepLPFNet(colour_knots=16)
    without.eval()
    with_head.eval()
    img = torch.rand(1, 3, 48, 48)
    with torch.no_grad():
        assert torch.allclose(without(img), with_head(img), atol=1e-6)


def test_colour_head_can_mix_channels_once_trained():
    """The head must be able to express what the diagonal filters cannot."""
    torch.manual_seed(0)
    head = model.ColourHead(64, knots=0)
    # Swap R and B through the mixer: dM = swap - I.
    swap = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    with torch.no_grad():
        head.fc.bias[0:9] = (swap - torch.eye(3)).reshape(-1)
    img = torch.rand(1, 3, 8, 8)
    out = head(torch.zeros(1, 64, 8, 8), img)
    assert torch.allclose(out[:, 0], img[:, 2], atol=1e-6)
    assert torch.allclose(out[:, 2], img[:, 0], atol=1e-6)


# ------------------------------------------------------------- batching

def test_batched_forward_matches_per_image_forward():
    """A batch must give each image the same result as running it alone."""
    torch.manual_seed(0)
    net = model.DeepLPFNet()
    net.eval()
    x = torch.rand(4, 3, 48, 48)
    with torch.no_grad():
        batched = net(x)
        one_at_a_time = torch.cat([net(x[i:i + 1]) for i in range(4)])
    assert torch.allclose(batched, one_at_a_time, atol=1e-6)


def test_graduated_branch_decides_per_image_not_per_batch():
    """Images in one batch may take different sides of the factor >= 1 split."""
    torch.manual_seed(0)
    filt = model.GraduatedFilter()
    factor = torch.tensor([[[[[0.4]]]], [[[[1.6]]]]])       # (2, 1, 1, 1, 1)
    invert = torch.tensor([[[[[1.0]]]], [[[[0.0]]]]])
    d1 = torch.full((2, 1, 1, 1, 1), 0.4)
    d2 = torch.full((2, 1, 1, 1, 1), 0.3)
    top_line = torch.rand(2, 1, 1, 8, 8)
    mask = filt.get_inverted_mask(factor, invert, d1, d2, 2, top_line)
    # factor < 1 clamps into [0, 1]; factor >= 1 clamps into [1, 2].
    assert mask[0].max() <= 1.0 + 1e-6
    assert mask[1].min() >= 1.0 - 1e-6


def test_crop_size_gives_uniform_shapes_for_collation(tmp_path):
    rng = np.random.default_rng(0)
    for i, (h, w) in enumerate([(40, 60), (55, 48)]):
        px = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        _write_image(str(tmp_path / "input" / ("a%04d-x.png" % i)), px)
        _write_image(str(tmp_path / "output" / ("a%04d-x.png" % i)), px)
    ids = tmp_path / "ids.txt"
    ids.write_text("a0000\na0001\n")
    loaded = data.Adobe5kDataLoader(str(tmp_path), str(ids)).load_data()

    uncropped = data.Dataset(data_dict=loaded, normaliser=255, is_valid=False)
    assert uncropped[0]['input_img'].shape != uncropped[1]['input_img'].shape

    cropped = data.Dataset(data_dict=loaded, normaliser=255, is_valid=False,
                           crop_size=32)
    batch = torch.utils.data.DataLoader(cropped, batch_size=2)
    sample = next(iter(batch))
    assert sample['input_img'].shape == (2, 3, 32, 32)
    assert sample['output_img'].shape == (2, 3, 32, 32)


def test_crop_is_the_same_window_for_input_and_target(tmp_path):
    rng = np.random.default_rng(1)
    px = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
    _write_image(str(tmp_path / "input" / "a0000-x.png"), px)
    _write_image(str(tmp_path / "output" / "a0000-x.png"), px)
    ids = tmp_path / "ids.txt"
    ids.write_text("a0000\n")
    loaded = data.Adobe5kDataLoader(str(tmp_path), str(ids)).load_data()
    dataset = data.Dataset(data_dict=loaded, normaliser=255, is_valid=False,
                           crop_size=16)
    for _ in range(5):
        sample = dataset[0]
        # input and target are the same image here, so an identical crop of
        # both must produce identical tensors.
        assert torch.equal(sample['input_img'], sample['output_img'])


# ------------------------------------------------------- the training step

class _EagerCapture(trainstep.GraphedStep):
    def _capture(self, static_x, static_y):
        for _ in range(self.warmup):
            static_loss = trainstep.EagerStep.__call__(self, static_x, static_y).clone()

        def replay():
            # The warm-up consumed dropout draws; re-seed so the replayed
            # step draws the same mask as the eager step it is compared with.
            torch.manual_seed(self.seed)
            static_loss.copy_(trainstep.EagerStep.__call__(self, static_x, static_y))
        return static_loss, replay


def _build(seed):
    torch.manual_seed(seed)
    net = model.DeepLPFNet().train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    return net, model.DeepLPFLoss(ssim_window_size=5), opt


def test_graphed_step_matches_eager_step_and_warmup_does_not_advance_training():
    torch.manual_seed(0)
    batches = [(torch.rand(1, 3, h, w), torch.rand(1, 3, h, w))
               for h, w in ((40, 48), (48, 40), (40, 48), (48, 40), (40, 48))]

    net_e, crit_e, opt_e = _build(0)
    eager = trainstep.EagerStep(net_e, crit_e, opt_e)
    net_g, crit_g, opt_g = _build(0)
    graphed = _EagerCapture(net_g, crit_g, opt_g, warmup=3)

    for i, (x, y) in enumerate(batches):
        torch.manual_seed(100 + i)  # dropout draw
        loss_e = eager(x, y)
        graphed.seed = 100 + i
        loss_g = graphed(x, y)
        assert torch.equal(loss_e, loss_g), (i, loss_e, loss_g)

    assert len(graphed.graphs) == 2
    for pe, pg in zip(net_e.parameters(), net_g.parameters()):
        assert torch.equal(pe, pg)
    for pe, pg in zip(net_e.parameters(), net_g.parameters()):
        if pe in opt_e.state:
            assert torch.equal(opt_e.state[pe]['step'], opt_g.state[pg]['step'])
            assert torch.equal(opt_e.state[pe]['exp_avg_sq'], opt_g.state[pg]['exp_avg_sq'])


def test_eager_step_updates_the_parameters_and_returns_a_detached_loss():
    net, crit, opt = _build(0)
    step = trainstep.EagerStep(net, crit, opt)
    before = [p.detach().clone() for p in net.parameters()]
    loss = step(torch.rand(1, 3, 40, 40), torch.rand(1, 3, 40, 40))
    assert not loss.requires_grad
    assert any(not torch.equal(b, p) for b, p in zip(before, net.parameters()))


def test_gate_weight_adds_the_penalty_to_the_loss():
    torch.manual_seed(0)
    net = model.DeepLPFNet(learn_filter_count=True).train()
    opt = torch.optim.Adam(net.parameters(), lr=0.0)  # lr 0: compare on one input
    crit = model.DeepLPFLoss(ssim_window_size=5)
    x, y = torch.rand(1, 3, 40, 40), torch.rand(1, 3, 40, 40)

    torch.manual_seed(7)
    plain = trainstep.EagerStep(net, crit, opt)(x, y)
    torch.manual_seed(7)
    gated = trainstep.EagerStep(net, crit, opt, gate_weight=1.0)(x, y)
    assert gated.item() > plain.item()
    assert gated.item() == pytest.approx(plain.item() + net.gate_penalty.item(), rel=1e-5)


def test_cuda_graphs_are_rejected_without_cuda():
    """--cuda_graphs on a CPU-only machine must fail loudly, not silently."""
    import train
    assert "raise SystemExit('--cuda_graphs needs a CUDA device')" in \
        open(train.__file__).read()


# ------------------------------------------------------- the deeplpf command

def test_gather_images_expands_directories_and_skips_non_images(tmp_path):
    import deeplpf_cli
    rng = np.random.default_rng(0)
    px = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
    for name in ('b.png', 'a.jpg'):
        Image.fromarray(px).save(str(tmp_path / name))
    (tmp_path / 'notes.txt').write_text('not an image')
    found = deeplpf_cli.gather_images([str(tmp_path)])
    assert [os.path.basename(p) for p in found] == ['a.jpg', 'b.png']


def test_enhance_writes_one_output_per_input(tmp_path):
    import deeplpf_cli
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(
        str(tmp_path / 'photo.png'))

    torch.manual_seed(0)
    net = model.DeepLPFNet()
    checkpoint = tmp_path / 'model.pt'
    torch.save(net.state_dict(), str(checkpoint))

    out_dir = tmp_path / 'out'
    status = deeplpf_cli.main(['enhance', str(tmp_path / 'photo.png'),
                               '--out', str(out_dir),
                               '--checkpoint', str(checkpoint),
                               '--device', 'cpu'])
    assert status == 0
    written = out_dir / 'photo_enhanced.png'
    assert written.is_file()
    assert np.array(Image.open(str(written))).shape == (48, 48, 3)


def test_enhance_reports_an_undersized_image_and_keeps_going(tmp_path):
    import deeplpf_cli
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(
        str(tmp_path / 'tiny.png'))
    Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(
        str(tmp_path / 'ok.png'))

    torch.manual_seed(0)
    checkpoint = tmp_path / 'model.pt'
    torch.save(model.DeepLPFNet().state_dict(), str(checkpoint))

    out_dir = tmp_path / 'out'
    status = deeplpf_cli.main(['enhance', str(tmp_path),
                               '--out', str(out_dir),
                               '--checkpoint', str(checkpoint),
                               '--device', 'cpu'])
    # One failure out of two is not a failed run.
    assert status == 0
    assert (out_dir / 'ok_enhanced.png').is_file()
    assert not (out_dir / 'tiny_enhanced.png').exists()


def test_enhance_accepts_a_greyscale_photograph(tmp_path):
    import deeplpf_cli
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (48, 48), dtype=np.uint8), mode='L').save(
        str(tmp_path / 'grey.png'))
    torch.manual_seed(0)
    checkpoint = tmp_path / 'model.pt'
    torch.save(model.DeepLPFNet().state_dict(), str(checkpoint))
    out_dir = tmp_path / 'out'
    assert deeplpf_cli.main(['enhance', str(tmp_path / 'grey.png'),
                             '--out', str(out_dir),
                             '--checkpoint', str(checkpoint),
                             '--device', 'cpu']) == 0
    assert (out_dir / 'grey_enhanced.png').is_file()


def test_released_checkpoint_loads_into_the_current_model():
    """The weights shipped in this repository must still load strictly."""
    import glob
    matches = sorted(glob.glob(os.path.join(os.path.dirname(model.__file__),
                                            'pretrained_models', '*', '*.pt')))
    if not matches:
        pytest.skip('no bundled checkpoint in this checkout')
    net = model.DeepLPFNet()
    net.load_state_dict(torch.load(matches[0], map_location='cpu'), strict=True)


SEEDS = range(8)


def _graduated_row_gradients(seed):
    """Per-output-row gradient magnitude of ``fc_graduated.weight``, (24,)."""
    torch.manual_seed(seed)
    net = model.DeepLPFNet()
    net.train()
    prediction = net(torch.rand(1, 3, 48, 48))
    target = torch.rand_like(prediction)
    model.DeepLPFLoss()(torch.clamp(prediction, 0, 1), target).backward()
    weight = dict(net.named_parameters())[
        'deeplpfnet.graduated_filter.fc_graduated.weight']
    return weight.grad.abs().sum(dim=1)


SEEDS = range(8)


def test_no_graduated_output_is_structurally_starved():
    """No output of ``fc_graduated`` may be cut off from the loss by construction.

    There are two ways a row of this layer can show a zero gradient. One is
    ordinary saturation: the scale factors ([15:24]) pass through a clamp, and
    a value already outside its bounds passes nothing back on that step. That
    is the clamp doing its job, and it clears on the next batch.

    The other is structural. ``G[:, 6:9]`` used to be consumed only as
    ``y_axis_dist.data`` - a detached lower bound on the next line - so it set
    the forward value while receiving no gradient on any step, in any seed,
    ever. In the shipped checkpoint those three units are still at their
    initialisation, tanh01(0) = 0.5, and that untrained constant beat the
    trained offset on two of the three filter instances for every bundled
    example. Removing ``.data`` alone does not fix it: ``maximum`` routes the
    gradient to whichever argument it selected, so the pair take turns and
    each is still starved whenever the other wins.

    So the geometry outputs - the inversion indicator, the slope, the intercept
    and the two offsets, ``G[:, 0:15]`` - must carry gradient on *every* seed,
    and no output at all may be starved on every seed.
    """
    gradients = torch.stack([_graduated_row_gradients(s) for s in SEEDS])

    always_starved = [row for row in range(gradients.shape[1])
                      if not gradients[:, row].any()]
    assert always_starved == [], (
        'fc_graduated outputs receiving no gradient in any of %d seeds - a '
        'detached or unused path, not clamp saturation: %s'
        % (len(SEEDS), always_starved))

    geometry = gradients[:, 0:15]
    starved_somewhere = (geometry == 0).nonzero().tolist()
    assert starved_somewhere == [], (
        'line-geometry outputs G[:, 0:15] must receive gradient on every '
        'step; (seed index, output) pairs that did not: %s' % starved_somewhere)
