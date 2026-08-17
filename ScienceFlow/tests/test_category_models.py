# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

"""
End-to-end forward passes aligned with scienceflow coldstart
`models_guidance_classified.json`.

Run after ``uv sync``: ``uv run pytest tests/test_category_models.py -m pretrained -v``.
Skip in CI without GPU/cache: ``pytest -m "not pretrained"``.
CPU dev run: ``SCIENCEFLOW_ALLOW_PRETRAINED_CPU=1 pytest tests/test_category_models.py -m pretrained -v``.
DINOv3 standalone smoke: ``scripts/dev/smoke_dinov3_e2e.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch")

_MODELS_DIR = "./data/models/mlebench"
_DINOV3_HUBCONF = Path(_MODELS_DIR) / "dinov3-main" / "hubconf.py"
_DINOV3_WEIGHTS_FILE = Path(_MODELS_DIR) / "checkpoints" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
_ENV_UNSET = object()


@pytest.fixture(autouse=True)
def _restore_cuda_visible_devices():
    """Keep third-party model setup from leaking GPU visibility across tests."""
    original = os.environ.get("CUDA_VISIBLE_DEVICES", _ENV_UNSET)
    yield
    if original is _ENV_UNSET:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = original


def _dinov3_local_files_exist() -> bool:
    return _DINOV3_HUBCONF.is_file() and _DINOV3_WEIGHTS_FILE.is_file()


def _setup_local_model_env() -> None:
    """Set env vars for local model cache if directory exists."""
    root = Path(_MODELS_DIR)
    if not root.is_dir():
        return
    os.environ.setdefault("HF_HUB_CACHE", str(root))
    os.environ.setdefault("TORCH_HOME", str(root))
    hub = root / "dinov3-main"
    if hub.is_dir():
        os.environ.setdefault("DINOV3_HUB_DIR", str(hub))
    ckpt = root / "checkpoints" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    if ckpt.is_file():
        os.environ.setdefault("DINOV3_WEIGHTS", str(ckpt))


_setup_local_model_env()


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _assert_cuda_for_skill_tests() -> None:
    allow_cpu = os.environ.get("SCIENCEFLOW_ALLOW_PRETRAINED_CPU", "").strip().lower() in {"1", "true", "yes"}
    if not torch.cuda.is_available() and not allow_cpu:
        pytest.skip(
            "pretrained category tests expect CUDA (see plan / H100 env); "
            "set SCIENCEFLOW_ALLOW_PRETRAINED_CPU=1 to run on CPU",
        )


# --- Image classification (timm + HF) ---


@pytest.mark.pretrained
def test_efficientnetv2_l_timm_pipeline():
    _assert_cuda_for_skill_tests()
    timm = pytest.importorskip("timm")
    dev = device()
    model = timm.create_model(
        "tf_efficientnetv2_l.in21k_ft_in1k",
        pretrained=True,
        num_classes=0,
    ).to(dev)
    data_config = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**data_config, is_training=False)
    x = torch.randn(2, 3, 480, 480, device=dev)
    with torch.no_grad():
        feat = model(x)
    assert feat.shape == (2, model.num_features)
    head = nn.Linear(model.num_features, 10).to(dev)
    logits = head(feat)
    assert logits.shape == (2, 10)


@pytest.mark.pretrained
def test_eva02_large_timm_pipeline():
    _assert_cuda_for_skill_tests()
    timm = pytest.importorskip("timm")
    dev = device()
    backbone = timm.create_model(
        "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        pretrained=True,
        num_classes=0,
    ).to(dev)
    x = torch.randn(2, 3, 448, 448, device=dev)
    with torch.no_grad():
        feat = backbone(x)
    assert feat.shape == (2, 1024)


@pytest.mark.pretrained
def test_siglip2_vision_pipeline():
    _assert_cuda_for_skill_tests()
    pytest.importorskip("transformers")
    from transformers import AutoModel

    dev = device()
    full = AutoModel.from_pretrained("google/siglip2-so400m-patch16-256")
    vision = full.vision_model.to(dev).eval()
    del full
    x = torch.randn(2, 3, 256, 256, device=dev)
    with torch.no_grad():
        out = vision(pixel_values=x)
    pooled = out.pooler_output
    assert pooled.shape == (2, vision.config.hidden_size)


@pytest.mark.pretrained
@pytest.mark.skipif(
    not _dinov3_local_files_exist(),
    reason=f"DINOv3 local repo or weights not found under {_MODELS_DIR}",
)
def test_dinov3_vitl16_classification_features():
    _assert_cuda_for_skill_tests()
    pytest.importorskip("torchmetrics")
    dev = device()
    hub = os.environ["DINOV3_HUB_DIR"]
    w = os.environ["DINOV3_WEIGHTS"]
    backbone = torch.hub.load(hub, "dinov3_vitl16", source="local", weights=w).to(dev).eval()
    x = torch.randn(2, 3, 224, 224, device=dev)
    with torch.no_grad():
        out = backbone.forward_features(x)
    assert "x_norm_clstoken" in out
    assert out["x_norm_clstoken"].shape == (2, 1024)


@pytest.mark.pretrained
@pytest.mark.skipif(
    not _dinov3_local_files_exist(),
    reason=f"DINOv3 local repo or weights not found under {_MODELS_DIR}",
)
def test_dinov3_vitl16_linear_probe_e2e():
    """Frozen backbone + linear head + CE + backward (minimal competition-style path)."""
    _assert_cuda_for_skill_tests()
    pytest.importorskip("torchmetrics")
    import torch.nn.functional as F

    dev = device()
    hub = os.environ["DINOV3_HUB_DIR"]
    w = os.environ["DINOV3_WEIGHTS"]
    backbone = torch.hub.load(hub, "dinov3_vitl16", source="local", weights=w).to(dev)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    num_classes = 5
    head = nn.Linear(1024, num_classes).to(dev)
    x = torch.randn(2, 3, 224, 224, device=dev)
    with torch.no_grad():
        cls_feat = backbone.forward_features(x)["x_norm_clstoken"]
        logits_infer = head(cls_feat)
    assert logits_infer.shape == (2, num_classes)

    head.train()
    x2 = torch.randn(2, 3, 224, 224, device=dev)
    logits = head(backbone.forward_features(x2)["x_norm_clstoken"])
    y = torch.tensor([0, 3], device=dev)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    assert head.weight.grad is not None
    assert torch.isfinite(loss).item()


# --- Detection 2D ---


@pytest.mark.pretrained
def test_yolov8_load_and_predict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _assert_cuda_for_skill_tests()
    pytest.importorskip("ultralytics")
    monkeypatch.chdir(tmp_path)
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")  # smaller than x for test speed; cwd=tmp_path avoids repo root
    # ultralytics API: predict on numpy image
    import numpy as np

    img = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
    _dev = 0 if torch.cuda.is_available() else "cpu"
    results = model.predict(img, verbose=False, device=_dev)
    assert len(results) >= 1


@pytest.mark.pretrained
def test_faster_rcnn_torchvision_forward():
    _assert_cuda_for_skill_tests()
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

    dev = device()
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(dev)
    model.train()
    images = [torch.rand(3, 256, 256, device=dev)]
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]], device=dev),
            "labels": torch.tensor([1], dtype=torch.int64, device=dev),
        }
    ]
    loss_dict = model(images, targets)
    assert "loss_classifier" in loss_dict
    assert sum(loss_dict.values()).item() == sum(loss_dict.values()).item()


@pytest.mark.pretrained
def test_detr_transformers_forward():
    _assert_cuda_for_skill_tests()
    pytest.importorskip("transformers")
    from transformers import DetrForObjectDetection, DetrImageProcessor
    from PIL import Image

    dev = device()
    num_labels = 3
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    ).to(dev)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    image = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype(np.uint8))
    encoding = processor(images=image, return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(dev)
    pixel_mask = encoding["pixel_mask"].to(dev)
    labels = [
        {
            "class_labels": torch.tensor([1], dtype=torch.int64, device=dev),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=dev),
        }
    ]
    model.train()
    out = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
    assert out.loss is not None


# --- Detection 3D BEV ---


@pytest.mark.pretrained
def test_bev_efficientnet_features_only():
    _assert_cuda_for_skill_tests()
    timm = pytest.importorskip("timm")
    dev = device()
    backbone = timm.create_model(
        "tf_efficientnetv2_l.in21k_ft_in1k",
        pretrained=True,
        features_only=True,
        out_indices=(2, 3, 4),
        in_chans=4,
    ).to(dev)
    x = torch.randn(1, 4, 128, 128, device=dev)
    with torch.no_grad():
        feats = backbone(x)
    assert isinstance(feats, list) and len(feats) == 3


# --- Segmentation ---


@pytest.mark.pretrained
def test_smp_unet_forward():
    _assert_cuda_for_skill_tests()
    smp = pytest.importorskip("segmentation_models_pytorch")
    dev = device()
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(dev)
    x = torch.randn(2, 3, 256, 256, device=dev)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256, 256)


# --- NLP ---


@pytest.mark.pretrained
def test_modernbert_cls_pipeline():
    _assert_cuda_for_skill_tests()
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer, ModernBertModel

    dev = device()
    mid = "answerdotai/ModernBERT-large"
    tok = AutoTokenizer.from_pretrained(mid)
    backbone = ModernBertModel.from_pretrained(mid).to(dev).eval()
    enc = tok(["hello world", "test"], padding=True, truncation=True, max_length=32, return_tensors="pt")
    enc = {k: v.to(dev) for k, v in enc.items()}
    head = nn.Linear(backbone.config.hidden_size, 2).to(dev)
    with torch.no_grad():
        h = backbone(**enc).last_hidden_state[:, 0, :]
    logits = head(h)
    assert logits.shape == (2, 2)


@pytest.mark.pretrained
def test_deberta_v3_cls_pipeline():
    _assert_cuda_for_skill_tests()
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer, AutoModel

    dev = device()
    try:
        tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        model = AutoModel.from_pretrained("microsoft/deberta-v3-large").to(dev).eval()
    except (OSError, ValueError, ImportError) as exc:
        pytest.skip(f"deberta v3 load skipped (tokenizer/model cache or deps): {exc}")
    enc = tok(["hello"], padding=True, truncation=True, max_length=32, return_tensors="pt")
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        h = model(**enc).last_hidden_state[:, 0, :]
    assert h.shape[-1] == model.config.hidden_size


# --- Audio ---


@pytest.mark.pretrained
def test_ast_pipeline():
    _assert_cuda_for_skill_tests()
    pytest.importorskip("transformers")
    pytest.importorskip("librosa")
    from transformers import ASTModel, ASTFeatureExtractor

    dev = device()
    fe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(dev).eval()
    wav = np.random.randn(16000 * 2).astype(np.float32)
    inputs = fe(wav, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    assert out.pooler_output.shape[-1] == 768


@pytest.mark.pretrained
def test_muq_large_pipeline():
    _assert_cuda_for_skill_tests()
    pytest.importorskip("librosa")
    muq_mod = pytest.importorskip("muq")
    MuQ = muq_mod.MuQ

    dev = device()
    backbone = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(dev).eval()
    wav = np.random.randn(24000 * 1).astype(np.float32)
    wavs = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = backbone(wavs, output_hidden_states=True)
    pooled = out.last_hidden_state.mean(dim=1)
    assert pooled.shape[-1] == 1024


@pytest.mark.pretrained
def test_efficientnet_melspec_dummy_image():
    _assert_cuda_for_skill_tests()
    timm = pytest.importorskip("timm")
    dev = device()
    backbone = timm.create_model("efficientnet_b4.ra2_in1k", pretrained=True, num_classes=0).to(dev).eval()
    # 3-channel "mel" tensor resized conceptually to 380
    x = torch.randn(2, 3, 380, 380, device=dev)
    with torch.no_grad():
        feat = backbone(x)
    assert feat.shape[1] == backbone.num_features


# --- Tabular ---


@pytest.mark.pretrained
def test_lgb_xgb_catboost_oof_smoke(tmp_path: Path) -> None:
    _assert_cuda_for_skill_tests()
    pytest.importorskip("catboost")
    import lightgbm as lgb
    import pandas as pd
    import xgboost as xgb
    from catboost import CatBoostClassifier

    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(120, 6))
    y = pd.Series((X.iloc[:, 0] > 0).astype(int))
    X_train, X_val = X.iloc[:100], X.iloc[100:]
    y_train, y_val = y.iloc[:100], y.iloc[100:]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    lgbm = lgb.train(
        {"objective": "binary", "metric": "auc", "verbosity": -1},
        lgb_train,
        num_boost_round=64,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(12)],
    )
    p_lgb = lgbm.predict(X_val)

    dtr = xgb.DMatrix(X_train, y_train)
    dva = xgb.DMatrix(X_val, y_val)
    xgbm = xgb.train(
        {"objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0},
        dtr,
        num_boost_round=64,
        evals=[(dva, "val")],
        early_stopping_rounds=12,
    )
    p_xgb = xgbm.predict(dva)

    cat = CatBoostClassifier(
        iterations=64,
        depth=4,
        verbose=0,
        eval_metric="AUC",
        early_stopping_rounds=12,
        train_dir=str(tmp_path / "catboost_info"),
    )
    cat.fit(X_train, y_train, eval_set=(X_val, y_val))
    p_cat = cat.predict_proba(X_val)[:, 1]

    oof_ensemble = 0.4 * p_lgb + 0.3 * p_xgb + 0.3 * p_cat
    assert oof_ensemble.shape == (20,)
    assert np.isfinite(oof_ensemble).all()


# --- Signal processing ---


@pytest.mark.pretrained
def test_efficientnet_spectrogram_dummy():
    _assert_cuda_for_skill_tests()
    timm = pytest.importorskip("timm")
    dev = device()
    backbone = timm.create_model("efficientnet_b4.ra2_in1k", pretrained=True, num_classes=0).to(dev).eval()
    x = torch.randn(2, 3, 224, 224, device=dev)
    with torch.no_grad():
        f = backbone(x)
    assert f.shape[0] == 2


@pytest.mark.pretrained
def test_lgb_signal_features_smoke():
    _assert_cuda_for_skill_tests()
    from scipy import stats
    from scipy.signal import welch
    import lightgbm as lgb

    def extract_signal_features(signal: np.ndarray) -> dict:
        f = {}
        f["mean"] = float(np.mean(signal))
        f["std"] = float(np.std(signal))
        freqs, psd = welch(signal, fs=1.0, nperseg=min(256, len(signal)))
        f["spectral_centroid"] = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-8))
        f["skew"] = float(stats.skew(signal))
        return f

    X = np.array([list(extract_signal_features(np.random.randn(512)).values()) for _ in range(64)])
    y = (X[:, 0] > 0).astype(np.int32)
    train = lgb.Dataset(X, y)
    m = lgb.train({"objective": "binary", "verbosity": -1}, train, num_boost_round=10)
    assert m.predict(X[:2]).shape == (2,)


# --- Multimodal ---


@pytest.mark.pretrained
def test_vision_audio_fusion_mlp():
    _assert_cuda_for_skill_tests()
    timm = pytest.importorskip("timm")
    pytest.importorskip("transformers")
    from transformers import ASTModel, ASTFeatureExtractor

    dev = device()
    vision = timm.create_model("tf_efficientnetv2_l.in21k_ft_in1k", pretrained=True, num_classes=0).to(dev).eval()
    ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(dev).eval()
    fe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    img = torch.randn(1, 3, 224, 224, device=dev)
    wav = np.random.randn(16000).astype(np.float32)
    aud_in = fe(wav, sampling_rate=16000, return_tensors="pt")
    aud_in = {k: v.to(dev) for k, v in aud_in.items()}
    with torch.no_grad():
        vf = vision(img)
        af = ast(**aud_in).pooler_output
    fusion = nn.Linear(vf.shape[-1] + af.shape[-1], 5).to(dev)
    out = fusion(torch.cat([vf, af], dim=-1))
    assert out.shape == (1, 5)
