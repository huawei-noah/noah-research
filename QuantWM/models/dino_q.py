import torch
import torch.nn as nn
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
from .ptq import QAct, QLinear, QConv2d

# train_cfg.encoder:  {'_target_': 'models.dino.DinoV2Encoder', 'name': 'dinov2_vits14', 'feature_key': 'x_norm_patchtokens'}
class QDinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key, 
                 quant=False,
                 calibrate=False,
                 cfg=None):
        super().__init__()
        self.name = name
        # import pdb;pdb.set_trace()
        print("[QDinoV2Encoder] name: ", name)
        assert name == "dinov2_vits14"
        print("[QDinoV2Encoder] feature_key: ", feature_key)
        from .dinov2.hub.backbones import dinov2_vits14
        print("[QDinoV2Encoder] dinov2_vits14: ", dinov2_vits14)
        # self.base_model = torch.hub.load("facebookresearch/dinov2", name)

        self.base_model = dinov2_vits14(quant=quant, calibrate=calibrate, cfg=cfg)

        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size
        # import pdb;pdb.set_trace()

    def model_quant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct]:
                m.quant = True

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct]:
                m.calibrate = False
                       
    def forward(self, x):
        # import pdb;pdb.set_trace()
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb

