import torch
import torch.nn as nn
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

# train_cfg.encoder:  {'_target_': 'models.dino.DinoV2Encoder', 'name': 'dinov2_vits14', 'feature_key': 'x_norm_patchtokens'}
class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        # import pdb;pdb.set_trace()
        print("[DinoV2Encoder] name: ", name)
        assert name == "dinov2_vits14"
        print("[DinoV2Encoder] feature_key: ", feature_key)
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
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

    def forward(self, x):
        # import pdb;pdb.set_trace()
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb

