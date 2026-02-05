# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .models.models_r3m import R3M

import os
from os.path import expanduser
import omegaconf
import hydra
import gdown
import torch
import copy

VALID_ARGS = [
    "_target_",
    "device",
    "lr",
    "hidden_dim",
    "size",
    "l2weight",
    "l1weight",
    "langweight",
    "tcnweight",
    "l2dist",
    "bs",
]
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def cleanup_config(cfg):
    config = copy.deepcopy(cfg)
    keys = config.agent.keys()
    for key in list(keys):
        if key not in VALID_ARGS:
            del config.agent[key]
    config.agent["_target_"] = "models.encoder.r3m.R3M"
    config["device"] = device

    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    config.agent["langweight"] = 0
    return config.agent


def remove_language_head(state_dict):
    keys = state_dict.keys()
    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    for key in list(keys):
        if ("lang_enc" in key) or ("lang_rew" in key):
            del state_dict[key]
    return state_dict


def load_r3m(modelid):
    home = os.path.join(expanduser("~"), ".model_checkpoints", "r3m")
    if modelid == "resnet50":
        foldername = "r3m_50"
        modelurl = "https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA"
        configurl = "https://drive.google.com/uc?id=10jY2VxrrhfOdNPmsFdES568hjjIoBJx8"
    elif modelid == "resnet34":
        foldername = "r3m_34"
        modelurl = "https://drive.google.com/uc?id=15bXD3QRhspIRacOKyWPw5y2HpoWUCEnE"
        configurl = "https://drive.google.com/uc?id=1RY0NS-Tl4G7M1Ik_lOym0b5VIBxX9dqW"
    elif modelid == "resnet18":
        foldername = "r3m_18"
        modelurl = "https://drive.google.com/uc?id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-"
        configurl = "https://drive.google.com/uc?id=1nitbHQ-GRorxc7vMUiEHjHWP5N11Jvc6"
    else:
        raise NameError("Invalid Model ID")

    if not os.path.exists(os.path.join(home, foldername)):
        os.makedirs(os.path.join(home, foldername))
    modelpath = os.path.join(home, foldername, "model.pt")
    configpath = os.path.join(home, foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep)
    r3m_state_dict = remove_language_head(
        torch.load(modelpath, map_location=torch.device(device))["r3m"]
    )
    rep.load_state_dict(r3m_state_dict)
    rep = rep.module
    return rep

def load_r3m_reproduce(modelid):
    home = os.path.join(expanduser("~"), ".r3m")
    if modelid == "r3m":
        foldername = "original_r3m"
        modelurl = "https://drive.google.com/uc?id=1jLb1yldIMfAcGVwYojSQmMpmRM7vqjp9"
        configurl = "https://drive.google.com/uc?id=1cu-Pb33qcfAieRIUptNlG1AQIMZlAI-q"
    elif modelid == "r3m_noaug":
        foldername = "original_r3m_noaug"
        modelurl = "https://drive.google.com/uc?id=1k_ZlVtvlktoYLtBcfD0aVFnrZcyCNS9D"
        configurl = "https://drive.google.com/uc?id=1hPmJwDiWPkd6GGez6ywSC7UOTIX7NgeS"
    elif modelid == "r3m_nol1":
        foldername = "original_r3m_nol1"
        modelurl = "https://drive.google.com/uc?id=1LpW3aBMdjoXsjYlkaDnvwx7q22myM_nB"
        configurl = "https://drive.google.com/uc?id=1rZUBrYJZvlF1ReFwRidZsH7-xe7csvab"
    elif modelid == "r3m_nolang":
        foldername = "original_r3m_nolang"
        modelurl = "https://drive.google.com/uc?id=1FXcniRei2JDaGMJJ_KlVxHaLy0Fs_caV"
        configurl = "https://drive.google.com/uc?id=192G4UkcNJO4EKN46ECujMcH0AQVhnyQe"
    else:
        raise NameError("Invalid Model ID")

    if not os.path.exists(os.path.join(home, foldername)):
        os.makedirs(os.path.join(home, foldername))
    modelpath = os.path.join(home, foldername, "model.pt")
    configpath = os.path.join(home, foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep)
    r3m_state_dict = remove_language_head(
        torch.load(modelpath, map_location=torch.device(device))["r3m"]
    )

    rep.load_state_dict(r3m_state_dict)
    rep = rep.module
    return rep
