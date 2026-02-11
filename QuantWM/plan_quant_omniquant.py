import os
from contextlib import nullcontext
import re
import json
from collections import OrderedDict
import gym
import json
import hydra
import random
import torch
import pickle
import wandb
import logging
import warnings
import numpy as np
import submitit
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed

import copy
import math
from contextlib import nullcontext

from quant_utils.omni_quantize.utils import (
    NativeScalerWithGradNormCount,
    smooth_ln_fcs_inplace,
    set_lwc_state,
    smooth_ln_fcs_temporary,
    set_quant_state,
    let_parameters,
    lwc_parameters,
    get_omni_parameters,
    clear_temp_variable,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]


def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)


def launch_plan_jobs(
    epoch,
    cfg_dicts,
    plan_output_dir,
):
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = (
                f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}"
                f"_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            )
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(folder=subdir_path, slurm_max_num_timeout=20)
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(
                f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}"
            )
        return jobs


def build_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    planner=["gd", "cem"],
    goal_source=["dset"],
    goal_H=[1, 5, 10],
    alpha=[0, 0.1, 1],
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner_name = override_args["planner"]
        planner_cfg = OmegaConf.load(os.path.join(config_path, f"planner/{planner_name}.yaml"))
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)

        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
        cache_input: bool,
        calib_state: bool,

    ):

        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.calib_state = calib_state
        self.device = next(wm.parameters()).device


        if calib_state or cache_input:
            self.eval_seed = [cfg_dict["seed"] * n + 2 for n in range(cfg_dict["n_evals"])]
            self.dump_targets_file = "plan_targets_calib.pkl"
        else:
            self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
            self.dump_targets_file = "plan_targets.pkl"
        print("eval_seed: ", self.eval_seed)

        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]  

        objective_fn = hydra.utils.call(cfg_dict["objective"])

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
        )

        if self.wandb_run is None or isinstance(self.wandb_run, wandb.sdk.lib.disabled.RunDisabled):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
            do_quant=self.cfg_dict["quant"],
            do_quant_encoder=self.cfg_dict["quant_encoder"],
            calib_state=self.calib_state,
            quant_iter=self.cfg_dict["quant_iter"],
        )

        from planning.mpc import MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            self.planner.n_taken_actions = cfg_dict["goal_H"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]

        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []

        if self.goal_source == "random_state":
            observations, states, actions, env_info = self.sample_traj_segment_from_dset(traj_len=2)
            self.env.update_env(env_info)

            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(self.eval_seed)
            if self.env_name == "deformable_env":
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            observations, states, actions, env_info = self.sample_traj_segment_from_dset(
                traj_len=self.frameskip * self.goal_H + 1
            )
            self.env.update_env(env_info)

            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)

            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)

            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {key: np.expand_dims(arr[:, 0], axis=1) for key, arr in rollout_obses.items()}
            self.obs_g = {key: np.expand_dims(arr[:, -1], axis=1) for key, arr in rollout_obses.items()}
            self.state_0 = init_state
            self.state_g = rollout_states[:, -1]
            self.gt_actions = wm_actions

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        for _ in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {key: arr[offset : offset + traj_len] for key, arr in obs.items()}
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def dump_targets(self):
        with open(self.dump_targets_file, "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )
        file_path = os.path.abspath(self.dump_targets_file)
        print(f"Dumped plan targets to {file_path}")

    def perform_planning(self,tag):

        actions_init = self.gt_actions if self.debug_dset_init else None
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename=f"output_final_{tag}"
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (value.item() if isinstance(value, (np.float32, np.int32, np.int64)) else value)
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result

def remap_predictor_keys_for_block_format(state_dict: dict) -> dict:

    new_sd = OrderedDict()


    def strip_prefix(k: str, prefix: str) -> str:
        return k[len(prefix):] if k.startswith(prefix) else k

    for k, v in state_dict.items():
        k2 = k


        had_module = k2.startswith("module.")
        if had_module:
            k2_core = k2[len("module."):]
        else:
            k2_core = k2


        k2_core = re.sub(r"(transformer\.layers\.\d+)\.0\.", r"\1.attn.", k2_core)
        k2_core = re.sub(r"(transformer\.layers\.\d+)\.1\.", r"\1.ff.", k2_core)


        k2 = ("module." + k2_core) if had_module else k2_core

        new_sd[k2] = v

    return new_sd



def load_model(model_ckpt, train_cfg, cfg_dict, num_action_repeat, device):


    def _remap_predictor_sd_old_to_new(old_sd: dict) -> dict:
        """
        将旧 predictor 结构的 key:
          transformer.layers.N.0.xxx  -> transformer.layers.N.attn.xxx
          transformer.layers.N.1.xxx  -> transformer.layers.N.ff.xxx
        """
        new_sd = OrderedDict()
        for k, v in old_sd.items():
            k2 = k
            k2 = re.sub(r"(transformer\.layers\.\d+)\.0\.", r"\1.attn.", k2)
            k2 = re.sub(r"(transformer\.layers\.\d+)\.1\.", r"\1.ff.", k2)
            new_sd[k2] = v
        return new_sd
    result = {}
    print("model_ckpt: ", model_ckpt)
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    print("train_cfg.encoder: ", train_cfg.encoder)
    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
            cfg=cfg_dict["quant_cfg_encoder"],
        )

    if result["encoder"].latent_ndim == 1:
        num_patches = 1
    else:
        decoder_scale = 16
        num_side_patches = train_cfg.img_size // decoder_scale
        num_patches = num_side_patches ** 2

    if train_cfg.concat_dim == 0:
        num_patches += 2

    new_predictor = hydra.utils.instantiate(
        train_cfg.predictor,
        num_patches=num_patches,
        num_frames=train_cfg.num_hist,
        dim=result["encoder"].emb_dim
        + (
            train_cfg.proprio_emb_dim * train_cfg.num_proprio_repeat
            + train_cfg.action_emb_dim * train_cfg.num_action_repeat
        )
        * (train_cfg.concat_dim),
        cfg=cfg_dict["quant_cfg_predictor"],
    )
    new_predictor.to(device)

    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    old_pred_sd = result["predictor"].state_dict()
    mapped_pred_sd = _remap_predictor_sd_old_to_new(old_pred_sd)

    missing, unexpected = new_predictor.load_state_dict(mapped_pred_sd, strict=True)
    print(f"[load_model/predictor] load_state_dict done. missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("[load_model/predictor] missing example:", missing[:30])
    if unexpected:
        print("[load_model/predictor] unexpected example:", unexpected[:30])


    print("check diff")
    new_sd = new_predictor.state_dict()
    diff_cnt = 0
    for k, v in mapped_pred_sd.items():
        if k in new_sd:
            if (v - new_sd[k]).abs().sum().item() > 0:
                diff_cnt += 1
                if diff_cnt <= 50:
                    print("diff key:", k)
    print(f"check diff done (diff_cnt={diff_cnt})")
    
    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path)
        else:
            raise ValueError("Decoder path not found in model checkpoint and is not provided in config")
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=new_predictor,
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    print("model: ", model)
    model.to(device)

    return model


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass




class ModuleListWrapper(nn.Module):
    def __init__(self, modules: nn.ModuleList):
        super().__init__()
        self.blocks = modules

    def forward(self, x, *args, **kwargs):
        out = x
        for m in self.blocks:
            out = m(out, *args, **kwargs)
        return out


def get_blocks_wm(submodel: nn.Module):
    if hasattr(submodel, "base_model") and hasattr(submodel.base_model, "blocks"):
        return submodel.base_model.blocks
    if hasattr(submodel, "blocks") and isinstance(submodel.blocks, nn.ModuleList):
        return submodel.blocks
    if hasattr(submodel, "transformer") and hasattr(submodel.transformer, "layers"):
        return submodel.transformer.layers
    return nn.ModuleList([submodel])


def register_layer0_input_hooks_for_wm(
    wm: nn.Module,
    encoder_buf: list,
    predictor_buf: list,
    hook_buffer_size: int = 8,
):
    import random
    handles = []

    if hasattr(wm, "encoder"):
        enc_layers = get_blocks_wm(wm.encoder)
        enc_layer0 = enc_layers[0]

        def enc_hook(m, x, y):
            feat = x[0].detach().cpu()
            if len(encoder_buf) < hook_buffer_size:
                encoder_buf.append(feat)
            else:
                j = random.randint(0, len(encoder_buf) - 1)
                encoder_buf[j] = feat

        handles.append(enc_layer0.register_forward_hook(enc_hook))

    if hasattr(wm, "predictor") and hasattr(wm.predictor, "transformer"):
        pred_layers = get_blocks_wm(wm.predictor)
        block0 = pred_layers[0]
        pred_layer0 = block0[0] if isinstance(block0, nn.ModuleList) else block0

        def pred_hook(m, x, y):
            feat = x[0].detach().cpu()   
            TARGET = 196

            N = feat.size(1)
            if N!= TARGET and N % TARGET==0:
                n = int(N/TARGET)
                feat = feat.view(feat.size(0), n, TARGET, *feat.shape[2:])[:, -1]  

            if len(predictor_buf) < hook_buffer_size:
                predictor_buf.append(feat)
            else:
                j = random.randint(0, len(predictor_buf) - 1)
                predictor_buf[j] = feat

        handles.append(pred_layer0.register_forward_hook(pred_hook))

    return handles



def build_base_omni_args(cfg_dict, output_dir):
    class OmniArgs:
        pass

    args = OmniArgs()
    args.nsamples = cfg_dict.get("omni_nsamples", 32)
    args.batch_size = cfg_dict.get("omni_batch_size", 8)
    args.epochs = cfg_dict.get("omni_epochs", 0)
    args.deactive_amp = cfg_dict.get("omni_deactive_amp", False)
    args.aug_loss = cfg_dict.get("omni_aug_loss", False)
    args.resume = cfg_dict.get("omni_resume", "")
    args.let = cfg_dict.get("omni_let", True)
    args.let_lr = cfg_dict.get("omni_let_lr", 1e-4)
    args.lwc_lr = cfg_dict.get("omni_lwc_lr", 1e-4)
    args.wd = cfg_dict.get("omni_wd", 0.0)
    args.real_quant = cfg_dict.get("omni_real_quant", False)
    args.output_dir = output_dir
    return args


def build_omni_args_from_ptq_config(ptq_cfg):

    w_bits = ptq_cfg.BIT_TYPE_W.bits
    a_bits = ptq_cfg.BIT_TYPE_A.bits

    symmetric = True

    if ptq_cfg.CALIBRATION_MODE_W in ["channel_wise", "per_channel"]:
        w_dynamic_method = "per_channel"
    else:
        w_dynamic_method = "per_token"

    if ptq_cfg.CALIBRATION_MODE_A in ["channel_wise", "per_channel"]:
        a_dynamic_method = "per_channel"
    else:
        a_dynamic_method = "per_token"

    weight_quant_params = dict(
        n_bits=w_bits,
        symmetric=symmetric,
        dynamic=True,
        dynamic_method=w_dynamic_method,
        metric="minmax",
        group_size=None,
        lwc=True,
        disable_zero_point=False,
    )

    act_quant_params = dict(
        n_bits=a_bits,
        symmetric=False,
        dynamic=True,
        dynamic_method=a_dynamic_method,
        metric="minmax",
        group_size=None,
        lwc=False,
        disable_zero_point=False,
    )

    return weight_quant_params, act_quant_params


def patch_wm_encode_obs_dtype_safe(wm: nn.Module):
    """
    修复 RuntimeError: Input type (float) and bias type (Half)
    通过在 encode_obs 内部强制 visual dtype 与 encoder 参数 dtype 对齐。
    """
    if not hasattr(wm, "encode_obs"):
        return

    old_encode_obs = wm.encode_obs

    def _encode_obs_safe(obs):
        if isinstance(obs, dict) and "visual" in obs:
            visual = obs["visual"]
            try:
                enc_dtype = next(wm.encoder.parameters()).dtype
            except Exception:
                enc_dtype = None
            if torch.is_tensor(visual) and enc_dtype is not None and visual.dtype != enc_dtype:
                obs = dict(obs)
                obs["visual"] = visual.to(dtype=enc_dtype)
        return old_encode_obs(obs)

    wm.encode_obs = _encode_obs_safe


def finalize_omniquant_for_inference(model: nn.Module, device: torch.device):

    model.eval()

    for m in model.modules():
        for k in [ "temp_w", "temp_b", "temp_scale", "temp_zero", "temp_zeros"]:
            if hasattr(m, k):
                try:
                    delattr(m, k)
                except Exception:
                    pass

    try:
        clear_temp_variable(model)
    except Exception:
        pass

    try:
        set_quant_state(model, weight_quant=True, act_quant=True)
    except Exception:
        pass

    model.to(device)
    torch.cuda.empty_cache()
    gc.collect()




def _get_logger(logger):
    if logger is None:
        class _Dummy:
            def info(self, *a, **k): print(*a)
        return _Dummy()
    return logger

def _traincast_and_dtype(args):

    if getattr(args, "deactive_amp", False) and int(getattr(args, "epochs", 0)) > 0:
        return nullcontext, torch.float32
    else:
        return torch.cuda.amp.autocast, torch.float16



def register_wm_let_params_encoder_block(block: nn.Module, dtype, device):

    if not (hasattr(block, "norm1") and hasattr(block, "attn") and hasattr(block.attn, "qkv")):
        return
    if not (hasattr(block, "norm2") and hasattr(block, "mlp") and hasattr(block.mlp, "fc1")):
        return

    # norm1 -> attn.qkv
    ln1 = block.norm1
    qkv = block.attn.qkv
    dim1 = ln1.normalized_shape[0]
    if getattr(qkv, "in_features", None) == dim1:
        scale = torch.ones(dim1, dtype=dtype, device=device)
        shift = torch.zeros(dim1, dtype=dtype, device=device)
        block.register_parameter("norm1__attn_qkv_smooth_scale", nn.Parameter(scale))
        block.register_parameter("norm1__attn_qkv_smooth_shift", nn.Parameter(shift))

    # norm2 -> mlp.fc1
    ln2 = block.norm2
    fc1 = block.mlp.fc1
    dim2 = ln2.normalized_shape[0]
    if getattr(fc1, "in_features", None) == dim2:
        scale = torch.ones(dim2, dtype=dtype, device=device)
        shift = torch.zeros(dim2, dtype=dtype, device=device)
        block.register_parameter("norm2__mlp_fc1_smooth_scale", nn.Parameter(scale))
        block.register_parameter("norm2__mlp_fc1_smooth_shift", nn.Parameter(shift))


def register_wm_let_params_predictor_block(block: nn.Module, dtype, device):

    if not hasattr(block, "attn") or not hasattr(block, "ff"):
        return


    if hasattr(block.attn, "norm") and hasattr(block.attn, "to_qkv"):
        ln = block.attn.norm
        lin = block.attn.to_qkv
        dim = ln.normalized_shape[0]
        if getattr(lin, "in_features", None) == dim:
            scale = torch.ones(dim, dtype=dtype, device=device)
            shift = torch.zeros(dim, dtype=dtype, device=device)
            block.register_parameter("attn_norm__attn_to_qkv_smooth_scale", nn.Parameter(scale))
            block.register_parameter("attn_norm__attn_to_qkv_smooth_shift", nn.Parameter(shift))


    if hasattr(block.ff, "net") and isinstance(block.ff.net, nn.Sequential) and len(block.ff.net) >= 2:
        if isinstance(block.ff.net[0], nn.LayerNorm) and isinstance(block.ff.net[1], nn.Module):
            ln = block.ff.net[0]
            lin = block.ff.net[1]
            dim = ln.normalized_shape[0]
            if getattr(lin, "in_features", None) == dim:
                scale = torch.ones(dim, dtype=dtype, device=device)
                shift = torch.zeros(dim, dtype=dtype, device=device)
                block.register_parameter("ff_ln0__ff_net1_smooth_scale", nn.Parameter(scale))
                block.register_parameter("ff_ln0__ff_net1_smooth_shift", nn.Parameter(shift))


@torch.no_grad()
def init_let_from_act_scales_wm_block(
    block: nn.Module,
    *,
    submodel_name: str,  # "encoder" or "predictor"
    layer_idx: int,
    act_scales: dict,
    act_shifts: dict = None,
    alpha: float = 0.5,
    dtype=None,
    device=None,
    logger=None,
    verbose: bool = True,           
    verbose_topk: int = 20,         
    dump_debug_path: str = None,    
):

    logger = _get_logger(logger)
    if act_scales is None:
        logger.info("[LET-init] act_scales is None, skip.")
        return

    if dtype is None:
        dtype = next(block.parameters()).dtype
    if device is None:
        device = next(block.parameters()).device

    hit = 0
    miss = 0
    badshape = 0

    debug_records = []  

    def _stat_tensor(t: torch.Tensor):
        t = t.detach()
        return dict(
            shape=list(t.shape),
            dtype=str(t.dtype),
            device=str(t.device),
            min=float(t.min().item()) if t.numel() else None,
            max=float(t.max().item()) if t.numel() else None,
            mean=float(t.mean().item()) if t.numel() else None,
        )

    def _compute_scale_and_shift(lin: nn.Module, key: str):
        nonlocal hit, miss, badshape

        rec = {
            "submodel": submodel_name,
            "layer_idx": layer_idx,
            "key": key,
            "found": key in act_scales,
        }

        if key not in act_scales:
            miss += 1
            debug_records.append(rec)
            return None, None

        act = act_scales[key].to(device=device, dtype=dtype).clamp(min=1e-5)
        w = lin.weight
        wstat = w.abs().max(dim=0)[0].to(device=device, dtype=dtype).clamp(min=1e-5)

        rec["act_stats"] = _stat_tensor(act)
        rec["wstat_stats"] = _stat_tensor(wstat)

        if act.numel() != wstat.numel():
            badshape += 1
            rec["badshape"] = True
            rec["act_numel"] = int(act.numel())
            rec["wstat_numel"] = int(wstat.numel())
            debug_records.append(rec)
            return None, None

        scale = (act.pow(alpha) / wstat.pow(1.0 - alpha)).clamp(min=1e-5)
        if act_shifts is not None and key in act_shifts:
            shift = act_shifts[key].to(device=device, dtype=dtype)
            rec["shift_from"] = "act_shifts"
            rec["shift_stats"] = _stat_tensor(shift)
        else:
            shift = torch.zeros_like(scale)
            rec["shift_from"] = "zeros"

        rec["scale_stats"] = _stat_tensor(scale)

        hit += 1
        debug_records.append(rec)
        return scale, shift

    # --------- encoder: norm1->qkv, norm2->fc1 ----------
    if submodel_name == "encoder":
        if hasattr(block, "norm1") and hasattr(block, "attn") and hasattr(block.attn, "qkv") \
           and hasattr(block, "norm1__attn_qkv_smooth_scale"):
            key = f"encoder.base_model.blocks.{layer_idx}.attn.qkv"
            scale, shift = _compute_scale_and_shift(block.attn.qkv, key)
            if scale is not None:
                block.norm1__attn_qkv_smooth_scale.copy_(scale)
                block.norm1__attn_qkv_smooth_shift.copy_(shift)

        if hasattr(block, "norm2") and hasattr(block, "mlp") and hasattr(block.mlp, "fc1") \
           and hasattr(block, "norm2__mlp_fc1_smooth_scale"):
            key = f"encoder.base_model.blocks.{layer_idx}.mlp.fc1"
            scale, shift = _compute_scale_and_shift(block.mlp.fc1, key)
            if scale is not None:
                block.norm2__mlp_fc1_smooth_scale.copy_(scale)
                block.norm2__mlp_fc1_smooth_shift.copy_(shift)

    # --------- predictor: attn.norm->to_qkv, ff.ln0->ff.net1 ----------
    elif submodel_name == "predictor":
        if hasattr(block, "attn") and hasattr(block.attn, "to_qkv") and hasattr(block.attn, "norm") \
           and hasattr(block, "attn_norm__attn_to_qkv_smooth_scale"):
            key = f"predictor.transformer.layers.{layer_idx}.0.to_qkv"
            scale, shift = _compute_scale_and_shift(block.attn.to_qkv, key)
            if scale is not None:
                block.attn_norm__attn_to_qkv_smooth_scale.copy_(scale)
                block.attn_norm__attn_to_qkv_smooth_shift.copy_(shift)

        if hasattr(block, "ff") and hasattr(block.ff, "net") and isinstance(block.ff.net, nn.Sequential) \
           and len(block.ff.net) >= 2 and hasattr(block, "ff_ln0__ff_net1_smooth_scale"):
            key = f"predictor.transformer.layers.{layer_idx}.1.net.1"
            lin = block.ff.net[1]
            if hasattr(lin, "weight"):
                scale, shift = _compute_scale_and_shift(lin, key)
                if scale is not None:
                    block.ff_ln0__ff_net1_smooth_scale.copy_(scale)
                    block.ff_ln0__ff_net1_smooth_shift.copy_(shift)
    else:
        raise ValueError(f"submodel_name must be encoder/predictor, got {submodel_name}")


    logger.info(
        f"[LET-init] {submodel_name} layer={layer_idx} summary: "
        f"hit={hit}, miss={miss}, badshape={badshape}, alpha={alpha}"
    )

    if verbose:

        show = debug_records[:verbose_topk]
        for r in show:
            if not r.get("found", False):
                logger.info(f"[LET-init][MISS] key={r['key']}")
            elif r.get("badshape", False):
                logger.info(
                    f"[LET-init][BADSHAPE] key={r['key']} act_numel={r.get('act_numel')} wstat_numel={r.get('wstat_numel')}"
                )
            else:
                logger.info(
                    f"[LET-init][HIT] key={r['key']} "
                    f"scale_min={r['scale_stats']['min']:.6g} scale_max={r['scale_stats']['max']:.6g} "
                    f"scale_mean={r['scale_stats']['mean']:.6g} "
                    f"act_max={r['act_stats']['max']:.6g} wstat_max={r['wstat_stats']['max']:.6g} "
                    f"shift_from={r['shift_from']}"
                )

    if dump_debug_path is not None:

        os.makedirs(os.path.dirname(dump_debug_path), exist_ok=True)
        with open(dump_debug_path, "a") as f:
            for r in debug_records:
                f.write(json.dumps(r) + "\n")



def smooth_and_quant_temporary_wm_block(block: nn.Module, args, submodel_name: str):

    for n, p in block.named_parameters():
        if n.endswith("_smooth_scale"):
            p.data = torch.clamp(p.data, min=1e-5)

    if getattr(args, "let", True):
        if submodel_name == "encoder":
            if hasattr(block, "norm1__attn_qkv_smooth_scale") and hasattr(block, "norm1"):
                smooth_ln_fcs_temporary(
                    block.norm1,
                    block.attn.qkv,
                    block.norm1__attn_qkv_smooth_scale,
                    block.norm1__attn_qkv_smooth_shift,
                )
            if hasattr(block, "norm2__mlp_fc1_smooth_scale") and hasattr(block, "norm2"):
                smooth_ln_fcs_temporary(
                    block.norm2,
                    block.mlp.fc1,
                    block.norm2__mlp_fc1_smooth_scale,
                    block.norm2__mlp_fc1_smooth_shift,
                )

        elif submodel_name == "predictor":
            if hasattr(block, "attn_norm__attn_to_qkv_smooth_scale") and hasattr(block.attn, "norm"):
                smooth_ln_fcs_temporary(
                    block.attn.norm,
                    block.attn.to_qkv,
                    block.attn_norm__attn_to_qkv_smooth_scale,
                    block.attn_norm__attn_to_qkv_smooth_shift,
                )
            if hasattr(block, "ff_ln0__ff_net1_smooth_scale") and hasattr(block.ff, "net"):
                smooth_ln_fcs_temporary(
                    block.ff.net[0],
                    block.ff.net[1],
                    block.ff_ln0__ff_net1_smooth_scale,
                    block.ff_ln0__ff_net1_smooth_shift,
                )
        else:
            raise ValueError(submodel_name)

    from models.ptq.layers import QLinear
    for m in block.modules():
        if isinstance(m, QLinear):
            if hasattr(m, "temp_weight"):
                m.quantizer.observer.update(m.temp_weight)
                m.quantizer.update_quantization_params()
                m.temp_weight = m.quantizer(m.temp_weight)
            else:
                m.temp_weight = m.quantizer(m.weight)
            if not hasattr(m, "temp_bias"):
                m.temp_bias = m.bias
            m.use_temporary_parameter = True




@torch.no_grad()
def smooth_and_quant_inplace_wm_block(block: nn.Module, args, submodel_name: str):


    if getattr(args, "let", True):
        for n, p in block.named_parameters():
            if n.endswith("_smooth_scale"):
                p.data = torch.clamp(p.data, min=1e-5)

        if submodel_name == "encoder":
            if hasattr(block, "norm1__attn_qkv_smooth_scale"):
                with torch.cuda.amp.autocast(enabled=False):
                    smooth_ln_fcs_inplace(
                        block.norm1,
                        block.attn.qkv,
                        block.norm1__attn_qkv_smooth_scale,
                        block.norm1__attn_qkv_smooth_shift,
                    )
            if hasattr(block, "norm2__mlp_fc1_smooth_scale"):
                with torch.cuda.amp.autocast(enabled=False):
                    smooth_ln_fcs_inplace(
                        block.norm2,
                        block.mlp.fc1,
                        block.norm2__mlp_fc1_smooth_scale,
                        block.norm2__mlp_fc1_smooth_shift,
                    )

        elif submodel_name == "predictor":
            if hasattr(block, "attn_norm__attn_to_qkv_smooth_scale"):
                with torch.cuda.amp.autocast(enabled=False):
                    smooth_ln_fcs_inplace(
                        block.attn.norm,
                        block.attn.to_qkv,
                        block.attn_norm__attn_to_qkv_smooth_scale,
                        block.attn_norm__attn_to_qkv_smooth_shift,
                    )
            if hasattr(block, "ff_ln0__ff_net1_smooth_scale"):
                with torch.cuda.amp.autocast(enabled=False):
                    smooth_ln_fcs_inplace(
                        block.ff.net[0],
                        block.ff.net[1],
                        block.ff_ln0__ff_net1_smooth_scale,
                        block.ff_ln0__ff_net1_smooth_shift,
                    )
        else:
            raise ValueError(submodel_name)



def omniquant_wm_new(
    submodel: nn.Module,
    inps: torch.Tensor,
    args,
    submodel_name: str,         
    act_scales: dict = None,
    act_shifts: dict = None,
    logger=None,
):

    logger = _get_logger(logger)
    dev = next(submodel.parameters()).device
    logger.info(f"[OmniQuant-WM-NEW] Start on {submodel_name} dev={dev}")

    TrainCast, dtype = _traincast_and_dtype(args)
    epochs = int(getattr(args, "epochs", 20))
    ns_total = int(inps.size(0))
    nsamples = min(int(getattr(args, "nsamples", 32)), ns_total)
    bs = int(getattr(args, "batch_size", 8))
    alpha = float(getattr(args, "alpha", 0.5))


    layers_raw = get_blocks_wm(submodel)
    layers = list(layers_raw) if isinstance(layers_raw, nn.ModuleList) else [layers_raw]


    quant_inps = inps[:nsamples].detach().cpu().contiguous().to(dev, dtype=dtype)
    fp_inps = quant_inps.clone()  
    fp_inps_2 = quant_inps.clone() if getattr(args, "aug_loss", False) else None

    loss_func = torch.nn.MSELoss()


    for i, layer_ref in enumerate(layers):
        logger.info(f"[OmniQuant-WM-NEW] === Quantize layer {i}/{len(layers)-1} ===")


        qlayer = layer_ref


        if submodel_name == "encoder":

            register_wm_let_params_encoder_block(qlayer, dtype=dtype, device=dev)
        else:

            register_wm_let_params_predictor_block(qlayer, dtype=dtype, device=dev)


        if getattr(args, "let", True):
            init_let_from_act_scales_wm_block(
                qlayer,
                submodel_name=submodel_name,
                layer_idx=i,
                act_scales=act_scales,
                act_shifts=act_shifts,
                alpha=alpha,
                dtype=dtype,
                device=dev,
                logger=logger,
                verbose=True,
                verbose_topk=10,
            )


        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        with torch.no_grad():
            fp_targets = []
            with TrainCast():
                for j in range(nsamples):
                    out = qlayer(fp_inps[j].unsqueeze(0))
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    fp_targets.append(out.squeeze(0).detach())
            fp_targets = torch.stack(fp_targets, dim=0).contiguous()


            if fp_inps_2 is not None:
                with TrainCast():
                    for j in range(nsamples):
                        out2 = qlayer(fp_inps_2[j].unsqueeze(0))
                        if isinstance(out2, (tuple, list)):
                            out2 = out2[0]
                        fp_inps_2[j] = out2.squeeze(0).detach()


        set_quant_state(qlayer, weight_quant=False, act_quant=True)

        if epochs > 0:
            qlayer.float() 
            optimizer = torch.optim.AdamW(
                [
                    {"params": let_parameters(qlayer, use_shift=True), "lr": float(getattr(args, "let_lr", 1e-4))},
                    {"params": lwc_parameters(qlayer), "lr": float(getattr(args, "lwc_lr", 1e-4))},
                ],
                weight_decay=float(getattr(args, "wd", 0.0)),
            )
            loss_scaler = NativeScalerWithGradNormCount()

            n_batches = max(1, (nsamples + bs - 1) // bs)

            for ep in range(epochs):
                loss_list = []
                norm_list = []
                for b in range(n_batches):
                    idx0 = b * bs
                    idx1 = min(nsamples, (b + 1) * bs)

                    with TrainCast():

                        smooth_and_quant_temporary_wm_block(qlayer, args, submodel_name=submodel_name)

                        out_q = qlayer(quant_inps[idx0:idx1])
                        if isinstance(out_q, (tuple, list)):
                            out_q = out_q[0]

                        loss = loss_func(fp_targets[idx0:idx1].to(out_q.dtype), out_q)
                        if fp_inps_2 is not None:
                            loss = loss + loss_func(fp_inps_2[idx0:idx1].to(out_q.dtype), out_q)

                    if not math.isfinite(loss.item()):
                        logger.info("[OmniQuant-WM-NEW] Loss is NaN/Inf, stop.")
                        raise RuntimeError("OmniQuant-WM-NEW training diverged")

                    loss_list.append(loss.detach().float().cpu())

                    optimizer.zero_grad(set_to_none=True)
                    norm = loss_scaler(
                        loss,
                        optimizer,
                        parameters=get_omni_parameters(qlayer, use_shift=True),
                    )
                    if norm is not None:
                        norm_list.append(norm.detach().float().cpu())

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean() if len(norm_list) else torch.tensor(0.)
                logger.info(
                    f"[OmniQuant-WM-NEW] layer {i} epoch {ep} "
                    f"loss={loss_mean.item():.6f} norm={norm_mean.item():.6f} "
                    f"max_mem={torch.cuda.max_memory_allocated(dev)/1024**2:.2f} MB"
                )

            clear_temp_variable(qlayer)
            del optimizer
            torch.cuda.empty_cache()
            gc.collect()


        smooth_and_quant_inplace_wm_block(qlayer, args, submodel_name=submodel_name)

        with torch.no_grad():
            set_quant_state(qlayer, weight_quant=False, act_quant=False)  
            with TrainCast():
                for j in range(nsamples):
                    outn = qlayer(quant_inps[j].unsqueeze(0))
                    if isinstance(outn, (tuple, list)):
                        outn = outn[0]
                    quant_inps[j] = outn.squeeze(0).detach()

         
        if isinstance(layers_raw, nn.ModuleList):
            layers_raw[i] = qlayer
        else:
            layer_ref.load_state_dict(qlayer.state_dict())

        torch.cuda.empty_cache()
        gc.collect()

    logger.info(f"[OmniQuant-WM-NEW] Finished {submodel_name}.")



def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict)
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}/"
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)
    model_cfg["predictor"]["_target_"] = "models.vit_q.QViTPredictor"
    model_cfg["encoder"]["_target_"] = "models.dino_q.QDinoV2Encoder"

    from models.ptq import Config
    quant_cfg_predictor = Config(
        w_bit=cfg_dict["predictor_wbit"],
        a_bit=cfg_dict["predictor_abit"],
        w_quant_method=cfg_dict["w_quant_method"],
        a_quant_method=cfg_dict["a_quant_method"],
        calib_mode_a=cfg_dict['calib_mode_a']
    )
    cfg_dict["quant_cfg_predictor"] = quant_cfg_predictor
    quant_cfg_encoder = Config(
        w_bit=cfg_dict["encoder_wbit"],
        a_bit=cfg_dict["encoder_abit"],
        w_quant_method=cfg_dict["w_quant_method"],
        a_quant_method=cfg_dict["a_quant_method"],
        calib_mode_a=cfg_dict['calib_mode_a']
    )
    cfg_dict["quant_cfg_encoder"] = quant_cfg_encoder
    print("model_cfg: ", model_cfg)
    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    model = load_model(model_ckpt, model_cfg, cfg_dict, num_action_repeat, device=device)


    if model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [lambda: gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs) for _ in range(cfg_dict["n_evals"])]
        )

    import datetime
    print(">>> current time: ", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    cfg_dict_train = copy.deepcopy(cfg_dict)
    cfg_dict_train["planner"]["max_iter"] = cfg_dict["quant_iter"]

    encoder_layer0_buf = []
    predictor_layer0_buf = []
    hook_buffer_size = cfg_dict.get("hook_buffer_size", 8)

    layer0_handles = register_layer0_input_hooks_for_wm(
        wm=model,
        encoder_buf=encoder_layer0_buf,
        predictor_buf=predictor_layer0_buf,
        hook_buffer_size=hook_buffer_size,
    )

    print("Registered layer[0] input hooks for encoder/predictor.")
    plan_workspace_train = PlanWorkspace(
        cfg_dict=cfg_dict_train,
        wm=model,
        dset=dset["train"],
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
        cache_input= True,
        calib_state= False,
    )

    _ = plan_workspace_train.perform_planning(tag="cache_input")

    for h in layer0_handles:
        h.remove()
    print("Removed layer[0] input hooks.")

    encoder_inps = torch.cat(encoder_layer0_buf, dim=0).contiguous() if len(encoder_layer0_buf) > 0 else None
    predictor_inps = torch.cat(predictor_layer0_buf, dim=0).contiguous() if len(predictor_layer0_buf) > 0 else None

    if cfg_dict.get("quant", False) and cfg_dict.get("w_quant_method", "").lower() == "omniquant":
        print("========== Start OmniQuant quantization (world model) ==========")

        scales_dir = "/cache/scale"
        if 'wall' in cfg_dict['model_name']:
            scales_path = os.path.join(scales_dir, f"wm_act_scales_wall_{cfg_dict['scale_tag']}.pt")
        elif 'pusht' in cfg_dict['model_name']:
            scales_path = os.path.join(scales_dir, f"wm_act_scales_pusht_{cfg_dict['scale_tag']}.pt")
        act_scales = torch.load(scales_path, map_location="cpu")
        
        
        w_qparams_enc, a_qparams_enc = build_omni_args_from_ptq_config(quant_cfg_encoder)
        w_qparams_pred, a_qparams_pred = build_omni_args_from_ptq_config(quant_cfg_predictor)

        omni_args_enc = build_base_omni_args(cfg_dict, output_dir=os.path.join(output_dir, "omniquant_encoder"))
        omni_args_pred = build_base_omni_args(cfg_dict, output_dir=os.path.join(output_dir, "omniquant_predictor"))

        omni_args_enc.weight_quant_params = w_qparams_enc
        omni_args_enc.act_quant_params = a_qparams_enc
        omni_args_pred.weight_quant_params = w_qparams_pred
        omni_args_pred.act_quant_params = a_qparams_pred

        if cfg_dict.get("quant_encoder", False) and encoder_inps is not None:
            print("[OmniQuant-WM] Running OmniQuant on encoder ...")
            omniquant_wm_new(
                model.encoder,
                encoder_inps,
                omni_args_enc,
                submodel_name="encoder",
                act_scales=act_scales,
                act_shifts=None,   
                logger=log,
            )
            print("[OmniQuant-WM] Encoder quantization finished.")

        if predictor_inps is not None:
            print("[OmniQuant-WM] Running OmniQuant on predictor ...")
            omniquant_wm_new(
                model.predictor,
                predictor_inps,
                omni_args_pred,
                submodel_name="predictor",
                act_scales=act_scales,
                act_shifts=None,
                logger=log,
            )
            print("[OmniQuant-WM] Predictor quantization finished.")


        print("========== OmniQuant quantization (world model) finished ==========")
    for submodel in [model.encoder,model.predictor]:
        layers_raw = get_blocks_wm(submodel)
        layers = list(layers_raw) if isinstance(layers_raw, nn.ModuleList) else [layers_raw]
        for i, layer in enumerate(layers):
            set_quant_state(layer, weight_quant=False, act_quant=False)

    plan_workspace_train = PlanWorkspace(
        cfg_dict=cfg_dict_train,
        wm=model,
        dset=dset["train"],
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
        cache_input= False,
        calib_state=True,
    )
    _ = plan_workspace_train.perform_planning(tag="calib")
    for submodel in [model.encoder,model.predictor]:
        layers_raw = get_blocks_wm(submodel)
        layers = list(layers_raw) if isinstance(layers_raw, nn.ModuleList) else [layers_raw]
        for i, layer in enumerate(layers):
            set_quant_state(layer, weight_quant=True, act_quant=True)
            set_lwc_state(layer,lwc=True)


    try:
        del encoder_inps, predictor_inps
    except Exception:
        pass
    try:
        encoder_layer0_buf.clear()
        predictor_layer0_buf.clear()
    except Exception:
        pass

    torch.cuda.empty_cache()
    gc.collect()

    print(">>> current time: ", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset["valid"],
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
        cache_input= False,
        calib_state=False,
    )
    logs = plan_workspace.perform_planning(tag="valid")

    torch.cuda.empty_cache()
    print(">>> current time: ", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    return logs


@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = False
    print("cfg_dict: ", cfg_dict)
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
