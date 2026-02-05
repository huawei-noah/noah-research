import os
import gym
import json
import hydra
import random
import pickle
import wandb
import logging
import warnings
import numpy as np
import submitit
import torch
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

from quant_utils.awq.auto_clip import auto_clip_block, apply_clip
from quant_utils.awq.auto_scale import auto_scale_block, apply_scale
from quant_utils.awq.module import ModuleListWrapper
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
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  
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
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
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

        if calib_state:
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

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )
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
        else: # random_state or dset or random_action
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
            n_plot_samples=self.cfg_dict["n_plot_samples"], # 10
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"], # planning.mpc.MPCPlanner
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
            do_quant=self.cfg_dict['quant'],
            do_quant_encoder=self.cfg_dict['quant_encoder'],
            calib_state = self.calib_state,
            quant_iter=self.cfg_dict['quant_iter'],
        )


        from planning.mpc import MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"] # 5
            self.planner.n_taken_actions = cfg_dict["goal_H"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]

        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []
        
        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
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
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + 1)
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")


        for i in range(self.n_evals): # 50
            max_offset = -1
            while max_offset < 0: 
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
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

    def perform_planning(self):
        if self.debug_dset_init: 
            actions_init = self.gt_actions
        else:
            actions_init = None
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )

        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
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

import re
from collections import OrderedDict
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
    import os
    import re
    from collections import OrderedDict
    import torch
    import hydra

    def _remap_predictor_sd_old_to_new(old_sd: dict) -> dict:

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
            train_cfg.encoder,  # e.g. models.dino_q.QDinoV2Encoder
            cfg=cfg_dict["quant_cfg_encoder"],
        )


    if result["encoder"].latent_ndim == 1:
        num_patches = 1
    else:
        decoder_scale = 16  # from vqvae
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
            raise ValueError(
                "Decoder path not found in model checkpoint and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None


    model = hydra.utils.instantiate(
        train_cfg.model,  # e.g. models.visual_world_model.VWorldModel
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=new_predictor,  # 用新 predictor
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


def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]: # False
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}/"
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    model_cfg['predictor']['_target_'] = "models.vit_q.QViTPredictor" # models.vit.ViTPredictor
    model_cfg['encoder']['_target_'] = "models.dino_q.QDinoV2Encoder" # models.dino.DinoV2Encoder
    from models.ptq import Config
    q_config = {
        "zero_point": True,  # by default True
        "q_group_size": int(os.environ['W_GROUP_SIZE']) if 'W_GROUP_SIZE' in os.environ else -1,  # whether to use group quantization
    }
    
    
    quant_cfg_predictor = Config(w_bit=cfg_dict['predictor_wbit'], a_bit=cfg_dict['predictor_abit'], w_quant_method=cfg_dict['w_quant_method'], a_quant_method=cfg_dict['a_quant_method'],calib_mode_a=cfg_dict['calib_mode_a'])
    cfg_dict['quant_cfg_predictor'] = quant_cfg_predictor
    quant_cfg_encoder = Config(w_bit=cfg_dict['encoder_wbit'], a_bit=cfg_dict['encoder_abit'], w_quant_method=cfg_dict['w_quant_method'], a_quant_method=cfg_dict['a_quant_method'],calib_mode_a=cfg_dict['calib_mode_a'])
    cfg_dict['quant_cfg_encoder'] = quant_cfg_encoder

    print("model_cfg: ", model_cfg)

    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call( 
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist, # 3
        num_pred=model_cfg.num_pred, # 1
        frameskip=model_cfg.frameskip, # 5
    )


    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    ) 
    model = load_model(model_ckpt, model_cfg, cfg_dict, num_action_repeat, device=device)


    if model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"]) # 50
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"]) # 50
            ]
        )

  
    import datetime
    print(">>> current time: ", datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    import copy
    cfg_dict_train = copy.deepcopy(cfg_dict)
    cfg_dict_train["planner"]["max_iter"] = cfg_dict['quant_iter']
    print('cfg_dict_train["planner"]["max_iter"]: ', cfg_dict_train["planner"]["max_iter"])
    print('cfg_dict["planner"]["max_iter"]: ', cfg_dict["planner"]["max_iter"])


    def get_blocks_wm(submodel: nn.Module):

        if hasattr(submodel, "base_model") and hasattr(submodel.base_model, "blocks"):
            return submodel.base_model.blocks

        if hasattr(submodel, "blocks") and isinstance(submodel.blocks, nn.ModuleList):
            return submodel.blocks

        if hasattr(submodel, "transformer") and hasattr(submodel.transformer, "layers"):
            return submodel.transformer.layers

        return nn.ModuleList([submodel])



    def get_named_linears(module: nn.Module):

        return {
            name: m
            for name, m in module.named_modules()
            if isinstance(m, nn.Linear)
        }

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

            h_enc = enc_layer0.register_forward_hook(enc_hook)
            handles.append(h_enc)



        if hasattr(wm, "predictor") and hasattr(wm.predictor, "transformer"):
            pred_layers = get_blocks_wm(wm.predictor)
            block0 = pred_layers[0]

            pred_layer0 = block0[0] if isinstance(block0, nn.ModuleList) else block0

            def pred_hook(m, x, y):
                feat = x[0].detach().cpu()   # typically [B, N, C] or [N, C]
                TARGET = 196
                N = feat.size(1)

                if N!= TARGET and N % TARGET==0:
                    n = int(N/TARGET)
                    feat = feat.view(feat.size(0), n, TARGET, *feat.shape[2:])[:, -1]  # [B,196,C]

                        
                
                if len(predictor_buf) < hook_buffer_size:
                    predictor_buf.append(feat)
                else:
                    j = random.randint(0, len(predictor_buf) - 1)
                    predictor_buf[j] = feat

            h_pred = pred_layer0.register_forward_hook(pred_hook)
            handles.append(h_pred)

        return handles



    def run_awq_wm(
        submodel: nn.Module,
        inps: torch.Tensor,
        w_bit: int,
        q_config,
        tag,
        auto_scale: bool = True,
        mse_range: bool = True,
        layer_batch_size: int = 8,

    ):

        layers_raw = get_blocks_wm(submodel)  #


        if isinstance(layers_raw, nn.ModuleList):
            layers_iter = list(layers_raw)
        elif isinstance(layers_raw, nn.Module):
            layers_iter = [layers_raw]
        else:

            layers_iter = list(layers_raw)

        layer_kwargs = {}

        device = next(submodel.parameters()).device


        cur_inps = inps.detach().cpu()

        awq_results = {
            "scale": [],
            "clip": [],
        }

        for i, layer_ref in enumerate(layers_iter):

            if isinstance(layer_ref, nn.ModuleList):
                layer = ModuleListWrapper(layer_ref)
            else:
                layer = layer_ref

            print(f"[AWQ-WM] Processing layer {i}: {type(layer_ref)} (wrapped as {type(layer)})")

            layer = layer.to(device)


            named_linears = get_named_linears(layer)
            input_feat = defaultdict(list)
            handles = []

            def cache_input_hook(m, x, y, name, feat_dict):

                x = x[0].detach().cpu()
                feat_dict[name].append(x)

            for name, m in named_linears.items():
                h = m.register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
                handles.append(h)


            with torch.no_grad():
                outs = []
                N = cur_inps.size(0)
                for start in range(0, N, layer_batch_size):
                    end = min(N, start + layer_batch_size)

                    chunk = cur_inps[start:end].to(device)

                    out_chunk = layer(chunk, **layer_kwargs)
                    if isinstance(out_chunk, (tuple, list)):
                        out_chunk = out_chunk[0]

                    outs.append(out_chunk.detach().cpu())
                    del chunk, out_chunk
                    torch.cuda.empty_cache()

                cur_inps = torch.cat(outs, dim=0)

            for h in handles:
                h.remove()
            handles.clear()

            input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}


            print(f"[DEBUG] Layer {i} ({type(layer_ref)}), input_feat keys:", list(input_feat.keys()))
            torch.cuda.empty_cache()
                       

            if auto_scale:
                try:

                    scales_list = auto_scale_block(
                        layer,
                        layer_kwargs,
                        w_bit=w_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                    )

                except NotImplementedError as e:

                    print(
                        f"[AWQ-WM] auto_scale_block not supported for layer {i} "
                        f"({type(layer)}), skip auto_scale. Error: {e}"
                    )

                    scales_list = []

                else:

                    apply_scale(layer, scales_list, input_feat_dict=input_feat)
                    awq_results["scale"].extend(scales_list)

            torch.cuda.empty_cache()

            if mse_range:
                try:
                    clip_list = auto_clip_block(
                        layer,
                        w_bit=w_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                    )
                except NotImplementedError as e:
                    print(
                        f"[AWQ-WM] auto_clip_block not supported for layer {i} "
                        f"({type(layer)}), skip mse_range. Error: {e}"
                    )
                    clip_list = []
                else:
                    apply_clip(layer, clip_list)
                    awq_results["clip"].extend(clip_list)

            if isinstance(layers_raw, nn.ModuleList):

                layers_raw[i] = layer_ref
            else:
                pass

            layer.cpu()
            del input_feat
            gc.collect()
            torch.cuda.empty_cache()

        return awq_results
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
        # dset=dset,
        dset=dset["train"],
        # env=env_train,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
        calib_state=True
    )

    logs = plan_workspace_train.perform_planning()


    for h in layer0_handles:
        h.remove()
    print("Removed layer[0] input hooks.")


    encoder_inps = None
    predictor_inps = None

    if len(encoder_layer0_buf) > 0:
        encoder_inps = torch.cat(encoder_layer0_buf, dim=0).contiguous()   
        print("encoder layer[0] inps shape:", encoder_inps.shape, encoder_inps.device)

    if len(predictor_layer0_buf) > 0:
        predictor_inps = torch.cat(predictor_layer0_buf, dim=0).contiguous()  
        print("predictor layer[0] inps shape:", predictor_inps.shape, predictor_inps.device)


    try:
        del plan_workspace_train
    except Exception:
        pass

    gc.collect()
    torch.cuda.empty_cache()


    if cfg_dict.get("quant", False) and cfg_dict.get("w_quant_method", "").lower() == "awq":
        print("========== Start AWQ quantization (world model) ==========")

        awq_auto_scale = cfg_dict.get("awq_auto_scale", True)
        awq_mse_range = cfg_dict.get("awq_mse_range", True)
        # awq_auto_scale = False
        # awq_mse_range = False
        # ---------- encoder ----------
        if cfg_dict.get("quant_encoder", False) and encoder_inps is not None:
            print("Running AWQ on encoder with captured layer[0] inputs ...")
            awq_res_encoder = run_awq_wm(
                submodel=model.encoder,
                inps=encoder_inps,
                w_bit=cfg_dict["encoder_wbit"],
                q_config=q_config ,
                tag="encoder",
                auto_scale=awq_auto_scale,
                mse_range=awq_mse_range,
            )
            print("AWQ on encoder finished.")

        # ---------- predictor ----------
        if predictor_inps is not None:
            print("Running AWQ on predictor with captured layer[0] inputs ...")
            awq_res_pred = run_awq_wm(
                submodel=model.predictor,
                inps=predictor_inps,
                w_bit=cfg_dict["predictor_wbit"],
                q_config=q_config ,
                tag="predictor",
                auto_scale=awq_auto_scale,
                mse_range=awq_mse_range,
            )
            print("AWQ on predictor finished.")

        print("========== AWQ quantization (world model) finished ==========")
    else:
        if cfg_dict.get("quant", False) and cfg_dict.get("w_quant_method", "").lower() == "awq":
            print("Warning: quant=True & w_quant_method=awq, but no captured inps or run_awq_wm not executed.")


        print("========== AWQ quantization (world model) finished ==========")

    try:
        del encoder_inps
    except NameError:
        pass
    try:
        del predictor_inps
    except NameError:
        pass

    try:
        encoder_layer0_buf.clear()
    except Exception:
        pass
    try:
        predictor_layer0_buf.clear()
    except Exception:
        pass
    torch.cuda.empty_cache()
    gc.collect()



    import datetime
    print(">>> current time: ", datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    model = model.to(device)
    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        # dset=dset,
        dset=dset["valid"],
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
        calib_state=False
    )



    logs = plan_workspace.perform_planning()

    torch.cuda.empty_cache()
    

    import datetime
    print(">>> current time: ", datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

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


