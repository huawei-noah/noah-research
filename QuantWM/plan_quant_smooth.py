import os
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
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict
import re
from collections import OrderedDict
from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed
from quant_utils.smoothquant_utils import smooth_wm_predictor, smooth_wm_encoder


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
        self.debug_dset_init = cfg_dict["debug_dset_init"] # False

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
            n_plot_samples=self.cfg_dict["n_plot_samples"], # 10
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
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
            do_quant=self.cfg_dict['quant'],
            do_quant_encoder=self.cfg_dict['quant_encoder'],
            calib_state = self.calib_state,
            quant_iter=self.cfg_dict['quant_iter'],
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
       
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)
      
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )
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

            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + 1)
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
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
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


        for i in range(self.n_evals):
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
                    "goal_H": self.goal_H, # 5
                },
                f,
            )

        file_path = os.path.abspath(self.dump_targets_file)
        print(f"Dumped plan targets to {file_path}") 

    def perform_planning(self):
        if self.debug_dset_init: # False
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


def load_model(model_ckpt, train_cfg, cfg_dict, num_action_repeat, device):
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
        if True:
            result["encoder"] = hydra.utils.instantiate(
                train_cfg.encoder, 
                cfg=cfg_dict['quant_cfg_encoder']
            )
        else:
            result["encoder"] = hydra.utils.instantiate(
                train_cfg.encoder, 
            )


    if result["encoder"].latent_ndim == 1:  
        num_patches = 1
    else:
        decoder_scale = 16  
        num_side_patches = train_cfg.img_size // decoder_scale
        num_patches = num_side_patches**2

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
        cfg=cfg_dict['quant_cfg_predictor']
    )
    new_predictor.to(device)

    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")
    old_pred_sd = result["predictor"].state_dict()
    mapped_pred_sd = _remap_predictor_sd_old_to_new(old_pred_sd)
    missing, unexpected = new_predictor.load_state_dict(
        mapped_pred_sd,
        strict=True
    )
    print(f"[load_model/smoothquant] predictor load done. "
        f"missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("[load_model/smoothquant] missing example:", missing[:20])
    if unexpected:
        print("[load_model/smoothquant] unexpected example:", unexpected[:20])
    # 数值一致性 sanity check（强烈建议保留）
    print("[load_model/smoothquant] check predictor weight diff")
    new_sd = new_predictor.state_dict()
    diff_cnt = 0
    for k, v in mapped_pred_sd.items():
        if k in new_sd:
            if (v - new_sd[k]).abs().sum().item() > 0:
                diff_cnt += 1
                if diff_cnt <= 20:
                    print("diff key:", k)
    print(f"[load_model/smoothquant] check diff done (diff_cnt={diff_cnt})")



    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

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
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None
    

    model = hydra.utils.instantiate(
        train_cfg.model, 
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        # predictor=result["predictor"],
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


    #-----------Smooth-----------
    scales_dir = "/cache/scale"
    if 'wall' in cfg_dict['model_name']:
        scales_path = os.path.join(scales_dir, f"wm_act_scales_wall_{cfg_dict['scale_tag']}.pt")
    elif 'pusht' in cfg_dict['model_name']:
        scales_path = os.path.join(scales_dir, f"wm_act_scales_pusht_{cfg_dict['scale_tag']}.pt")
    act_scales = torch.load(scales_path, map_location="cpu")


    smooth_wm_encoder(model, act_scales, alpha=0.5, logger=print)
    smooth_wm_predictor(model, act_scales, alpha=0.5, logger=print) 

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
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    import datetime
    print(">>> current time: ", datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    import copy
    cfg_dict_train = copy.deepcopy(cfg_dict)
    cfg_dict_train["planner"]["max_iter"] = cfg_dict['quant_iter']
    print('cfg_dict_train["planner"]["max_iter"]: ', cfg_dict_train["planner"]["max_iter"])
    print('cfg_dict["planner"]["max_iter"]: ', cfg_dict["planner"]["max_iter"])

    plan_workspace_train = PlanWorkspace(
        cfg_dict=cfg_dict_train,
        wm=model,
        # dset=dset,
        dset=dset["train"],
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
        calib_state=True
    )


    logs = plan_workspace_train.perform_planning()

    torch.cuda.empty_cache()

    import datetime
    print(">>> current time: ", datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

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
