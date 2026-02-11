import torch
import hydra
import copy
import numpy as np
from einops import rearrange, repeat
from utils import slice_trajdict_with_t
from .base_planner import BasePlanner


class MPCPlanner(BasePlanner):
    """
    an online planner so feedback from env is allowed
    """

    def __init__(
        self,
        max_iter,
        n_taken_actions,
        sub_planner,
        wm,
        env,  
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="mpc",
        log_filename="logs.json",
        do_quant=False,
        do_quant_encoder=False,
        calib_state=False,
        quant_iter = 2,
        tag = 'valid',
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.tag = tag
        self.env = env
        self.do_quant = do_quant
        self.do_quant_encoder = do_quant_encoder
        self.calib_state = calib_state
        self.quant_iter = quant_iter
        self.max_iter = np.inf if max_iter is None else max_iter
        self.n_taken_actions = n_taken_actions 
        self.logging_prefix = logging_prefix
        sub_planner["_target_"] = sub_planner["target"]
        self.sub_planner = hydra.utils.instantiate(
            sub_planner,
            wm=self.wm,
            action_dim=self.action_dim,
            objective_fn=self.objective_fn,
            preprocessor=self.preprocessor,
            evaluator=self.evaluator,  
            wandb_run=self.wandb_run,

            log_filename=log_filename,
        )
        self.is_success = None
        self.action_len = None  
        self.iter = 0
        self.planned_actions = []

    def _apply_success_mask(self, actions):
        device = actions.device
        mask = torch.tensor(self.is_success).bool()
        actions[mask] = 0
        masked_actions = rearrange(
            actions[mask], "... (f d) -> ... f d", f=self.evaluator.frameskip
        )
        masked_actions = self.preprocessor.normalize_actions(masked_actions.cpu())
        masked_actions = rearrange(masked_actions, "... f d -> ... (f d)")
        actions[mask] = masked_actions.to(device)
        return actions

    def plan(self, obs_0, obs_g, actions=None):

        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()
        print("[MPCPlanner.plan] n_evals: ", n_evals) 
        print("[MPCPlanner.plan] self.is_success: ", self.is_success)
        print("[MPCPlanner.plan] self.max_iter: ", self.max_iter) 
        print("MPC plan begin-------------------------------")
        

        if self.do_quant and self.calib_state:
            print('calibration begin...')
            self.wm.predictor.model_open_calibrate()
            if self.do_quant_encoder:
                self.wm.encoder.model_open_calibrate()
            
        plot_trajs=True
        cur_obs_0 = obs_0 
        memo_actions = None
        while not np.all(self.is_success) and self.iter < self.max_iter: 
            self.sub_planner.logging_prefix = f"plan_{self.iter}"
            print("\n-------------------------------")
            print(f">>> MPC iter {self.iter} begin ------- ")
                        
            if self.do_quant and self.calib_state:
                if self.iter < self.quant_iter: 
                    print('calibration stage...')
                


            actions, _ = self.sub_planner.plan(
                obs_0=cur_obs_0,
                obs_g=obs_g,
                actions=memo_actions,
            )           
            

            torch.cuda.empty_cache()

            taken_actions = actions.detach()[:, : self.n_taken_actions]
            self._apply_success_mask(taken_actions)
            memo_actions = actions.detach()[:, self.n_taken_actions :]
            self.planned_actions.append(taken_actions)

            action_so_far = torch.cat(self.planned_actions, dim=1)
            self.evaluator.assign_init_cond(
                obs_0=init_obs_0,
                state_0=init_state_0,
            )
            


            if self.iter >= 6:
                plot_trajs=False

            import datetime
            print(f">>>MPC iter {self.iter} Eval, current time: {datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')} ------- ")
            logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                action_so_far,
                self.action_len,
                filename=f"plan{self.iter}_{self.tag}",
                save_video=True,
                # save_video=False,
                plot_trajs=plot_trajs
            )

            new_successes = successes & ~self.is_success  
            self.is_success = (
                self.is_success | successes
            )  
            self.action_len[new_successes] = (
                (self.iter + 1) * self.n_taken_actions
            )  

            print("self.is_success: ", self.is_success)
            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}

            logs.update({"plan_iter": self.iter + 1})
            self.wandb_run.log(logs)
            self.dump_logs(logs)

            e_final_obs = slice_trajdict_with_t(e_obses, start_idx=-1)
            cur_obs_0 = e_final_obs
            e_final_state = e_states[:, -1]
            self.evaluator.assign_init_cond(
                obs_0=e_final_obs,
                state_0=e_final_state,
            )

            if self.do_quant and self.calib_state:
                if self.iter + 1 == self.quant_iter:
                    print('calibration last time...')
                    self.wm.predictor.model_open_last_calibrate()
                    if self.do_quant_encoder:
                        self.wm.encoder.model_open_last_calibrate()
                    self.sub_planner.plan_onestep(
                        obs_0=cur_obs_0,
                        obs_g=obs_g,
                        actions=memo_actions,
                    ) 
                    self.wm.predictor.model_close_calibrate()
                    if self.do_quant_encoder:
                        self.wm.encoder.model_close_calibrate()
                    print('calibration done. Using quant model mode.')
                    self.wm.predictor.model_quant()
                    if self.do_quant_encoder:
                        self.wm.encoder.model_quant()
                    print('quantization stage...')

            self.iter += 1
            self.sub_planner.logging_prefix = f"plan_{self.iter}"

            torch.cuda.empty_cache()
            


        planned_actions = torch.cat(self.planned_actions, dim=1)
        self.evaluator.assign_init_cond(
            obs_0=init_obs_0,
            state_0=init_state_0,
        )

        return planned_actions, self.action_len
