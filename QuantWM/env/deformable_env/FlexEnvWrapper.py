from .src.sim.sim_env.flex_env import FlexEnv

import os
import numpy as np
import gym
import torch
import math
from .src.sim.utils import load_yaml

ENV_ACTION_DIM = 4
BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../"))


def aggregate_dct(dcts):
    full_dct = {}
    for dct in dcts:
        for key, value in dct.items():
            if key not in full_dct:
                full_dct[key] = []
            full_dct[key].append(value)
    for key, value in full_dct.items():
        if isinstance(value[0], torch.Tensor):
            full_dct[key] = torch.stack(value)
        else:
            full_dct[key] = np.stack(value)
    return full_dct

def chamfer_distance(x, y):
    # x: [B, N, D]
    # y: [B, M, D]
    # NOTE: only the first 3 dim is taken!
    x = x[:, :, None, :3].repeat(1, 1, y.size(1), 1) # x: [B, N, M, D]
    y = y[:, None, :, :3].repeat(1, x.size(1), 1, 1) # y: [B, N, M, D]
    dis = torch.norm(torch.add(x, -y), 2, dim=3)    # dis: [B, N, M]
    dis_xy = torch.mean(torch.min(dis, dim=2)[0])   # dis_xy: mean over N
    dis_yx = torch.mean(torch.min(dis, dim=1)[0])   # dis_yx: mean over M
    return dis_xy + dis_yx

class FlexEnvWrapper(FlexEnv):
    def __init__(self, object_name):
        config = load_yaml(os.path.join(BASE_DIR, f'conf/env/{object_name}.yaml'))
        super().__init__(config=config)
        self.action_dim = 4
        self.proprio_start_idx = 0
        self.proprio_end_idx = 2
        self.success_threshold = 0

    def eval_state(self, goal_state, cur_state):
        CD = chamfer_distance(torch.tensor([goal_state]), torch.tensor([cur_state]))
        print("CD: ", CD.item())
        success, chamfer_dist = CD.item() < 0, CD.item()
        metrics = {
            "success": success,
            "chamfer_distance": chamfer_dist,
        }
        return metrics

    def update_env(self, env_info): 
        pass

    def sample_random_init_goal_states(self, seed):
        """
        Return a random state
        """
        self.seed(seed)
        imgs_list, particle_pos_list, eef_states_list = self.reset(save_data=True)

        def transfer_state(state, scale, theta, delta):
            assert state.shape[1] == 4
            state = state.clone()
            state[:, 0] *= scale
            state[:, 2] *= scale
            theta = math.radians(theta)
            rotation_matrix_y = torch.tensor(
                [
                    [math.cos(theta), 0, math.sin(theta)],
                    [0, 1, 0],
                    [-math.sin(theta), 0, math.cos(theta)],
                ]
            )
            rotated_state = torch.matmul(state[:, :3], rotation_matrix_y.T)
            state[:, :3] = rotated_state

            # randomly select negative or positive delta
            delta_x = delta * np.random.choice([-1, 1])
            delta_z = delta * np.random.choice([-1, 1])

            state[:, 0] += delta_x
            state[:, 2] += delta_z
            return state

        if self.obj == 'granular':
            state = torch.tensor(particle_pos_list[0])
            goal_scale, goal_theta, goal_delta = (
                np.random.uniform(0.6, 0.9),
                0,
                np.random.uniform(-1, 1),
            )
            goal_state = transfer_state(state, goal_scale, goal_theta, goal_delta)
            return state, goal_state
        
        elif self.obj == 'rope':
            state = torch.tensor(particle_pos_list[0])
            goal_scale, goal_theta, goal_delta = (
                1,
                np.random.uniform(0, 90),
                np.random.uniform(1, -1),
            )
            goal_state = transfer_state(state, goal_scale, goal_theta, goal_delta)
            return state, goal_state


    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        """
        self.seed(seed)
        imgs_list, particle_pos_list, eef_states_list = self.reset(save_data=True)
        self.set_states(init_state)
        img = self.get_one_view_img()
        obs = {
            "visual": img[..., :3][..., ::-1],
            "proprio": np.zeros(1).astype(np.float32),
        }

        state_dct = {"state": particle_pos_list[-1], "proprio": eef_states_list[-1]}
        return obs, state_dct

    def step_multiple(self, actions):
        obses = []
        infos = []
        for action in actions:
            step_data = [], [], []
            obs, out_data = self.step(
                action, save_data=True, data=step_data
            )  # o: (H, W, 5)
            imgs_list, particle_pos_list, eef_states_list = (
                out_data  # imgs_list: (num_cameras, H, W, 5); particle_pos_list: (n_particles, 4); eef_states_list: (1,14)
            )

            obs = {
                "visual": imgs_list[-1][self.camera_view][..., :3][..., ::-1],
                "proprio": np.zeros(1).astype(np.float32), # dummy proprio
            }
            info = {"pos_agent": eef_states_list[-1], "state": particle_pos_list[-1]} # dummy 
            obses.append(obs)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = 0
        dones = False
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, state_dct = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state_dct["state"], 0), infos["state"]])
        states = np.stack(states)
        return obses, states
