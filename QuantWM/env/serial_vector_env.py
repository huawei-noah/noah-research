import numpy as np
from utils import aggregate_dct

# for debugging environments, and envs that are not compatible with SubprocVectorEnv
class SerialVectorEnv:
    """
    obs, reward, done, info
    obs: dict, each key has shape (num_env, ...)
    reward: (num_env, )
    done: (num_env, )
    info: tuple of length num_env, each element is a dict
    """

    def __init__(self, envs):
        self.envs = envs
        self.num_envs = len(envs)

    def sample_random_init_goal_states(self, seed):
        init_state, goal_state = zip(*(self.envs[i].sample_random_init_goal_states(seed[i]) for i in range(self.num_envs)))
        return np.stack(init_state), np.stack(goal_state)
    
    def update_env(self, env_info):
        [self.envs[i].update_env(env_info[i]) for i in range(self.num_envs)]
    
    def eval_state(self, goal_state, cur_state): 
        eval_result = []
        for i in range(self.num_envs):
            env = self.envs[i]
            eval_result.append(env.eval_state(goal_state[i], cur_state[i]))
        eval_result = aggregate_dct(eval_result)
        return eval_result

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        obs: (num_envs, H, W, C)
        state: tuple of num_envs dicts
        """
        obs = []
        state = []
        for i in range(self.num_envs):
            env = self.envs[i]
            cur_seed = seed[i]
            cur_init_state = init_state[i]
            o, s = env.prepare(cur_seed, cur_init_state)
            obs.append(o)
            state.append(s)
        obs = aggregate_dct(obs)
        state = np.stack(state)
        return obs, state

    def step_multiple(self, actions):
        """
        actions: (num_envs, T, action_dim)
        obses: (num_envs, T, H, W, C)
        infos: tuple of length num_envs, each element is a dict
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for i in range(self.num_envs):
            env = self.envs[i]
            cur_actions = actions[i]
            obs, reward, done, info = env.step_multiple(cur_actions)
            obses.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        obses = np.stack(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = tuple(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        obses: (num_envs, T, H, W, C)
        states: (num_envs, T, D)
        proprios: (num_envs, T, D_p)
        """
        obses = []
        states = []
        for i in range(self.num_envs):
            env = self.envs[i]
            cur_seed = seed[i]
            cur_init_state = init_state[i]
            cur_actions = actions[i]
            obs, state = env.rollout(cur_seed, cur_init_state, cur_actions)
            obses.append(obs)
            states.append(state)
        obses = aggregate_dct(obses)
        states = np.stack(states)
        return obses, states
