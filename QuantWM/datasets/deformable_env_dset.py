import yaml
import torch
import decord
import numpy as np
from einops import rearrange
from pathlib import Path
from typing import Callable, Optional
from .traj_dset import TrajDataset, get_train_val_sliced
decord.bridge.set_bridge("torch")

def load_yaml(filename):
    # load YAML file
    return yaml.safe_load(open(filename, "r"))

class DeformDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "data/deformable",
        object_name: str = "rope",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale=1.0,
    ):
        self.data_path = Path(data_path) / object_name
        self.transform = transform
        self.normalize_action = normalize_action
        self.states = torch.load(
            self.data_path / "states.pth"
        ).float()  # (n_rollout, n_timestep, n_particles, 4)
        self.states = rearrange(self.states, "N T P D -> N T (P D)")

        self.actions = torch.load(self.data_path / "actions.pth").float()
        self.actions = self.actions / action_scale  # scaled back up in env

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]

        self.seq_lengths = torch.tensor([self.states.shape[1]] * len(self.states))

        self.proprios = torch.zeros(
            (self.states.shape[0], self.states.shape[1], 1)
        )  # dummy proprio
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action: 
            self.action_mean, self.action_std = self.get_data_mean_std(
                self.actions, self.seq_lengths
            )
            self.state_mean, self.state_std = self.get_data_mean_std(
                self.states, self.seq_lengths
            )
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(
                self.proprios, self.seq_lengths
            )
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std


    def get_data_mean_std(self, data, traj_lengths):
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = traj_lengths[traj]
            traj_data = data[traj, :traj_len]
            all_data.append(traj_data)
        all_data = torch.vstack(all_data)
        data_mean = torch.mean(all_data, dim=0)
        data_std = torch.std(all_data, dim=0) + 1e-6
        return data_mean, data_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        obs_dir = self.data_path / f"{idx:06d}"
        image = torch.load(obs_dir / "obses.pth")
        proprio = self.proprios[idx, frames]
        act = self.actions[idx, frames]
        state = self.states[idx, frames]

        image = image[frames]  # THWC
        image = rearrange(image, "T H W C -> T C H W") / 255.0
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {} # infos is None

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w")


def load_deformable_dset_slice_train_val(
    transform,
    n_rollout=50,
    data_path="data/deformable",
    object_name="rope",
    normalize_action=False,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
):
    dset = DeformDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        object_name=object_name,
        normalize_action=normalize_action,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
    )

    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    traj_dset = {}
    traj_dset["train"] = dset_train
    traj_dset["valid"] = dset_val
    return datasets, traj_dset
