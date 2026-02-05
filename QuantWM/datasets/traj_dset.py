import abc
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Sequence, List
from torch.utils.data import Dataset, Subset
from torch import default_generator, randperm
from einops import rearrange

# https://github.com/JaidedAI/EasyOCR/issues/1243
def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total

class TrajDataset(Dataset, abc.ABC):
    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

class TrajSubset(TrajDataset, Subset):
    """
    Subset of a trajectory dataset at specified indices.

    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset: TrajDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    def __getattr__(self, name):
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class TrajSlicerDataset(TrajDataset):
    def __init__(
        self,
        dataset: TrajDataset,
        num_frames: int,
        frameskip: int = 1,
        process_actions: str = "concat",
    ):
        self.dataset = dataset
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.slices = []
        for i in range(len(self.dataset)): 
            T = self.dataset.get_seq_length(i)
            if T - num_frames < 0:
                print(f"Ignored short sequence #{i}: len={T}, num_frames={num_frames}")
            else:
                self.slices += [
                    (i, start, start + num_frames * self.frameskip)
                    for start in range(T - num_frames * frameskip + 1)
                ]  # slice indices follow convention [start, end)
        # randomly permute the slices
        self.slices = np.random.permutation(self.slices)
        
        self.proprio_dim = self.dataset.proprio_dim
        if process_actions == "concat":
            self.action_dim = self.dataset.action_dim * self.frameskip
        else:
            self.action_dim = self.dataset.action_dim

        self.state_dim = self.dataset.state_dim


    def get_seq_length(self, idx: int) -> int:
        return self.num_frames

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        obs, act, state, _ = self.dataset[i]
        for k, v in obs.items():
            obs[k] = v[start:end:self.frameskip]
        state = state[start:end:self.frameskip]
        act = act[start:end]
        act = rearrange(act, "(n f) d -> n (f d)", n=self.num_frames)  # concat actions
        return tuple([obs, act, state])


def random_split_traj(
    dataset: TrajDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
) -> List[TrajSubset]:
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    print(
        [
            indices[offset - length : offset]
            for offset, length in zip(_accumulate(lengths), lengths)
        ]
    )
    return [
        TrajSubset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set


def get_train_val_sliced(
    traj_dataset: TrajDataset,
    train_fraction: float = 0.9,
    random_seed: int = 42,
    num_frames: int = 10,
    frameskip: int = 1,
):
    train, val = split_traj_datasets(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    train_slices = TrajSlicerDataset(train, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val, num_frames, frameskip)
    return train, val, train_slices, val_slices