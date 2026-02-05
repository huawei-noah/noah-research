from enum import Enum, auto
from .single import DotDataset, DotDatasetConfig, Sample
from .wall import WallDataset, WallDatasetConfig
from .configs import ConfigBase

class DatasetType(Enum):
    Single = auto()
    Multiple = auto()
    Wall = auto()
    WallExpert = auto()
    WallEigenfunc = auto()
