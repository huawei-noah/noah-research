import numpy as np
import pyflex

from .scenes import *


class FlexScene:
    def __init__(self):
        self.obj = None
        self.env_idx = None
        self.scene_params = None

        self.property_params = None
        self.clusters = None

    def set_scene(self, obj, obj_params):
        self.obj = obj

        if self.obj == "rope":
            self.env_idx = 26
            self.scene_params, self.property_params = rope_scene(obj_params)
        elif self.obj == "granular":
            self.env_idx = 35
            self.scene_params, self.property_params = granular_scene(obj_params)
        elif self.obj == "cloth":
            self.env_idx = 29
            self.scene_params, self.property_params = cloth_scene(obj_params)
        else:
            raise ValueError("Unknown Scene.")

        assert self.env_idx is not None
        assert self.scene_params is not None
        zeros = np.array([0])
        pyflex.set_scene(self.env_idx, self.scene_params, zeros, zeros, zeros, zeros, 0)

    def get_property_params(self):
        assert self.property_params is not None
        return self.property_params
