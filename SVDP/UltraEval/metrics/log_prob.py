import hashlib
from typing import Any

import numpy as np


class LogProb:
    def __init__(
        self,
    ):
        pass

    def __call__(self, doc, ground_truth, results) -> Any:
        lowest_score_index = self._argmin(results[0])
        if lowest_score_index is None:
            res = 0
        else:
            keys_list = list(doc["target_scores"].keys())
            target_key = keys_list[lowest_score_index]
            res = doc["target_scores"][target_key]
        return res

    def _argmin(self, array):
        """argmin with deterministic pseudorandom tie breaking."""
        if all(val == '' or np.isnan(val) for val in array):
            print("WARNING: all val are none in array")
            return None
        array = np.where(np.isnan(array), np.inf, array)
        min_indices = np.arange(len(array))[array == np.min(array)]
        idx = int(hashlib.sha256(np.asarray(array).tobytes()).hexdigest(), 16) % len(
            min_indices
        )
        return min_indices[idx]
