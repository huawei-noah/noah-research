from typing import Any

import numpy as np


class LogProbMC2:
    def __init__(
        self,
    ):
        pass

    def __call__(self, doc, ground_truth, results) -> Any:
        """Take a single document and the LM input/output/ground_truth.
        Returns the  values of the metric for that one document
        """
        split_idx = list(doc["target_scores"].values()).index(0)
        correct_probs = results[0][:split_idx]
        log_prob_mc2 = np.exp(-np.array(correct_probs)) / sum(
            np.exp(-np.array(results[0]))
        )
        return sum(log_prob_mc2)
