import itertools

import numpy as np


class Mean:
    def __init__(self):
        pass

    def __call__(self, individual_values):
        individual_values = np.array([float(x) for x in individual_values])
        pass_rate = individual_values.mean()
        return pass_rate


class PassK:
    def __init__(
        self,
    ):
        self.k = [1, 2, 10, 25, 50, 100]

    def __call__(self, individual_values):
        num_samples = []
        num_correct = []
        num_samples.append(len(individual_values))
        num_correct.append(sum(individual_values))
        num_samples = np.array(num_samples)
        num_correct = np.array(num_correct)
        return {
            f"pass@{k}": self.estimate_pass_at_k(num_samples, num_correct, k).mean()
            for k in self.k
            if (num_samples >= k).all()
        }

    def estimate_pass_at_k(self, num_samples, num_correct, k: int) -> np.ndarray:
        """
        Estimates pass@k of each problem and returns them in an array.
        """

        def estimator(n: int, c: int, k: int) -> float:
            """
            Calculates 1 - comb(n - c, k) / comb(n, k).
            """
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

        if isinstance(num_samples, int):
            num_samples_it = itertools.repeat(num_samples, len(num_correct))
        else:
            assert len(num_samples) == len(num_correct)
            num_samples_it = iter(num_samples)

        return np.array(
            [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
        )


AGGREGATOR_REGISTRY = {"pass_k": PassK, "mean": Mean}


def get_aggregator(aggregator_name):
    return AGGREGATOR_REGISTRY[aggregator_name]
