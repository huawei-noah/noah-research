from collections import Counter
from typing import Any


class F1Score:
    def __init__(
        self,
    ):
        pass

    def __call__(self, doc, ground_truth, results) -> Any:
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]

        f1_scores = [
            self.f1_score(results[0], single_answer) for single_answer in ground_truth
        ]

        return max(f1_scores, default=0.0)

    def f1_score(self, prediction, ground_truth):
        prediction_tokens = prediction.split()
        ground_truth_tokens = ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
