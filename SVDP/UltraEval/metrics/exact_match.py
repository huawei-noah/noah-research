from typing import Any


class ExactMatch:
    def __init__(
        self,
    ):
        pass

    def __call__(self, doc, ground_truth, results) -> Any:
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]

        if len(results[0].strip()) == 0:
            return 0
        scores_for_ground_truths = []
        for single_ground_truth in ground_truth:
            score = 1 if results[0] == single_ground_truth else 0
            score = 1 if results[0].replace(" ", "") == single_ground_truth.replace(" ", "") else 0
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)
