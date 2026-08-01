from typing import Any


class InMatch:
    def __init__(
        self,
    ):
        pass

    def __call__(self, doc, ground_truth, results) -> Any:
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]

        return 1.0 if results[0].lower().strip() in ground_truth else 0.0
