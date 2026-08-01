from typing import Any


class GaoKaoBenchMatch:
    def __init__(
        self,
    ):
        pass

    def __call__(self, doc, ground_truth, results) -> Any:
        """Take a single document and the LM input/output/ground_truth.
        Returns the  values of the metric for that one document
        """
        score = doc["score"]
        total_score = 0.0
        correct_score = 0.0
        ans_len = len(ground_truth)
        total_score += ans_len * score
        if "[SEP]" in results[0]:
            text1, text2 = results[0].split("[SEP]")
            if len(text1) == ans_len:
                results = [text1]
            else:
                results = [text2[:ans_len]]
        for j in range(min(len(results[0]), len(ground_truth))):
            if results[0][j] == ground_truth[j]:
                correct_score += score
        return correct_score / total_score
