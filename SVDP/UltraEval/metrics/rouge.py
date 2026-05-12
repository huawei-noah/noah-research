from typing import Any

from rouge_chinese import Rouge
from sacrebleu.metrics.bleu import _TOKENIZERS, _get_tokenizer


class ROUGE:
    def __init__(self, **kwargs):
        self.tokenizer = kwargs.get("tokenizer", "13a")
        self.index = kwargs.get("index", "f1")[0].lower()
        self.n_gram = kwargs.get("n_gram", "1")[0].lower()
        self.METRIC_NAME = f"rouge-{self.n_gram}"
        assert self.tokenizer in _TOKENIZERS, "invalid tokenizer name"
        assert self.index in ("f", "p", "r"), "invalid index name"
        assert self.n_gram in ("1", "2", "l"), "invalid index name"

    def __call__(self, doc, ground_truth, results) -> Any:
        try:
            results = [(ground_truth, results[0])]
            return self.rouge(results, self.METRIC_NAME, self.tokenizer, self.index)
        except:
            return 0

    def rouge(self, items, rouge_type, tokenizer_type="13a", index="f", avg=True):
        """
        Source:
        Paper:

        Higher is better
        """
        scorer = Rouge()
        tokenizer = _get_tokenizer(tokenizer_type)()
        refs = list(zip(*items))[0]
        preds = list(zip(*items))[1]
        if isinstance(refs[0], tuple) or isinstance(refs[0], list):
            refs = tuple(zip(*refs))
            ret = []
            for i, (pred, ref_list) in enumerate(zip(preds, refs)):
                scores_of_refs = []
                for ref in ref_list:
                    pred, ref = tokenizer(pred), tokenizer(ref)
                    scores_of_refs.append(
                        scorer.get_scores(pred, ref, avg)[rouge_type][index]
                    )
                ret.append(max(scores_of_refs))

            score = self.mean(ret)
        else:
            preds, refs = tuple(tokenizer(pred) for pred in preds), tuple(
                tokenizer(ref) for ref in refs
            )
            score = scorer.get_scores(preds, refs, avg)[rouge_type][index]
        return 100.0 * score

    def mean(self, arr):
        return sum(arr) / len(arr)
