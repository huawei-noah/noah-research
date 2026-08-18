from collections.abc import Iterable
from typing import Any

import sacrebleu


class CHRF:
    def __init__(
        self,
    ):
        pass

    def __call__(self, doc, ground_truth, results) -> Any:
        try:
            results = [(ground_truth, results[0])]
            return self.chrf(results)
        except:
            return 0

    def chrf(self, items):
        """chrF++ is a tool for automatic evaluation of machine translation output
        based on character n-gram precision and recall enhanced with word n-grams.
        Source: https://github.com/m-popovic/chrF
        Paper: https://www.aclweb.org/anthology/W15-3049.pdf

        Higher is better  # TODO I think
        """
        refs = list(zip(*items))[0]
        preds = list(zip(*items))[1]
        refs, preds = self._sacreformat(refs, preds)
        return sacrebleu.corpus_chrf(preds, refs).score

    def _sacreformat(self, refs, preds):
        """Format refs and preds for sacrebleu corpus calculation. It is very particular"""

        if not self.is_non_str_iterable(refs):
            refs = list(refs)
        if not self.is_non_str_iterable(refs[0]):
            refs = [[ref] for ref in refs]
        refs = list(zip(*refs))

        if not self.is_non_str_iterable(preds):
            preds = list(preds)
        if self.is_non_str_iterable(preds[0]):
            assert len(preds[0]) == 1, f"Pred must be a str, was {preds[0]}"
            preds = [pred[0] for pred in preds]

        return refs, preds

    def is_non_str_iterable(self, obj):
        return isinstance(obj, Iterable) and not isinstance(obj, str)
