from collections.abc import Iterable
from typing import Any

import sacrebleu


class BLEU:
    def __init__(self, **kwargs):
        self.tokenizer = kwargs.get("tokenizer", "13a")

    def __call__(self, doc, ground_truth, results) -> Any:
        try:
            results = [(ground_truth, results[0])]
            return self.bleu(results, tokenizer=self.tokenizer)
        except:
            return 0

    def bleu(self, items, tokenizer="13a"):
        """The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric
        for evaluating a generated sentence to a reference sentence. It counts matching
        n-grams in the candidate translation to n-grams in the reference text, where
        1-gram or unigram would be each token and a bigram comparison would be each
        word pair. The comparison is made regardless of word order
        Source: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
        Paper: https://www.aclweb.org/anthology/P02-1040/

        Higher is better
        """
        refs = list(zip(*items))[0]
        preds = list(zip(*items))[1]
        refs, preds = self._sacreformat(refs, preds)
        return sacrebleu.corpus_bleu(preds, refs, tokenize=tokenizer).score

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
