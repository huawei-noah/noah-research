SCORE_FUNCTION_REGISTRY = {}


def register_score(name):
    def register_score_cls(cls):
        if name in SCORE_FUNCTION_REGISTRY:
            raise ValueError(f"Duplicate score function {name} appeared!")
        SCORE_FUNCTION_REGISTRY[name] = cls
        return cls
    return register_score_cls


from lib.score_functions.logits_score import *
from lib.score_functions.mahalanobis_score import *
