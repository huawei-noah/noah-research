"""Guard the SDK -> eval package boundary.

The SDK must not depend on or reference the evaluation package: the dependency
direction is eval -> sdk only. These tests pin that invariant so the lazy-load
leftover from the package split (``from mindmemos_sdk import eval``) cannot creep
back in.
"""

from __future__ import annotations

import sys

import pytest

import mindmemos_sdk

# Names that used to be re-exported from the now-removed ``mindmemos_sdk.eval``
# subpackage. They live in the standalone ``mindmemos_eval`` package now.
_FORMER_EVAL_EXPORTS = [
    "Env",
    "AnswerResult",
    "EvalResult",
    "LLMClient",
    "LLMConfig",
    "Scorer",
    "ScoreResult",
    "ExactMatchScorer",
    "LLMJudgeScorer",
    "LocomoEnv",
    "LocomoLLMJudgeScorer",
    "LocomoRunResult",
]


def test_sdk_imports_without_eval_package():
    """Importing the SDK must not pull in the eval package."""
    assert "mindmemos_eval" not in sys.modules
    assert mindmemos_sdk.__version__


@pytest.mark.parametrize("name", _FORMER_EVAL_EXPORTS)
def test_eval_names_are_not_exposed_by_sdk(name):
    """Eval symbols must not be reachable from the SDK namespace."""
    assert name not in mindmemos_sdk.__all__
    with pytest.raises(AttributeError):
        getattr(mindmemos_sdk, name)


def test_sdk_has_no_eval_submodule():
    """The removed ``mindmemos_sdk.eval`` subpackage must stay removed."""
    with pytest.raises(ImportError):
        __import__("mindmemos_sdk.eval")
