from __future__ import annotations

from mindmemos.pipelines.search.agentic.wrapper import _agentic_config_with_max_rounds
from omegaconf import OmegaConf

from mindmemos.config.algo.search import AgenticConfig


def test_agentic_config_with_max_rounds_supports_dataclass_config() -> None:
    config = AgenticConfig(max_rounds=5, top_k_per_round=12)

    updated = _agentic_config_with_max_rounds(config, 2)

    assert updated.max_rounds == 2
    assert updated.top_k_per_round == 12
    assert config.max_rounds == 5


def test_agentic_config_with_max_rounds_supports_omegaconf_dict_config() -> None:
    config = OmegaConf.create({"max_rounds": 5, "top_k_per_round": 12})

    updated = _agentic_config_with_max_rounds(config, 2)

    assert updated.max_rounds == 2
    assert updated.top_k_per_round == 12
    assert config.max_rounds == 5


def test_agentic_config_with_max_rounds_supports_plain_dict_config() -> None:
    config = {"max_rounds": 5, "top_k_per_round": 12}

    updated = _agentic_config_with_max_rounds(config, 2)

    assert updated.max_rounds == 2
    assert updated.top_k_per_round == 12
    assert config["max_rounds"] == 5
