from types import SimpleNamespace

import pytest
from mindmemos.api import mappers as api_mappers
from mindmemos.api.schemas import SearchRequest
from mindmemos.config import SearchConfig


def test_search_patches_defaults_to_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    search_config = SearchConfig()
    monkeypatch.setattr(
        api_mappers,
        "get_config",
        lambda: SimpleNamespace(algo_config=SimpleNamespace(search=search_config)),
    )

    inp = api_mappers.to_search_pipeline_input(
        SearchRequest(user_id="u1", query="Qdrant"),
        search_pipline="vanilla",
    )

    assert search_config.include_patches is True
    assert inp.include_patches is True
