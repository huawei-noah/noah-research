import pytest
from mindmemos.config import init_config, reset_config
from mindmemos.errors import InvalidConfigError, MissingConfigValueError


def test_gateway_jwt_requires_secret_during_config_init(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MINDMEMOS_GATEWAY_JWT_SECRET", "env-secret")
    monkeypatch.setenv("MINDMEMOS_GATEWAY_INTERNAL_SECRET", "legacy-env-secret")
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
auth:
  mode: gateway_jwt
""",
        encoding="utf-8",
    )

    try:
        with pytest.raises(MissingConfigValueError, match="auth.gateway_jwt_secret"):
            init_config(config_path=config_path)
    finally:
        reset_config()


def test_telemetry_requires_endpoint_during_config_init(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MINDMEMOS_TELEMETRY_ENDPOINT", "")
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
telemetry:
  enabled: true
  telemetry_endpoint:
""",
        encoding="utf-8",
    )

    try:
        with pytest.raises(MissingConfigValueError, match="telemetry.telemetry_endpoint"):
            init_config(config_path=config_path)
    finally:
        reset_config()


def test_embedding_dimensions_must_match_qdrant_vector_size(tmp_path) -> None:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
embed_model_router:
  endpoints:
    - model: text-embedding-test
      api_key: sk-test
      api_base: http://example.test/v1
      dimensions: 128
database:
  qdrant:
    vector_size: 256
""",
        encoding="utf-8",
    )

    try:
        with pytest.raises(InvalidConfigError, match="dimensions"):
            init_config(config_path=config_path)
    finally:
        reset_config()


def test_vanilla_add_chunk_budget_must_leave_extractable_space(tmp_path) -> None:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
algo_config:
  add:
    vanilla:
      chunk_soft_token_budget: 8000
      chunk_hard_token_budget: 10000
""",
        encoding="utf-8",
    )

    try:
        with pytest.raises(InvalidConfigError, match="chunk_hard_token_budget"):
            init_config(config_path=config_path)
    finally:
        reset_config()
