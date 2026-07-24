from mindmemos.api.deps import resolve_api_key

from mindmemos.config import init_config, reset_config


def test_resolve_api_key_uses_configured_api_key_project(tmp_path):
    api_key_path = tmp_path / "api_keys.yaml"
    api_key_path.write_text(
        """
api_keys:
  - key_id: key-test
    api_key: test-api-key
    project_id: proj-test
    memory_algorithm: vanilla
    enabled: true
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        f"""
auth:
  mode: api_key
  api_key_file: {api_key_path}
""",
        encoding="utf-8",
    )

    try:
        init_config(config_path=config_path)

        resolved = resolve_api_key("test-api-key")

        assert resolved.project_id == "proj-test"
        assert resolved.api_key_uuid == "key-test"
        assert resolved.account_id == "memory_standalone"
        assert resolved.memory_algorithm == "vanilla"
    finally:
        reset_config()
