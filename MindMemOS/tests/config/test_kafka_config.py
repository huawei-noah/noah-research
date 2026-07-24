import pytest
from mindmemos.config import build_config, get_config, init_config, reset_config
from mindmemos.errors import InvalidConfigError


def test_dev_example_config_includes_async_add_consumers() -> None:
    try:
        init_config(config_path="config/mindmemos/dev.example.yaml")

        consumers = get_config().kafka.consumers

        topics = get_config().kafka.topics
        assert get_config().kafka.global_max_concurrency == 80
        assert any(topic.name == "memory.add" and topic.partitions == 16 for topic in topics)
        assert any(
            consumer.group_id == "memory-add-worker" and consumer.topics == ["memory.add"] for consumer in consumers
        )
        add_consumer = next(consumer for consumer in consumers if consumer.group_id == "memory-add-worker")
        assert add_consumer.session_timeout_ms == 180000
        assert add_consumer.heartbeat_interval_ms == 10000
        assert add_consumer.max_poll_interval_ms == 600000
        assert any(
            consumer.group_id == "schema-add-drain-worker" and consumer.topics == ["memory.add.drain"]
            for consumer in consumers
        )
        assert any(
            consumer.group_id == "schema-add-episode-worker" and consumer.topics == ["memory.add.episode"]
            for consumer in consumers
        )
        assert any(
            consumer.group_id == "memory-feedback-worker" and consumer.topics == ["memory.feedback"]
            for consumer in consumers
        )
    finally:
        reset_config()


def test_endpoint_envs_override_yaml(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MINDMEMOS_QDRANT_URL", "http://qdrant.example:16333")
    monkeypatch.setenv("MINDMEMOS_QDRANT_GRPC_PORT", "16334")
    monkeypatch.setenv("MINDMEMOS_QDRANT_PREFER_GRPC", "true")
    monkeypatch.setenv("MINDMEMOS_NEO4J_URI", "bolt://neo4j.example:17687")
    monkeypatch.setenv("MINDMEMOS_KAFKA_BOOTSTRAP_SERVERS", "kafka.example:19092")
    monkeypatch.setenv("MINDMEMOS_TELEMETRY_ENDPOINT", "http://otel.example:14318")
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
telemetry:
  enabled: true
  telemetry_endpoint: http://yaml.example:4318
database:
  qdrant:
    url: http://yaml.example:6333
    grpc_port: 6334
    prefer_grpc: false
  neo4j:
    uri: bolt://yaml.example:7687
kafka:
  enabled: true
  bootstrap_servers: yaml.example:9092
""",
        encoding="utf-8",
    )

    cfg = build_config(config_path=config_path)

    assert cfg.database.qdrant.url == "http://qdrant.example:16333"
    assert cfg.database.qdrant.grpc_port == 16334
    assert cfg.database.qdrant.prefer_grpc is True
    assert cfg.database.neo4j.uri == "bolt://neo4j.example:17687"
    assert cfg.kafka.bootstrap_servers == "kafka.example:19092"
    assert cfg.telemetry.telemetry_endpoint == "http://otel.example:14318"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("global_max_concurrency", 0),
        ("per_key_max_concurrency", -2),
        ("max_buffered", 0),
        ("session_timeout_ms", 0),
        ("heartbeat_interval_ms", 0),
        ("max_poll_interval_ms", 0),
    ],
)
def test_kafka_consumer_concurrency_limits_reject_non_positive_values(tmp_path, field, value) -> None:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        f"""
kafka:
  consumers:
    - topics: ["memory.add"]
      group_id: "memory-add-worker"
      {field}: {value}
""",
        encoding="utf-8",
    )
    try:
        with pytest.raises(InvalidConfigError, match=field):
            init_config(config_path=config_path)
    finally:
        reset_config()


def test_kafka_global_concurrency_rejects_negative_value(tmp_path) -> None:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
kafka:
  global_max_concurrency: -1
""",
        encoding="utf-8",
    )
    try:
        with pytest.raises(InvalidConfigError, match="global_max_concurrency"):
            init_config(config_path=config_path)
    finally:
        reset_config()


def test_kafka_topic_partitions_reject_non_positive_values(tmp_path) -> None:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
kafka:
  topics:
    - name: "memory.add"
      partitions: 0
""",
        encoding="utf-8",
    )
    try:
        with pytest.raises(InvalidConfigError, match="partitions"):
            init_config(config_path=config_path)
    finally:
        reset_config()
