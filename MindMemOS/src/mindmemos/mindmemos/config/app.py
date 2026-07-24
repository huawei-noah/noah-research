import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import yaml
from dotenv import load_dotenv

from ..logging import get_logger
from .algo import MemoryAlgoConfig
from .base import build, frozen_field, secret_field
from .validation import validate_config

T = TypeVar("T")
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_ROOT = REPO_ROOT / "config"
DEFAULT_MINDMEMOS_CONFIG_ROOT = DEFAULT_CONFIG_ROOT / "mindmemos"
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
logger = get_logger(__name__)

LLM_CONFIG_MAP_MUST_EXIST = "default"
EMBEDDING_CONFIG_MAP_MUST_EXIST = "default"
DEFAULT_API_KEY_FILE = "api_keys.yaml"


def default_config_path(config_name: str) -> Path:
    if config_name == "dev":
        return DEFAULT_MINDMEMOS_CONFIG_ROOT / "dev.yaml"
    return DEFAULT_CONFIG_ROOT / f"{config_name}.yaml"


def _env_str(name: str, default: str | None = None) -> str | None:
    value = _getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    value = _getenv(name)
    if value is None or value == "":
        return default
    return _truthy(value)


def _env_int(name: str, default: int) -> int:
    value = _env_str(name)
    return int(value) if value is not None else default


def _truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _merge_database_env(data: dict) -> dict:
    """Overlay deployment-coupled connection settings from the environment.

    Only connection endpoints and credentials stay env-configurable so they can
    align with docker-compose ports / ``NEO4J_AUTH`` and with deploy-injected
    secrets. All other database tuning is yaml-only and must be set in the config
    file.
    """
    database = data.setdefault("database", {})
    qdrant = database.setdefault("qdrant", {})
    neo4j = database.setdefault("neo4j", {})

    _set_if_env(qdrant, "url", "MINDMEMOS_QDRANT_URL")
    _set_if_env(qdrant, "api_key", "MINDMEMOS_QDRANT_API_KEY")
    _set_if_env(qdrant, "grpc_port", "MINDMEMOS_QDRANT_GRPC_PORT", int)
    _set_if_env(qdrant, "prefer_grpc", "MINDMEMOS_QDRANT_PREFER_GRPC", _truthy)
    if "collection" in qdrant and "memory_collection" not in qdrant:
        qdrant["memory_collection"] = qdrant.pop("collection")

    _set_if_env(neo4j, "uri", "MINDMEMOS_NEO4J_URI")
    _set_if_env(neo4j, "username", "MINDMEMOS_NEO4J_USERNAME")
    _set_if_env(neo4j, "password", "MINDMEMOS_NEO4J_PASSWORD")
    return data


def _merge_kafka_env(data: dict) -> dict:
    kafka = data.setdefault("kafka", {})
    _set_if_env(kafka, "bootstrap_servers", "MINDMEMOS_KAFKA_BOOTSTRAP_SERVERS")
    return data


def _merge_telemetry_env(data: dict) -> dict:
    telemetry = data.setdefault("telemetry", {})
    _set_if_env(telemetry, "telemetry_endpoint", "MINDMEMOS_TELEMETRY_ENDPOINT")
    return data


def _normalize_auth_paths(data: dict, config_path: Path) -> dict:
    auth = data.setdefault("auth", {})
    if not isinstance(auth, dict):
        return data

    if "gateway_internal_secret" in auth and "gateway_jwt_secret" not in auth:
        auth["gateway_jwt_secret"] = auth.pop("gateway_internal_secret")
    if "gateway_internal_issuer" in auth and "gateway_jwt_issuer" not in auth:
        auth["gateway_jwt_issuer"] = auth.pop("gateway_internal_issuer")
    if "gateway_internal_audience" in auth and "gateway_jwt_audience" not in auth:
        auth["gateway_jwt_audience"] = auth.pop("gateway_internal_audience")

    api_key_file = auth.get("api_key_file") or DEFAULT_API_KEY_FILE

    path = Path(str(api_key_file)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    auth["api_key_file"] = str(path.resolve())
    return data


def _set_if_env(target: dict, key: str, env_name: str, caster=None) -> None:
    value = _getenv(env_name)
    if value is None or value == "":
        return
    target[key] = caster(value) if caster else value


def _getenv(name: str) -> str | None:
    return os.getenv(name)


@dataclass
class TelemetryConfig:
    enabled: bool = field(default=False)
    """Whether OpenTelemetry recording is enabled."""

    telemetry_endpoint: str | None = field(
        default_factory=lambda: _env_str("MINDMEMOS_TELEMETRY_ENDPOINT", "http://localhost:4318")
    )
    """Telemetry endpoint URL as an OTLP/HTTP base, such as http://localhost:4318.
        Trace and log signal paths are completed as /v1/traces and /v1/logs.
    """

    service_name: str = field(default="mindmemos")
    """Service name reported to the backend as OTel resource service.name.
        This separates source services in ClickHouse and Grafana.
    """

    telemetry_timeout: int = field(default=5)
    """Telemetry export timeout in seconds."""

    span_type: str = field(default="simple")
    """Span processor type. Use simple only in development; use batch in production."""

    logs_enabled: bool = field(default=True)
    """Whether to export application logs through OTLP using the same telemetry endpoint.
        When disabled, @traced and structlog logs stay on the local console only.
    """

    trace_sampling_ratio: float = field(default=1.0)
    """Trace sampling ratio."""

    max_queue_size: int = field(default=10240)
    """Maximum trace queue length."""

    max_export_batch_size: int = field(default=1024)
    """Trace batch size"""

    log_level: str = field(default="INFO")
    """Telemetry log export level."""

    metric_export_interval_millis: int = field(default=5000)
    """Metric export interval in seconds."""


@dataclass
class ModelEndpointConfig:
    """Configuration for one model endpoint aligned with LiteLLM router parameters."""

    model: str
    """LiteLLM model identifier, such as ``openai/doubao-seed-2-0-mini-260428``."""

    api_key: str
    """Model API key."""

    api_base: str
    """Model API base URL."""

    rpm: int | None = None
    """Allowed requests per minute."""

    tpm: int | None = None
    """Allowed tokens per minute."""

    timeout: int = 600
    """Per-request timeout in seconds."""

    num_retries: int = 50
    """Maximum retries for one endpoint, matching the LiteLLM field name."""

    temperature: float | None = None
    """Sampling temperature."""

    top_p: float | None = None
    """nucleus sampling"""

    max_tokens: int | None = None
    """Maximum generated token count."""

    max_completion_tokens: int | None = None
    """Maximum completion token count for newer OpenAI APIs."""

    encoding_format: str | None = None
    """Embedding response encoding format, such as ``float`` or ``base64``."""

    dimensions: int | None = None
    """Embedding dimensions when supported by the provider."""

    extra_body: dict = field(default_factory=dict)
    """Extra request body fields passed through to the underlying SDK for non-standard provider options.
        Examples include Doubao ``thinking`` and Qwen ``enable_thinking``.
    """


@dataclass
class ModelRouterConfig:
    endpoints: list[ModelEndpointConfig] = field(default_factory=list)
    """Model endpoint configuration list."""

    routing_strategy: str = "simple-shuffle"
    """Routing strategy, such as simple-shuffle or least-busy."""

    allowed_fails: int | None = None

    cool_down: int | float | None = None

    format_parser_max_attempts: int = 3
    """Maximum chat generations when format_parser rejects model output."""

    dimensions_supported_models: list[str] = field(default_factory=list)
    """Model name prefixes (provider prefix stripped) known to support the
    ``dimensions`` embedding request param. litellm otherwise drops
    ``dimensions`` for any openai-compatible model whose name lacks the literal
    ``text-embedding-3``; listing a model here tells litellm (via
    ``allowed_openai_params``) to keep it. Only meaningful for embedding routers."""


@dataclass
class QdrantConfig:
    url: str = field(default_factory=lambda: _env_str("MINDMEMOS_QDRANT_URL", "http://localhost:6333"))
    """Qdrant HTTP endpoint"""

    api_key: str | None = secret_field(default_factory=lambda: _env_str("MINDMEMOS_QDRANT_API_KEY"))
    """Qdrant API key, usually not required for local development."""

    grpc_port: int = field(default_factory=lambda: _env_int("MINDMEMOS_QDRANT_GRPC_PORT", 6334))
    """Qdrant gRPC port."""

    prefer_grpc: bool = field(default_factory=lambda: _env_bool("MINDMEMOS_QDRANT_PREFER_GRPC", False))
    """Whether the Qdrant client should prefer gRPC for supported operations."""

    memory_collection: str = field(default="memory_item_v1")
    """Qdrant memory item collection name"""

    entity_collection: str = field(default="entity_item_v1")
    """Qdrant entity item collection name"""

    source_collection: str = field(default="source_item_v1")
    """Qdrant source item collection name"""

    add_record_collection: str = field(default="add_record_v1")
    """Qdrant add request/response record collection name"""

    schema_add_buffer_collection: str = field(default="schema_add_buffer_v1")
    """Qdrant schema add durable buffer collection name"""

    search_record_collection: str = field(default="search_record_v1")
    """Qdrant search request/response record collection name"""

    skill_version_collection: str = field(default="skill_version_v1")
    """Qdrant skill version metadata collection name"""

    skill_blob_collection: str = field(default="skill_blob_v1")
    """Qdrant skill bundle content (dedup) collection name"""

    skill_trace_pending_collection: str = field(default="skill_trace_pending_v1")
    """Qdrant pending skill-trace collection name"""

    skill_trace_summary_collection: str = field(default="skill_trace_summary_v1")
    """Qdrant skill trajectory-summary collection name (self-evolution input)"""

    semantic_vector_name: str = field(default="semantic")
    """Dense semantic vector name."""

    bm25_vector_name: str = field(default="bm25")
    """Sparse BM25 vector name."""

    vector_size: int = field(default=1024)
    """Semantic embedding dimension."""

    distance: str = field(default="Cosine")
    """Dense vector distance function: Cosine, Euclid, Dot, or Manhattan."""

    auto_create: bool = field(default=True)
    """Whether to create collections and payload indexes at startup."""

    memory_on_disk_payload: bool = field(default=False)
    """Whether memory collection payloads are stored on disk instead of memory."""

    timeout: float = field(default=10.0)
    """Qdrant client timeout in seconds."""

    pool_size: int = field(default=100)
    """Qdrant HTTP connection pool size. Zero uses the client default."""

    max_client_concurrency: int = field(default=100)
    """Client-side maximum concurrent requests sent to Qdrant."""

    max_client_concurrency_cap: int = field(default=64)
    """Configured safety cap for Qdrant client concurrency; values <= 0 disable capping."""

    request_read_budget: int = field(default=100)
    """Default per-request Qdrant point-read budget for fanout-prone readers."""

    max_retries: int = field(default=5)
    """Maximum Qdrant request attempts before surfacing an error."""

    retry_base_delay: float = field(default=0.1)
    """Base delay in seconds for Qdrant exponential backoff between retries."""

    batch_upsert_enabled: bool = field(default=False)
    """Whether Qdrant upserts should be collected into short-lived client-side batches."""

    batch_upsert_size: int = field(default=128)
    """Maximum points per Qdrant upsert batch."""

    batch_upsert_flush_interval_ms: int = field(default=25)
    """Maximum time to wait before flushing a non-empty Qdrant upsert batch."""

    batch_upsert_max_queue_size: int = field(default=10000)
    """Maximum pending Qdrant upsert requests before producers apply backpressure."""

    batch_upsert_max_inflight_batches: int = field(default=32)
    """Maximum Qdrant upsert batches in flight at once."""


@dataclass
class Neo4jConfig:
    uri: str = field(default_factory=lambda: _env_str("MINDMEMOS_NEO4J_URI", "bolt://localhost:7687"))
    """Neo4j Bolt URI"""

    username: str = field(default_factory=lambda: _env_str("MINDMEMOS_NEO4J_USERNAME", "neo4j"))
    """Neo4j username."""

    password: str = secret_field(default_factory=lambda: _env_str("MINDMEMOS_NEO4J_PASSWORD", "mindmemos_dev_password"))
    """Neo4j password."""

    database: str = field(default="neo4j")
    """Neo4j database name"""

    auto_create_schema: bool = field(default=True)
    """Whether to create constraints and indexes at startup."""

    max_connection_lifetime: int = field(default=3600)
    """Maximum connection lifetime in seconds."""

    max_connection_pool_size: int = field(default=100)
    """Bolt connection pool size and concurrency limit."""

    connection_acquisition_timeout: float = field(default=60.0)
    """Maximum seconds to wait for a pool connection before raising instead of queueing forever."""

    max_client_concurrency: int = field(default=100)
    """Client-side maximum concurrent requests sent to Neo4j."""

    max_client_concurrency_cap: int = field(default=64)
    """Configured safety cap for Neo4j client concurrency; values <= 0 disable capping."""

    request_row_budget: int = field(default=100)
    """Default per-request Neo4j row budget for fanout-prone graph reads."""

    write_max_retries: int = field(default=5)
    """Internal Neo4j write transient error (deadlock etc.) retry attempts"""

    write_retry_base_delay: float = field(default=0.1)
    """Base delay in seconds for exponential backoff between write retries"""

    read_max_retries: int = field(default=5)
    """Internal Neo4j read transient error retry attempts."""

    read_retry_base_delay: float = field(default=0.1)
    """Base delay in seconds for Neo4j read exponential backoff."""


@dataclass
class DatabaseConfig:
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    """Qdrant configuration."""

    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    """Neo4j configuration."""

    default_consistency: str = field(default="fast")
    """Default write consistency: fast or strong."""


@dataclass
class PipelineConfig:
    get: str = field(default="default_get")
    """Get pipeline implementation name."""

    delete: str = field(default="default_delete")
    """Delete pipeline implementation name."""

    update: str = field(default="default_update")
    """Update pipeline implementation name."""

    skill_evolve: str = field(default="trace_v2_summary")
    """Skill self-evolution algorithm version (registered under ``skill_evolve``)."""

    feedback: str = field(default="default_feedback")
    """Feedback pipeline implementation name."""

    dreaming: str = field(default="default_dreaming")
    """Dreaming pipeline implementation name."""


@dataclass
class AuthConfig:
    mode: str = field(default="api_key")
    """Authentication mode: api_key or gateway_jwt."""

    api_key_file: str = field(default=DEFAULT_API_KEY_FILE)
    """YAML file containing standalone memory-system API keys."""

    gateway_jwt_secret: str | None = secret_field(default=None)
    """Shared HMAC secret used to verify short-lived Gateway JWTs."""

    gateway_jwt_issuer: str = field(default="mindmemos-gateway")
    """Expected issuer for Gateway JWTs."""

    gateway_jwt_audience: str = field(default="memory-data-plane")
    """Expected audience for Gateway JWTs."""


@dataclass
class KafkaConsumerConfig:
    """Configuration for one Kafka consumer group."""

    topics: list[str] = field(default_factory=list)
    """Subscribed topic list."""

    group_id: str = ""
    """Consumer group id. Partitions are shared across consumers in the same group for scale-out."""

    max_poll_records: int = 50
    """Maximum messages fetched by one poll."""

    global_max_concurrency: int = 1
    """Global concurrency limit per consumer across all dispatch keys."""

    per_key_max_concurrency: int = 1
    """Maximum concurrency per dispatch key, such as one user. Default 1 keeps each key serial."""

    max_buffered: int = 1000
    """Limit for submitted but unfinished messages. Reaching it blocks polling to create backpressure."""

    session_timeout_ms: int = 10000
    """Broker session timeout for consumer heartbeats."""

    heartbeat_interval_ms: int = 3000
    """Consumer heartbeat interval."""

    max_poll_interval_ms: int = 300000
    """Maximum delay between consumer polls before the broker may revoke partitions."""

    auto_offset_reset: str = "earliest"
    """Offset reset policy when no committed offset exists: earliest or latest."""

    max_retries: int = 50
    """Maximum retries for one failed message before sending it to the dead-letter topic."""

    retry_base_delay: float = 0.1
    """Base delay in seconds for consumer handler retry backoff."""

    dlq_suffix: str = ".dlq"
    """Dead-letter topic suffix. Final topics look like ``{topic}{dlq_suffix}``."""


@dataclass
class KafkaTopicConfig:
    """Kafka topic definition ensured before producers and consumers start."""

    name: str = ""
    """Topic name."""

    partitions: int = 1
    """Target partition count. Existing topics are only expanded, never shrunk."""

    replication_factor: int = 1
    """Replication factor used when creating a missing topic."""


@dataclass
class KafkaConfig:
    enabled: bool = field(default=False)
    """Whether Kafka infrastructure is enabled."""

    bootstrap_servers: str = field(default_factory=lambda: _env_str("MINDMEMOS_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    """Kafka broker addresses, comma-separated."""

    client_id: str = field(default="memos")
    """Client id used to distinguish broker-side logs."""

    acks: str = field(default="all")
    """Producer acknowledgement level: 0, 1, or all. all gives the strongest durability."""

    enable_idempotence: bool = field(default=True)
    """Enable idempotent production to avoid duplicate broker writes during retries."""

    producer_linger_ms: int = field(default=5)
    """Producer batching linger time in milliseconds."""

    producer_max_batch_size: int = field(default=262144)
    """Maximum producer batch size per topic-partition in bytes."""

    request_timeout_ms: int = field(default=40000)
    """Request timeout in milliseconds."""

    producer_max_retries: int = field(default=5)
    """Producer send retry attempts before surfacing broker or network errors."""

    producer_retry_base_delay: float = field(default=0.1)
    """Base delay in seconds for producer send retry backoff."""

    producer_max_inflight: int = field(default=0)
    """Max un-acked in-flight messages before send() applies backpressure; 0 = unlimited."""

    global_max_concurrency: int = field(default=80)
    """Process-wide Kafka handler concurrency cap across all consumers; 0 disables the shared cap."""

    topics: list[KafkaTopicConfig] = field(default_factory=list)
    """Topics to create or expand before Kafka clients start."""

    consumers: list[KafkaConsumerConfig] = field(default_factory=list)
    """Consumer group configurations, each mapping to one background consumer."""


@dataclass
class MemoryConfig:
    telemetry: TelemetryConfig = frozen_field(default_factory=TelemetryConfig)
    """Telemetry configuration for logging and tracing."""

    chat_model_router: ModelRouterConfig = field(default_factory=ModelRouterConfig)
    """Chat model router configuration."""

    embed_model_router: ModelRouterConfig = field(default_factory=ModelRouterConfig)
    """Embedding model router configuration."""

    database: DatabaseConfig = frozen_field(default_factory=DatabaseConfig)
    """Memory database configuration."""

    pipelines: PipelineConfig = field(default_factory=PipelineConfig)
    """Pipeline implementation selection."""

    auth: AuthConfig = field(default_factory=AuthConfig)
    """HTTP API authentication config."""

    kafka: KafkaConfig = frozen_field(default_factory=KafkaConfig)
    """Kafka infrastructure config"""

    algo_config: MemoryAlgoConfig = field(default_factory=MemoryAlgoConfig)
    """Algorithm configuration."""

    rerank_model_router: ModelRouterConfig = field(default_factory=ModelRouterConfig)
    """Rerank model Router configuration. Empty endpoints -> cosine fallback."""


def build_config(config_name: str = "dev", config_path: str | Path | None = None) -> MemoryConfig:
    load_dotenv(DEFAULT_ENV_PATH, override=False)
    if config_path is None:
        config_path = default_config_path(config_name)
    config_path = Path(config_path).expanduser().resolve()
    logger.info("loading config", config_path=config_path, config_name=config_name)

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    data = _normalize_auth_paths(data, config_path)
    data = _merge_database_env(data)
    data = _merge_kafka_env(data)
    data = _merge_telemetry_env(data)
    cfg = build(MemoryConfig, data)
    validate_config(cfg)
    return cfg
