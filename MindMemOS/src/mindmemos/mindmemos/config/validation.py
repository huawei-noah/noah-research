"""Central config validation rules."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

from omegaconf.errors import ConfigAttributeError, MissingMandatoryValue

from ..errors import InvalidConfigError, MissingConfigValueError


@dataclass(frozen=True)
class ChoiceRule:
    path: str
    choices: frozenset[str]
    allow_none: bool = False
    case_insensitive: bool = False


@dataclass(frozen=True)
class RangeRule:
    path: str
    min_value: float | None = None
    max_value: float | None = None
    allow_none: bool = False
    include_min: bool = True
    include_max: bool = True
    support: str | None = None


CHOICE_RULES: tuple[ChoiceRule, ...] = (
    ChoiceRule("telemetry.span_type", frozenset({"simple", "batch"}), case_insensitive=True),
    ChoiceRule("telemetry.log_level", frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}), case_insensitive=True),
    ChoiceRule("auth.mode", frozenset({"api_key", "gateway_jwt"})),
    ChoiceRule("database.qdrant.distance", frozenset({"Cosine", "Euclid", "Dot", "Manhattan"}), case_insensitive=True),
    ChoiceRule("database.default_consistency", frozenset({"fast", "strong"})),
    ChoiceRule("algo_config.common.prompt_language", frozenset({"EN", "ZH"}), case_insensitive=True),
    ChoiceRule(
        "algo_config.text_processing.unicode_normal_form",
        frozenset({"NFC", "NFKC", "NFD", "NFKD"}),
        allow_none=True,
    ),
    ChoiceRule("algo_config.text_processing.sparse_fallback_mode", frozenset({"tf", "log_tf"})),
    ChoiceRule("algo_config.add.schema.chunker.split_mode", frozenset({"llm", "rule"})),
    ChoiceRule("kafka.acks", frozenset({"0", "1", "all"})),
)

RANGE_RULES: tuple[RangeRule, ...] = (
    RangeRule("telemetry.telemetry_timeout", min_value=0, include_min=False, support="positive number"),
    RangeRule("telemetry.trace_sampling_ratio", min_value=0, max_value=1, support="0 <= value <= 1"),
    RangeRule("telemetry.max_queue_size", min_value=1, support="positive integer >= 1"),
    RangeRule("telemetry.max_export_batch_size", min_value=1, support="positive integer >= 1"),
    RangeRule("telemetry.metric_export_interval_millis", min_value=1, support="positive integer >= 1"),
    RangeRule("database.qdrant.grpc_port", min_value=1, support="positive integer >= 1"),
    RangeRule("database.qdrant.vector_size", min_value=1, support="positive integer >= 1"),
    RangeRule("database.qdrant.timeout", min_value=0, include_min=False, support="positive number"),
    RangeRule("database.qdrant.pool_size", min_value=0, support="non-negative integer"),
    RangeRule("database.qdrant.max_client_concurrency", min_value=1, support="positive integer >= 1"),
    RangeRule("database.qdrant.request_read_budget", min_value=1, support="positive integer >= 1"),
    RangeRule("database.qdrant.max_retries", min_value=0, support="non-negative integer"),
    RangeRule("database.qdrant.retry_base_delay", min_value=0, support="non-negative number"),
    RangeRule("database.qdrant.batch_upsert_size", min_value=1, support="positive integer >= 1"),
    RangeRule("database.qdrant.batch_upsert_flush_interval_ms", min_value=0, support="non-negative integer"),
    RangeRule("database.qdrant.batch_upsert_max_queue_size", min_value=1, support="positive integer >= 1"),
    RangeRule("database.qdrant.batch_upsert_max_inflight_batches", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.vanilla.hybrid_prefetch_factor", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.vanilla.hybrid_prefetch_min", min_value=1, support="positive integer >= 1"),
    RangeRule("database.neo4j.max_connection_lifetime", min_value=0, include_min=False, support="positive number"),
    RangeRule("database.neo4j.max_connection_pool_size", min_value=1, support="positive integer >= 1"),
    RangeRule("database.neo4j.connection_acquisition_timeout", min_value=0, include_min=False, support="positive number"),
    RangeRule("database.neo4j.max_client_concurrency", min_value=1, support="positive integer >= 1"),
    RangeRule("database.neo4j.request_row_budget", min_value=1, support="positive integer >= 1"),
    RangeRule("database.neo4j.write_max_retries", min_value=0, support="non-negative integer"),
    RangeRule("database.neo4j.write_retry_base_delay", min_value=0, support="non-negative number"),
    RangeRule("database.neo4j.read_max_retries", min_value=0, support="non-negative integer"),
    RangeRule("database.neo4j.read_retry_base_delay", min_value=0, support="non-negative number"),
    RangeRule("chat_model_router.allowed_fails", min_value=0, allow_none=True, support="non-negative integer"),
    RangeRule("chat_model_router.cool_down", min_value=0, allow_none=True, support="non-negative number"),
    RangeRule("chat_model_router.format_parser_max_attempts", min_value=1, support="positive integer >= 1"),
    RangeRule("embed_model_router.allowed_fails", min_value=0, allow_none=True, support="non-negative integer"),
    RangeRule("embed_model_router.cool_down", min_value=0, allow_none=True, support="non-negative number"),
    RangeRule("embed_model_router.format_parser_max_attempts", min_value=1, support="positive integer >= 1"),
    RangeRule("rerank_model_router.allowed_fails", min_value=0, allow_none=True, support="non-negative integer"),
    RangeRule("rerank_model_router.cool_down", min_value=0, allow_none=True, support="non-negative number"),
    RangeRule("rerank_model_router.format_parser_max_attempts", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.text_processing.explicit_language_confidence", min_value=0, max_value=1),
    RangeRule("algo_config.text_processing.lang_zh_ratio", min_value=0, max_value=1),
    RangeRule("algo_config.text_processing.lang_en_latin_ratio", min_value=0, max_value=1),
    RangeRule("algo_config.text_processing.lang_mixed_zh_ratio", min_value=0, max_value=1),
    RangeRule("algo_config.text_processing.lang_mixed_latin_ratio", min_value=0, max_value=1),
    RangeRule("algo_config.text_processing.bm25_min_term_len", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.text_processing.sparse_hash_dim", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.text_processing.sparse_k1", min_value=0, include_min=False, support="positive number"),
    RangeRule("algo_config.text_processing.sparse_b", min_value=0, max_value=1, support="0 <= value <= 1"),
    RangeRule("algo_config.text_processing.bm25_idf_smoothing", min_value=0, include_min=False, support="positive number"),
    RangeRule("algo_config.text_processing.bm25_min_idf_denominator", min_value=0, include_min=False, support="positive number"),
    RangeRule("algo_config.text_processing.bm25_min_avg_doc_len", min_value=0, include_min=False, support="positive number"),
    RangeRule("algo_config.text_processing.spacy_entity_default_confidence", min_value=0, max_value=1),
    RangeRule("algo_config.text_processing.rule_entity_default_confidence", min_value=0, max_value=1),
    RangeRule("algo_config.text_processing.max_entity_count", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.text_processing.rule_zh_min_term_len", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.text_processing.nlp_max_retries", min_value=0, support="non-negative integer"),
    RangeRule("algo_config.text_processing.nlp_retry_base_delay", min_value=0, support="non-negative number"),
    RangeRule("algo_config.add.vanilla.chunk_soft_token_budget", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.vanilla.chunk_hard_token_budget", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.vanilla.turn_hard_token_budget", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.vanilla.history_soft_token_budget", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.vanilla.history_hard_token_budget", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.vanilla.history_min_turn_count", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.vanilla.compaction_soft_token_budget", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.vanilla.compaction_head_tokens", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.vanilla.compaction_tail_tokens", min_value=1, support="positive integer >= 1"),
    RangeRule(
        "algo_config.add.vanilla.compaction_summary_context_token_budget",
        min_value=1,
        support="positive integer >= 1",
    ),
    RangeRule(
        "algo_config.add.vanilla.compaction_summary_output_token_budget",
        min_value=1,
        support="positive integer >= 1",
    ),
    RangeRule("algo_config.add.vanilla.time_gap_threshold_seconds", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.vanilla.template_tokens", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.vanilla.recall_budget", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.vanilla.output_headroom", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.vanilla.recall.top_k", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.vanilla.recall.scan_limit", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.vanilla.recall.fusion_k", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.vanilla.recall.fusion_weight_semantic", min_value=0, support="non-negative number"),
    RangeRule("algo_config.add.vanilla.recall.fusion_weight_bm25", min_value=0, support="non-negative number"),
    RangeRule("algo_config.add.vanilla.recall.fusion_weight_entity", min_value=0, support="non-negative number"),
    RangeRule("algo_config.add.vanilla.recall.fusion_weight_recent", min_value=0, support="non-negative number"),
    RangeRule(
        "algo_config.add.vanilla.recall.fusion_weight_schema_property",
        min_value=0,
        support="non-negative number",
    ),
    RangeRule("algo_config.add.vanilla.safety_gate.min_content_chars", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.vanilla.safety_gate.min_update_confidence", min_value=0, max_value=1),
    RangeRule("algo_config.add.vanilla.safety_gate.min_merge_confidence", min_value=0, max_value=1),
    RangeRule("algo_config.add.schema.extraction.search_fields_max", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.schema.extraction.episode_augment_count", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.schema.merge.entity_recall_top_k", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.schema.merge.secondary_search_limit", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.schema.merge.max_merge_retries", min_value=0, support="non-negative integer"),
    RangeRule("algo_config.add.schema.merge.secondary_search_retries", min_value=0, support="non-negative integer"),
    RangeRule("algo_config.add.schema.higher_order.top_k", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.schema.higher_order.min_evidence_count", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.schema.episode_edge.top_k", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.schema.chunker.min_episode_length", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.schema.chunker.max_episode_length", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.schema.chunker.max_buffer_size", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.schema.chunker.max_minutes_from_first", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.add.schema.drain.episode_generation_max_retries", min_value=0, support="non-negative integer"),
    RangeRule("algo_config.search.request_top_k_max", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.default.top_k", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.vanilla.recall_size", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.vanilla.graph_seed_memory_limit", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.vanilla.graph_related_per_seed", min_value=1, support="positive integer >= 1"),
    RangeRule(
        "algo_config.search.vanilla.shared_entity_graph_limit_per_entity",
        min_value=1,
        support="positive integer >= 1",
    ),
    RangeRule("algo_config.search.vanilla.graph_max_candidates", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.vanilla.graph_decay", min_value=0, support="non-negative number"),
    RangeRule("algo_config.search.vanilla.graph_score", min_value=0, support="non-negative number"),
    RangeRule("algo_config.search.rerank.max_query_length", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.rerank.max_doc_length", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.rerank.max_batch_size", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.rerank.max_concurrent_batches", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.rerank.request_timeout", min_value=0, include_min=False, support="positive number"),
    RangeRule("algo_config.search.schema_search.entity.recall_size", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.schema_search.entity.rrf_k", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.schema_search.entity.top_k", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.schema_search.entity.top_n", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.schema_search.entity.max_rerank_candidates", min_value=1, support="positive integer >= 1"),
    RangeRule(
        "algo_config.search.schema_search.entity.search_field_overfetch_factor",
        min_value=1,
        support="positive integer >= 1",
    ),
    RangeRule("algo_config.search.schema_search.entity.maxsim_weight", min_value=0, max_value=1, support="0 <= value <= 1"),
    RangeRule("algo_config.search.schema_search.property.recall_size", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.schema_search.property.rrf_k", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.schema_search.property.top_k", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.schema_search.property.top_n", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.schema_search.property.extension_step", min_value=1, support="positive integer >= 1"),
    RangeRule(
        "algo_config.search.schema_search.property.higher_order_ratio",
        min_value=0,
        max_value=1,
        support="0 <= value <= 1",
    ),
    RangeRule(
        "algo_config.search.schema_search.property.alloc_min_factor",
        min_value=0,
        include_min=False,
        support="positive number",
    ),
    RangeRule(
        "algo_config.search.schema_search.dual_path.property_recall_size",
        min_value=1,
        support="positive integer >= 1",
    ),
    RangeRule("algo_config.search.schema_search.dual_path.property_rrf_k", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.schema_search.dual_path.property_top_k", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.schema_search.dual_path.property_top_n", min_value=1, support="positive integer >= 1"),
    RangeRule(
        "algo_config.search.schema_search.entity_weights.episode_weight",
        min_value=0,
        max_value=1,
        support="0 <= value <= 1",
    ),
    RangeRule(
        "algo_config.search.schema_search.entity_weights.non_episode_weight",
        min_value=0,
        max_value=1,
        support="0 <= value <= 1",
    ),
    RangeRule("algo_config.search.schema_search.edge.top_k", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.schema_search.edge.neighbor_fetch_limit", min_value=1, support="positive integer >= 1"),
    RangeRule(
        "algo_config.search.schema_search.edge.min_relevance_score",
        min_value=0,
        max_value=1,
        support="0 <= value <= 1",
    ),
    RangeRule("algo_config.search.schema_search.multi_hop", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.schema_search.output_max_edge_num", min_value=1, support="positive integer >= 1"),
    RangeRule(
        "algo_config.search.schema_search.min_time_window_days",
        min_value=1,
        allow_none=True,
        support="positive integer >= 1",
    ),
    RangeRule("algo_config.search.agentic.max_rounds", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.agentic.top_k_per_round", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.agentic.top_n_per_round", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.agentic.num_hops", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.search.agentic.output_max_edge_num", min_value=1, support="positive integer >= 1"),
    RangeRule(
        "algo_config.search.agentic.min_time_window_days",
        min_value=1,
        allow_none=True,
        support="positive integer >= 1",
    ),
    RangeRule("algo_config.dreaming.lookback_days", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.dreaming.max_memories_per_scope", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.dreaming.min_scope_updates", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.dreaming.min_cluster_size", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.dreaming.concurrency", min_value=1, support="positive integer >= 1"),
    RangeRule("algo_config.dreaming.scope_batch_size", min_value=1, support="positive integer >= 1"),
    RangeRule(
        "algo_config.dreaming.max_scopes_per_run",
        min_value=1,
        allow_none=True,
        support="positive integer >= 1",
    ),
    RangeRule(
        "algo_config.dreaming.max_seed_memories",
        min_value=1,
        allow_none=True,
        support="positive integer >= 1",
    ),
    RangeRule("kafka.producer_linger_ms", min_value=0, support="non-negative integer"),
    RangeRule("kafka.producer_max_batch_size", min_value=1, support="positive integer >= 1"),
    RangeRule("kafka.request_timeout_ms", min_value=1, support="positive integer >= 1"),
    RangeRule("kafka.producer_max_retries", min_value=0, support="non-negative integer"),
    RangeRule("kafka.producer_retry_base_delay", min_value=0, support="non-negative number"),
    RangeRule("kafka.producer_max_inflight", min_value=0, support="non-negative integer; 0 disables the cap"),
    RangeRule("kafka.global_max_concurrency", min_value=0, support="non-negative integer; 0 disables the cap"),
)

REQUIRED_STRING_PATHS: tuple[str, ...] = (
    "database.qdrant.url",
    "database.qdrant.memory_collection",
    "database.qdrant.entity_collection",
    "database.qdrant.source_collection",
    "database.qdrant.add_record_collection",
    "database.qdrant.schema_add_buffer_collection",
    "database.qdrant.search_record_collection",
    "database.qdrant.skill_version_collection",
    "database.qdrant.skill_blob_collection",
    "database.qdrant.skill_trace_pending_collection",
    "database.qdrant.skill_trace_summary_collection",
    "database.qdrant.semantic_vector_name",
    "database.qdrant.bm25_vector_name",
    "database.neo4j.uri",
    "database.neo4j.username",
    "database.neo4j.database",
    "auth.api_key_file",
    "chat_model_router.routing_strategy",
    "embed_model_router.routing_strategy",
    "rerank_model_router.routing_strategy",
    "algo_config.add.schema.entity_modeling_path",
)


def validate_config(cfg: Any) -> None:
    """Validate a fully built config object before it becomes the process config."""

    _validate_choice_rules(cfg)
    _validate_range_rules(cfg)
    _validate_required_strings(cfg)
    _validate_telemetry(cfg)
    _validate_auth(cfg)
    _validate_model_router("chat_model_router", cfg.chat_model_router)
    _validate_model_router("embed_model_router", cfg.embed_model_router)
    _validate_model_router("rerank_model_router", cfg.rerank_model_router)
    _validate_embedding_vector_size(cfg)
    _validate_algo(cfg)
    _validate_kafka(cfg.kafka)


def _validate_choice_rules(cfg: Any) -> None:
    for rule in CHOICE_RULES:
        value = _get(cfg, rule.path)
        if value is None and rule.allow_none:
            continue
        actual = str(value)
        choices = {choice.lower() for choice in rule.choices} if rule.case_insensitive else set(rule.choices)
        candidate = actual.lower() if rule.case_insensitive else actual
        if candidate not in choices:
            raise InvalidConfigError(rule.path, support=", ".join(sorted(rule.choices)))


def _validate_range_rules(cfg: Any) -> None:
    for rule in RANGE_RULES:
        _validate_range(rule.path, _get(cfg, rule.path), rule)


def _validate_required_strings(cfg: Any) -> None:
    for path in REQUIRED_STRING_PATHS:
        _require_string(path, _get(cfg, path))


def _validate_telemetry(cfg: Any) -> None:
    if cfg.telemetry.enabled:
        _require_string(
            "telemetry.telemetry_endpoint",
            cfg.telemetry.telemetry_endpoint,
            reason="telemetry is enabled",
        )
    if cfg.telemetry.max_export_batch_size > cfg.telemetry.max_queue_size:
        raise InvalidConfigError("telemetry.max_export_batch_size", support="<= telemetry.max_queue_size")


def _validate_auth(cfg: Any) -> None:
    if cfg.auth.mode == "gateway_jwt":
        _require_string(
            "auth.gateway_jwt_secret",
            cfg.auth.gateway_jwt_secret,
            reason="auth.mode is gateway_jwt",
        )


def _validate_model_router(path: str, router: Any) -> None:
    for index, endpoint in enumerate(router.endpoints):
        prefix = f"{path}.endpoints[{index}]"
        _require_string(f"{prefix}.model", _attr(endpoint, "model", f"{prefix}.model"))
        _require_string(f"{prefix}.api_key", _attr(endpoint, "api_key", f"{prefix}.api_key"))
        _require_string(f"{prefix}.api_base", _attr(endpoint, "api_base", f"{prefix}.api_base"))
        _positive_optional(endpoint, "rpm", f"{prefix}.rpm")
        _positive_optional(endpoint, "tpm", f"{prefix}.tpm")
        _validate_range(
            f"{prefix}.timeout",
            endpoint.timeout,
            RangeRule(f"{prefix}.timeout", min_value=0, include_min=False, support="positive number"),
        )
        _validate_range(
            f"{prefix}.num_retries",
            endpoint.num_retries,
            RangeRule(f"{prefix}.num_retries", min_value=0, support="non-negative integer"),
        )
        _validate_range(
            f"{prefix}.temperature",
            endpoint.temperature,
            RangeRule(f"{prefix}.temperature", min_value=0, allow_none=True, support="non-negative number"),
        )
        _validate_range(
            f"{prefix}.top_p",
            endpoint.top_p,
            RangeRule(f"{prefix}.top_p", min_value=0, max_value=1, allow_none=True, support="0 <= value <= 1"),
        )
        _positive_optional(endpoint, "max_tokens", f"{prefix}.max_tokens")
        _positive_optional(endpoint, "max_completion_tokens", f"{prefix}.max_completion_tokens")
        _positive_optional(endpoint, "dimensions", f"{prefix}.dimensions")


def _validate_embedding_vector_size(cfg: Any) -> None:
    vector_size = cfg.database.qdrant.vector_size
    for index, endpoint in enumerate(cfg.embed_model_router.endpoints):
        dimensions = endpoint.dimensions
        if dimensions is not None and dimensions != vector_size:
            raise InvalidConfigError(
                f"embed_model_router.endpoints[{index}].dimensions",
                support=f"must equal database.qdrant.vector_size ({vector_size})",
            )


def _validate_algo(cfg: Any) -> None:
    vanilla_add = cfg.algo_config.add.vanilla
    if vanilla_add.chunk_hard_token_budget < vanilla_add.chunk_soft_token_budget:
        raise InvalidConfigError(
            "algo_config.add.vanilla.chunk_hard_token_budget",
            support=">= algo_config.add.vanilla.chunk_soft_token_budget",
        )
    if vanilla_add.history_hard_token_budget < vanilla_add.history_soft_token_budget:
        raise InvalidConfigError(
            "algo_config.add.vanilla.history_hard_token_budget",
            support=">= algo_config.add.vanilla.history_soft_token_budget",
        )
    reserved = (
        vanilla_add.template_tokens
        + vanilla_add.output_headroom
        + vanilla_add.recall_budget
        + vanilla_add.history_hard_token_budget
    )
    if vanilla_add.chunk_hard_token_budget <= reserved:
        raise InvalidConfigError(
            "algo_config.add.vanilla.chunk_hard_token_budget",
            support="> template_tokens + output_headroom + recall_budget + history_hard_token_budget",
        )

    _validate_schema_add(cfg.algo_config.add.schema)
    _validate_search(cfg.algo_config.search)


def _validate_schema_add(schema_add: Any) -> None:
    if schema_add.chunker.max_episode_length < schema_add.chunker.min_episode_length:
        raise InvalidConfigError(
            "algo_config.add.schema.chunker.max_episode_length",
            support=">= algo_config.add.schema.chunker.min_episode_length",
        )


def _validate_search(search: Any) -> None:
    schema = search.schema_search
    if schema.property.alloc_max_factor < schema.property.alloc_min_factor:
        raise InvalidConfigError(
            "algo_config.search.schema_search.property.alloc_max_factor",
            support=">= algo_config.search.schema_search.property.alloc_min_factor",
        )


def _validate_kafka(kafka: Any) -> None:
    if kafka.enabled:
        _require_string("kafka.bootstrap_servers", kafka.bootstrap_servers, reason="kafka is enabled")
    _require_string("kafka.client_id", kafka.client_id)

    seen_topics: set[str] = set()
    for index, topic in enumerate(kafka.topics):
        name = topic.name
        path = f"kafka.topics[{index}].name"
        _require_string(path, name)
        if name in seen_topics:
            raise InvalidConfigError(path, support="unique topic name")
        seen_topics.add(name)
        for key in ("partitions", "replication_factor"):
            _validate_range(
                f"kafka.topics[{name}].{key}",
                getattr(topic, key),
                RangeRule("", min_value=1, support="positive integer >= 1"),
            )

    for consumer in kafka.consumers:
        group_id = consumer.group_id or "<unknown>"
        _require_string(f"kafka.consumers[{group_id}].group_id", consumer.group_id)
        if not consumer.topics:
            raise MissingConfigValueError(f"kafka.consumers[{group_id}].topics")
        for topic in consumer.topics:
            _require_string(f"kafka.consumers[{group_id}].topics", topic)
        for key in (
            "max_poll_records",
            "global_max_concurrency",
            "per_key_max_concurrency",
            "max_buffered",
            "session_timeout_ms",
            "heartbeat_interval_ms",
            "max_poll_interval_ms",
        ):
            _validate_range(
                f"kafka.consumers[{group_id}].{key}",
                getattr(consumer, key),
                RangeRule("", min_value=1, support="positive integer >= 1"),
            )
        _validate_range(
            f"kafka.consumers[{group_id}].max_retries",
            consumer.max_retries,
            RangeRule("", min_value=0, support="non-negative integer"),
        )
        _validate_range(
            f"kafka.consumers[{group_id}].retry_base_delay",
            consumer.retry_base_delay,
            RangeRule("", min_value=0, support="non-negative number"),
        )
        _validate_choice(
            f"kafka.consumers[{group_id}].auto_offset_reset",
            consumer.auto_offset_reset,
            {"earliest", "latest"},
        )


def _get(cfg: Any, path: str) -> Any:
    current = cfg
    for part in path.split("."):
        try:
            current = getattr(current, part)
        except MissingMandatoryValue as exc:
            raise MissingConfigValueError(path) from exc
        except (AttributeError, ConfigAttributeError, KeyError) as exc:
            raise MissingConfigValueError(path) from exc
    return current


def _attr(cfg: Any, name: str, path: str) -> Any:
    try:
        return getattr(cfg, name)
    except MissingMandatoryValue as exc:
        raise MissingConfigValueError(path) from exc


def _require_string(path: str, value: Any, *, reason: str = "") -> None:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise MissingConfigValueError(path, reason=reason)
    if not isinstance(value, str):
        raise InvalidConfigError(path, support="non-empty string")


def _positive_optional(cfg: Any, name: str, path: str) -> None:
    _validate_range(path, _attr(cfg, name, path), RangeRule(path, min_value=0, include_min=False, allow_none=True))


def _validate_choice(path: str, value: Any, choices: set[str]) -> None:
    if str(value) not in choices:
        raise InvalidConfigError(path, support=", ".join(sorted(choices)))


def _validate_range(path: str, value: Any, rule: RangeRule) -> None:
    if value is None and rule.allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidConfigError(path, support=rule.support or "number")
    if rule.min_value is not None:
        too_small = value < rule.min_value if rule.include_min else value <= rule.min_value
        if too_small:
            raise InvalidConfigError(path, support=rule.support or _range_support(rule))
    if rule.max_value is not None:
        too_large = value > rule.max_value if rule.include_max else value >= rule.max_value
        if too_large:
            raise InvalidConfigError(path, support=rule.support or _range_support(rule))


def _range_support(rule: RangeRule) -> str:
    parts: list[str] = []
    if rule.min_value is not None:
        op = ">=" if rule.include_min else ">"
        parts.append(f"{op} {rule.min_value:g}")
    if rule.max_value is not None:
        op = "<=" if rule.include_max else "<"
        parts.append(f"{op} {rule.max_value:g}")
    return " and ".join(parts) or "number"
