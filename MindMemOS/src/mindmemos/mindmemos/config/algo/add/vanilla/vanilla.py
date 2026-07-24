"""Vanilla flat-memory add pipeline configuration."""

from dataclasses import dataclass, field


@dataclass
class VanillaAddRecallConfig:
    """Related-memory recall tunables for the add pipeline's recall-before-extract phase.

    Purpose:
        Control how many related memories are surfaced to the extractor and how the
        multi-channel (hash / entity / bm25) candidates fuse via weighted RRF.

    Used in:
        - ``MemoryAlgoConfig.add.vanilla.recall`` for YAML-driven configuration
        - ``RelatedMemoryRecall`` constructor
    """

    top_k: int = field(default=5)
    """Maximum related memories returned to the extractor after RRF fusion."""

    scan_limit: int = field(default=100)
    """Maximum active memories scanned per project for hash/entity candidate generation."""

    fusion_k: int = field(default=60)
    """RRF smoothing constant; higher values flatten rank differences between channels."""

    fusion_weight_semantic: float = field(default=1.5)
    """RRF weight for the dense semantic channel."""

    fusion_weight_bm25: float = field(default=1.0)
    """RRF weight for the sparse BM25 channel."""

    fusion_weight_entity: float = field(default=1.2)
    """RRF weight for the entity-overlap channel."""

    fusion_weight_recent: float = field(default=0.5)
    """RRF weight for the recency channel."""

    fusion_weight_schema_property: float = field(default=2.0)
    """RRF weight for the schema-property channel."""


@dataclass
class VanillaAddSafetyGateConfig:
    """Deterministic safety-gate thresholds applied before DB write.

    Purpose:
        Confidence floors below which low-trust extractor hints (update / merge)
        downgrade to ADD. These gate write semantics and are meant to be tunable
        without redeployment.

    Used in:
        - ``MemoryAlgoConfig.add.vanilla.safety_gate`` for YAML-driven configuration
        - ``AddSafetyGate`` constructor
    """

    min_content_chars: int = field(default=1)
    """Minimum normalized content length; shorter candidates are skipped."""

    min_update_confidence: float = field(default=0.7)
    """Confidence floor for honoring an ``update`` hint; below it the action downgrades to ADD."""

    min_merge_confidence: float = field(default=0.8)
    """Confidence floor for honoring a ``merge`` hint; below it the action downgrades to ADD."""


@dataclass
class VanillaAddConfig:
    """Vanilla add pipeline parameters.

    Purpose:
        Chunking, history packing, and long-turn compaction tunables for
        the vanilla add pipeline. All values are token-based; message and
        turn counts are used only for observability.

    Budget hierarchy::

        chunk_hard_token_budget (total prompt limit, e.g. 32000)
        ├── template_tokens      (prompt template overhead)
        ├── output_headroom      (reserved for LLM output)
        ├── recall_budget        (related memory recall context)
        ├── history_hard_token_budget (history context cap)
        └── extractable_budget   (remaining for chunk messages)
            = hard - template - output - recall - history_hard

        chunk_soft_token_budget (target, e.g. 26000)
        └── soft_extractable_budget
            = soft - template - output - recall - history_soft

    Used in:
        - ``MemoryAlgoConfig.add.vanilla`` for YAML-driven configuration
        - ``AddCoreBuilder`` for chunking orchestration
        - ``TurnGrouper``, ``ChunkPlanner``, ``HistoryPacker``,
          ``LongTurnCompactor`` for behavior control
    """

    chunk_soft_token_budget: int = field(default=26000)
    """Soft target for total chunk prompt tokens. Turns are not split to hit this."""

    chunk_hard_token_budget: int = field(default=32000)
    """Maximum total prompt tokens per chunk (template + extractable + history + recall + output)."""

    turn_hard_token_budget: int = field(default=16000)
    """Token threshold above which a single turn is flagged for long-turn compaction."""

    history_soft_token_budget: int = field(default=2000)
    """Target token count for packed history context."""

    history_hard_token_budget: int = field(default=4000)
    """Maximum token count for packed history. Hard cap prevents prompt growth."""

    history_min_turn_count: int = field(default=1)
    """Minimum complete turns to include in history, even if they exceed soft budget."""

    compaction_soft_token_budget: int = field(default=16000)
    """Soft target for compacted long turns; preserving the first user message may exceed it."""

    compaction_head_tokens: int = field(default=4000)
    """Tokens preserved from the start of a long turn as raw evidence."""

    compaction_tail_tokens: int = field(default=4000)
    """Tokens preserved from the end of a long turn as raw evidence."""

    compaction_summary_context_token_budget: int = field(default=200000)
    """Maximum middle-section tokens sent in one long-turn summary call."""

    compaction_summary_output_token_budget: int = field(default=8000)
    """Maximum generated tokens for each long-turn summary call."""

    time_gap_threshold_seconds: int = field(default=1800)
    """Timestamp gap (seconds) that splits consecutive same-role messages into separate turns."""

    template_tokens: int = field(default=1000)
    """Estimated tokens consumed by the extraction prompt template and system instructions."""

    recall_budget: int = field(default=2000)
    """Token budget allocated for related memory recall context in the prompt."""

    output_headroom: int = field(default=4000)
    """Token headroom reserved for the LLM output (extraction results)."""

    enable_entities: bool = field(default=False)
    """Whether vanilla add writes entity nodes, MENTIONS edges, and entity embeddings."""

    # --- Related-memory recall (recall-before-extract phase) ---
    recall: VanillaAddRecallConfig = field(default_factory=VanillaAddRecallConfig)
    """Related-memory recall tunables; surfaced to the extractor as context."""

    # --- Safety gate (deterministic validation before DB write) ---
    safety_gate: VanillaAddSafetyGateConfig = field(default_factory=VanillaAddSafetyGateConfig)
    """Confidence floors and content checks applied to extractor action hints."""
