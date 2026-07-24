"""Root algorithm configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..base import frozen_field
from .add import AddAlgoConfig
from .common import CommonAlgoConfig
from .dreaming import DreamingConfig
from .search import SearchConfig
from .skill import SkillEvolutionConfig
from .text_processing import TextProcessingConfig


@dataclass
class MemoryAlgoConfig:
    """Top-level algorithm configuration grouped by operation domain."""

    common: CommonAlgoConfig = field(default_factory=CommonAlgoConfig)
    """Shared algorithm defaults used across add, search, and background jobs."""

    text_processing: TextProcessingConfig = frozen_field(default_factory=TextProcessingConfig)
    """Process-wide text preprocessing config.

    Tenant/project overrides are forbidden because TextPreprocessor shares lazy
    NLP models and sparse/entity normalization state globally.
    """

    add: AddAlgoConfig = field(default_factory=AddAlgoConfig)
    """Memory add pipeline configuration, including vanilla and schema add behavior."""

    dreaming: DreamingConfig = field(default_factory=DreamingConfig)
    """Dreaming pipeline configuration for asynchronous memory consolidation."""

    search: SearchConfig = field(default_factory=SearchConfig)
    """Memory search pipeline configuration for recall, ranking, and expansion."""

    skill_evolution: SkillEvolutionConfig = field(default_factory=SkillEvolutionConfig)
    """Skill evolution configuration for benchmark-driven skill optimization."""
