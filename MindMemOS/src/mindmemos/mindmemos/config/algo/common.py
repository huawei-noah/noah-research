"""Common algorithm configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CommonAlgoConfig:
    """Common settings shared by algorithm domains."""

    prompt_language: str = field(default="EN")
    """Fallback prompt language when per-request auto-detection is ambiguous (mixed/unknown). Supported values: EN, ZH."""
