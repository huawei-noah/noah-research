"""Activity-log reading components for dreaming / feedback pipelines."""

from .collector import (
    DEFAULT_LOOKBACK,
    DEFAULT_MAX_RECORDS,
    RecentActivityCollector,
)

__all__ = [
    "DEFAULT_LOOKBACK",
    "DEFAULT_MAX_RECORDS",
    "RecentActivityCollector",
]
