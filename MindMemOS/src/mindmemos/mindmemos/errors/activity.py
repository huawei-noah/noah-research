from .base import MindMemOSError


class ActivityCollectionError(MindMemOSError):
    """Raised when reading activity logs (add/search records) fails.

    The collector is read-only; on a scroll failure it raises this component-level
    error and lets the upstream pipeline decide on a degradation strategy.
    """
