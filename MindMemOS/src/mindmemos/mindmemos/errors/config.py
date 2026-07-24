from .base import MindMemOSError


class ConfigNotInitializedError(MindMemOSError):
    """Raised when get_config() is called before init_config()."""

    def __init__(self):
        super().__init__("Config has not been initialized. Call init_config() first.")


class MissingConfigValueError(MindMemOSError):
    """Raised when a required configuration value is missing."""

    def __init__(self, field: str, reason: str = ""):
        msg = f"Missing required config field: '{field}'"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)
        self.field = field
        self.reason = reason


class InvalidConfigError(MindMemOSError):
    def __init__(self, field: str, support: str | None = None):
        msg = f"Invalid config field: '{field}'"
        if support:
            msg += f" only support ({support})"
        super().__init__(msg)
        self.field = field
        self.support = support
