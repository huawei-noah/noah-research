"""MindMemOS SDK package."""

from .client import MindMemOSClient
from .config import ConfigManager, SDKConfig
from .errors import (
    ApiError,
    AuthRequiredError,
    ConfigError,
    MindMemOSSDKError,
    TransportError,
)
from .memory import (
    AddResult,
    AsyncMemoryClient,
    DialogueMessage,
    FeedbackMode,
    FileMessage,
    GetResult,
    MemoryClient,
    SearchResult,
    StatusResult,
    TextMessage,
    UrlMessage,
)

__all__ = [
    "__version__",
    "MindMemOSClient",
    "MemoryClient",
    "AsyncMemoryClient",
    "FeedbackMode",
    "ConfigManager",
    "SDKConfig",
    "AddResult",
    "SearchResult",
    "GetResult",
    "StatusResult",
    "TextMessage",
    "DialogueMessage",
    "UrlMessage",
    "FileMessage",
    "MindMemOSSDKError",
    "ConfigError",
    "AuthRequiredError",
    "TransportError",
    "ApiError",
]

__version__ = "0.1.0"
