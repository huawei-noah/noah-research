"""Root SDK client facade.

Assembles the SDK from local config: it reads ``~/.mindmemos/settings.json`` via
:class:`ConfigManager`, builds a shared :class:`HttpTransport`, and exposes resource
clients (currently ``memory``). Explicit constructor arguments override config so
callers can run without a config file.
"""

from __future__ import annotations

from .config import ConfigManager, SDKConfig
from .errors import AuthRequiredError
from .memory import MemoryClient
from .skills import SkillCloudClient, SkillManager
from .transport import HttpTransport


class MindMemOSClient:
    """Public root SDK client composed from config, transport, and resource clients."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        user_id: str | None = None,
        app_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        config: SDKConfig | None = None,
        config_manager: ConfigManager | None = None,
        transport: HttpTransport | None = None,
    ) -> None:
        """Handle init."""
        if config is None:
            manager = config_manager or ConfigManager()
            config = manager.load_or_default()
        else:
            manager = config_manager or ConfigManager()
        self._config = config

        self._base_url = base_url or config.base_url
        self._api_key = api_key or config.auth.api_key
        self._user_id = user_id or config.defaults.user_id
        self._app_id = app_id
        self._agent_id = agent_id
        self._session_id = session_id

        self._transport = transport or HttpTransport(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout_seconds=config.network.timeout_seconds,
            max_retries=config.network.max_retries,
        )

        self.skills = SkillManager.from_config_manager(manager, SkillCloudClient(self._transport))
        self.memory = MemoryClient(
            self._transport,
            default_user_id=self._user_id,
            default_app_id=self._app_id,
            default_agent_id=self._agent_id,
            default_session_id=self._session_id,
            skill_manager=self.skills,
        )

    def require_api_key(self) -> str:
        """Return the configured API key or raise an auth error."""
        if not self._api_key:
            raise AuthRequiredError("No api_key configured. Run `mindmemos auth` first.")
        return self._api_key

    def close(self) -> None:
        """Release underlying transport resources."""
        self._transport.close()

    def __enter__(self) -> MindMemOSClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
