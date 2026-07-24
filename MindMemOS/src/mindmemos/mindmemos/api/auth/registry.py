"""Authentication provider factory."""

from __future__ import annotations

from ...config import get_config
from ...errors import AuthenticationError
from ...logging import traced
from .api_key import ApiKeyAuthProvider
from .base import AuthProvider
from .gateway_jwt import GatewayJwtAuthProvider


@traced("auth.get_auth_provider", record_args=False, record_result=False)
def get_auth_provider() -> AuthProvider:
    """Build the configured authentication provider."""

    cfg = get_config().auth
    if cfg.mode == "api_key":
        return ApiKeyAuthProvider(api_key_file=cfg.api_key_file)
    elif cfg.mode == "gateway_jwt":
        return GatewayJwtAuthProvider(
            secret=cfg.gateway_jwt_secret,
            issuer=cfg.gateway_jwt_issuer,
            audience=cfg.gateway_jwt_audience,
        )
    else:
        raise AuthenticationError(
            f"unsupported auth mode: {cfg.mode}, only support api_key, gateway_jwt",
            code="auth.unsupported_mode"
        )
