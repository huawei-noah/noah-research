"""Gateway-issued JWT authentication provider."""

from __future__ import annotations

from ...errors import AuthenticationError
from .base import ResolvedIdentity
from .internal_token import resolve_gateway_internal_token_with_config


class GatewayJwtAuthProvider:
    """Resolve short-lived internal tokens issued by the product Gateway."""

    def __init__(self, *, secret: str | None, issuer: str, audience: str):
        if not secret:
            raise AuthenticationError(
                "gateway_jwt_secret is required for gateway_jwt auth mode",
                code="auth.gateway_jwt_secret_required",
            )
        self._secret = secret
        self._issuer = issuer
        self._audience = audience

    def resolve_api_key(self, api_key: str) -> ResolvedIdentity:
        return resolve_gateway_internal_token_with_config(
            api_key,
            secret=self._secret,
            issuer=self._issuer,
            audience=self._audience,
        )
