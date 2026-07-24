"""Gateway internal token authentication.

These helpers verify the HTTP-only contract between the commercial Gateway and
the Memory Data Plane. They intentionally do not import any account-system code.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
from typing import Any, NoReturn

from ...config import get_config
from ...errors import AuthenticationError
from ..algorithm import resolve_memory_algorithm
from .base import ResolvedIdentity

INVALID_INTERNAL_TOKEN_CODE = "auth.invalid_internal_token"


def resolve_gateway_internal_token(token: str) -> ResolvedIdentity | None:
    """Resolve a Gateway internal JWT-like token, or return None for non-JWT input."""

    if token.count(".") != 2:
        return None
    cfg = get_config().auth
    if not cfg.gateway_jwt_secret:
        return None

    return resolve_gateway_internal_token_with_config(
        token,
        secret=cfg.gateway_jwt_secret,
        issuer=cfg.gateway_jwt_issuer,
        audience=cfg.gateway_jwt_audience,
    )


def resolve_gateway_internal_token_with_config(
    token: str,
    *,
    secret: str,
    issuer: str,
    audience: str,
) -> ResolvedIdentity:
    """Resolve a Gateway internal token with explicit verification settings."""

    if token.count(".") != 2:
        raise AuthenticationError("expected gateway internal token", code="auth.invalid_internal_token")
    payload = _verify_hmac_token(
        token,
        secret=secret,
        issuer=issuer,
        audience=audience,
    )
    project_id = _required_string_claim(payload, "project_id")
    account_id = _required_string_claim(payload, "account_id", "sub")
    api_key_uuid = _required_string_claim(payload, "api_key_uuid", "jti")
    memory_algorithm = resolve_memory_algorithm(_required_string_claim(payload, "memory_algorithm"))
    return ResolvedIdentity(
        key_id=api_key_uuid,
        project_id=project_id,
        memory_algorithm=memory_algorithm,
        account_id=account_id,
        scopes=_string_list_claim(payload, "scopes"),
    )


def _verify_hmac_token(token: str, *, secret: str, issuer: str, audience: str) -> dict[str, Any]:
    try:
        head, body, sig = token.split(".", 2)
    except ValueError as exc:
        _raise_invalid_internal_token(cause=exc)
    if not head or not body or not sig:
        _raise_invalid_internal_token()

    header = _decode_json_object_segment(head)
    if header.get("alg") != "HS256":
        _raise_invalid_internal_token("invalid gateway internal token algorithm")

    signing_input = f"{head}.{body}"
    expected = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    if not hmac.compare_digest(_decode_b64url_segment(sig), expected):
        _raise_invalid_internal_token("invalid gateway internal token signature")

    payload = _decode_json_object_segment(body)
    if _required_int_claim(payload, "exp") < int(time.time()):
        raise AuthenticationError("gateway internal token expired", code="auth.internal_token_expired")
    if payload.get("iss") != issuer:
        _raise_invalid_internal_token("invalid gateway internal token issuer")
    if payload.get("aud") != audience:
        _raise_invalid_internal_token("invalid gateway internal token audience")
    return payload


def _decode_b64url_segment(value: str) -> bytes:
    if not value:
        _raise_invalid_internal_token()
    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode((value + padding).encode("ascii"), altchars=b"-_", validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        _raise_invalid_internal_token(cause=exc)


def _decode_json_object_segment(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(_decode_b64url_segment(value).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _raise_invalid_internal_token(cause=exc)
    if not isinstance(parsed, dict):
        _raise_invalid_internal_token("invalid gateway internal token JSON")
    return parsed


def _required_string_claim(payload: dict[str, Any], *names: str) -> str:
    for name in names:
        value = payload.get(name)
        if value in (None, ""):
            continue
        if isinstance(value, str):
            return value
        _raise_invalid_internal_token("invalid gateway internal token claims")
    _raise_invalid_internal_token("invalid gateway internal token claims")


def _optional_string_claim(payload: dict[str, Any], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    _raise_invalid_internal_token("invalid gateway internal token claims")


def _string_list_claim(payload: dict[str, Any], name: str) -> list[str]:
    value = payload.get(name, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _raise_invalid_internal_token("invalid gateway internal token claims")
    return value


def _required_int_claim(payload: dict[str, Any], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        _raise_invalid_internal_token("invalid gateway internal token claims")
    return value


def _raise_invalid_internal_token(
    message: str = "invalid gateway internal token",
    *,
    cause: BaseException | None = None,
) -> NoReturn:
    if cause is None:
        raise AuthenticationError(message, code=INVALID_INTERNAL_TOKEN_CODE)
    raise AuthenticationError(message, code=INVALID_INTERNAL_TOKEN_CODE) from cause
