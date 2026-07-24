from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid

import pytest
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from mindmemos.api.auth import (
    ApiKeyAuthProvider,
    GatewayJwtAuthProvider,
    get_auth_provider,
    resolve_gateway_internal_token,
)
from mindmemos.api.deps import ensure_scopes, get_internal_request_context, get_request_context
from mindmemos.api.routes import router
from mindmemos.api.schemas import AuthContext
from mindmemos.api.services import get_memory_service
from mindmemos.config import get_config, init_config, reset_config
from mindmemos.errors import ApiError, AuthenticationError, InvalidConfigError, PermissionDeniedError
from mindmemos.typing.service import AddPipelineSyncResult


def write_config(
    tmp_path,
    *,
    auth_mode: str = "api_key",
    api_key_file: str | None = None,
    gateway_jwt_secret: str | None = None,
):
    config_path = tmp_path / "dev.yaml"
    gateway_secret_line = f"  gateway_jwt_secret: {gateway_jwt_secret}\n" if gateway_jwt_secret is not None else ""
    config_path.write_text(
        f"""
auth:
  mode: {auth_mode}
  api_key_file: {api_key_file or tmp_path / "api_keys.yaml"}
{gateway_secret_line}
""",
        encoding="utf-8",
    )
    return config_path


def write_api_keys(tmp_path):
    api_key_path = tmp_path / "api_keys.yaml"
    api_key_path.write_text(
        """
api_keys:
  - key_id: key_dev_001
    api_key: sk-dev-001
    project_id: proj_dev_001
    memory_algorithm: schema
    enabled: true
    scopes: [memory:read, memory:write]
  - key_id: key_dev_002
    api_key: sk-dev-002
    project_id: proj_dev_001
    memory_algorithm: vanilla
    enabled: true
    scopes: [memory:read]
  - key_id: key_disabled
    api_key: sk-disabled
    project_id: proj_dev_002
    memory_algorithm: vanilla
    enabled: false
    scopes: [memory:read]
""",
        encoding="utf-8",
    )
    return api_key_path


def test_api_key_auth_provider_resolves_project_from_config(tmp_path) -> None:
    api_key_path = write_api_keys(tmp_path)

    provider = ApiKeyAuthProvider(api_key_file=api_key_path)
    identity = provider.resolve_api_key("sk-dev-001")

    assert identity.key_id == "key_dev_001"
    assert identity.project_id == "proj_dev_001"
    assert identity.memory_algorithm == "schema"
    assert identity.scopes == ["memory:read", "memory:write"]


def test_api_key_auth_provider_requires_memory_algorithm(tmp_path) -> None:
    api_key_path = tmp_path / "api_keys.yaml"
    api_key_path.write_text(
        """
api_keys:
  - key_id: key_dev_001
    api_key: sk-dev-001
    project_id: proj_dev_001
    enabled: true
    scopes: [memory:read, memory:write]
""",
        encoding="utf-8",
    )

    with pytest.raises(AuthenticationError) as exc_info:
        ApiKeyAuthProvider(api_key_file=api_key_path)

    assert exc_info.value.code == "auth.memory_algorithm_missing"


def test_api_key_auth_provider_rejects_unknown_key(tmp_path) -> None:
    api_key_path = write_api_keys(tmp_path)

    provider = ApiKeyAuthProvider(api_key_file=api_key_path)

    with pytest.raises(AuthenticationError) as exc_info:
        provider.resolve_api_key("missing")

    assert exc_info.value.code == "auth.invalid_api_key"


def test_api_key_auth_provider_rejects_disabled_key(tmp_path) -> None:
    api_key_path = write_api_keys(tmp_path)

    provider = ApiKeyAuthProvider(api_key_file=api_key_path)

    with pytest.raises(AuthenticationError) as exc_info:
        provider.resolve_api_key("sk-disabled")

    assert exc_info.value.code == "auth.api_key_disabled"


@pytest.mark.asyncio
async def test_get_request_context_uses_configured_api_key_project(tmp_path) -> None:
    api_key_path = write_api_keys(tmp_path)
    config_path = write_config(tmp_path, api_key_file=str(api_key_path))

    try:
        init_config(config_path=config_path)

        ctx = await get_request_context(authorization="Bearer sk-dev-002")

        # get_request_context now returns a security-only AuthContext (no actor identity).
        assert isinstance(ctx, AuthContext)
        # request_id is generated per request, not taken from a header.
        assert uuid.UUID(ctx.request_id)
        assert ctx.account_id == "memory_standalone"
        assert ctx.project_id == "proj_dev_001"
        assert ctx.api_key_uuid == "key_dev_002"
        assert ctx.memory_algorithm == "vanilla"
        assert ctx.scopes == ["memory:read"]
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_request_context_dependency_returns_security_only_context(tmp_path) -> None:
    api_key_path = write_api_keys(tmp_path)
    config_path = write_config(tmp_path, api_key_file=str(api_key_path))
    app = FastAPI()

    @app.get("/ctx")
    async def _ctx(ctx: AuthContext = Depends(get_request_context)):
        return {"project_id": ctx.project_id, "has_user_id": hasattr(ctx, "user_id")}

    try:
        init_config(config_path=config_path)

        response = TestClient(app).get("/ctx", headers={"Authorization": "Bearer sk-dev-002"})

        assert response.status_code == 200
        assert response.json() == {"project_id": "proj_dev_001", "has_user_id": False}
    finally:
        reset_config()


def test_get_auth_provider_resolves_relative_api_key_file_from_config_dir(tmp_path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    api_key_path = write_api_keys(config_dir)
    config_path = write_config(config_dir, api_key_file="api_keys.yaml")
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    try:
        init_config(config_path=config_path)

        provider = get_auth_provider()
        identity = provider.resolve_api_key("sk-dev-001")

        assert identity.key_id == "key_dev_001"
        assert identity.project_id == "proj_dev_001"
        assert str(api_key_path) != "api_keys.yaml"
    finally:
        reset_config()


def test_get_auth_provider_defaults_api_key_file_to_config_dir(tmp_path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    write_api_keys(config_dir)
    config_path = config_dir / "dev.yaml"
    config_path.write_text(
        """
auth:
  mode: api_key
""",
        encoding="utf-8",
    )
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    try:
        init_config(config_path=config_path)

        provider = get_auth_provider()
        identity = provider.resolve_api_key("sk-dev-001")

        assert identity.key_id == "key_dev_001"
    finally:
        reset_config()


def test_get_auth_provider_rejects_legacy_jwt_mode(tmp_path) -> None:
    config_path = write_config(tmp_path, auth_mode="jwt")

    try:
        with pytest.raises(InvalidConfigError, match="auth.mode"):
            init_config(config_path=config_path)
    finally:
        reset_config()


def test_get_auth_provider_returns_gateway_jwt_provider(tmp_path) -> None:
    config_path = write_config(tmp_path, auth_mode="gateway_jwt", gateway_jwt_secret="test-internal-secret")

    try:
        init_config(config_path=config_path)

        provider = get_auth_provider()

        assert isinstance(provider, GatewayJwtAuthProvider)
    finally:
        reset_config()


def test_legacy_gateway_internal_secret_is_normalized(tmp_path) -> None:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
auth:
  mode: gateway_jwt
  gateway_internal_secret: legacy-secret
  gateway_internal_issuer: mindmemos-gateway
  gateway_internal_audience: memory-data-plane
""",
        encoding="utf-8",
    )

    try:
        init_config(config_path=config_path)

        cfg = get_config()

        assert cfg.auth.gateway_jwt_secret == "legacy-secret"
        assert cfg.auth.gateway_jwt_issuer == "mindmemos-gateway"
        assert cfg.auth.gateway_jwt_audience == "memory-data-plane"
    finally:
        reset_config()


def test_get_auth_provider_rejects_none_mode_as_unsupported(tmp_path) -> None:
    config_path = write_config(tmp_path, auth_mode="none")

    try:
        with pytest.raises(InvalidConfigError, match="auth.mode"):
            init_config(config_path=config_path)
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_get_request_context_accepts_gateway_internal_token(tmp_path) -> None:
    secret = "test-internal-secret"
    config_path = write_config(tmp_path, auth_mode="gateway_jwt", gateway_jwt_secret=secret)
    token = make_gateway_token(
        secret=secret,
        account_id="acct_001",
        project_id="proj_gateway_001",
        api_key_uuid="key_gateway_001",
        scopes=["memory:read", "memory:write"],
    )

    try:
        init_config(config_path=config_path)

        ctx = await get_request_context(authorization=f"Bearer {token}")

        # AuthContext carries no actor identity; request_id is generated per request.
        assert isinstance(ctx, AuthContext)
        assert uuid.UUID(ctx.request_id)
        assert ctx.account_id == "acct_001"
        assert ctx.project_id == "proj_gateway_001"
        assert ctx.api_key_uuid == "key_gateway_001"
        assert ctx.scopes == ["memory:read", "memory:write"]
    finally:
        reset_config()


@pytest.mark.parametrize(
    "token",
    [
        "a.b.c",
        "abc..def",
        "not-ascii.负载.signature",
    ],
)
def test_gateway_internal_token_rejects_malformed_segments(tmp_path, token: str) -> None:
    config_path = write_config(tmp_path, gateway_jwt_secret="test-internal-secret")

    try:
        init_config(config_path=config_path)

        with pytest.raises(AuthenticationError) as exc_info:
            resolve_gateway_internal_token(token)

        assert exc_info.value.code == "auth.invalid_internal_token"
    finally:
        reset_config()


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        b"[]",
    ],
)
def test_gateway_internal_token_rejects_malformed_payload(tmp_path, payload: bytes) -> None:
    secret = "test-internal-secret"
    config_path = write_config(tmp_path, gateway_jwt_secret=secret)
    token = make_signed_gateway_token(
        secret=secret,
        header_segment=_json_b64({"alg": "HS256", "typ": "JWT"}),
        body_segment=_b64url(payload),
    )

    try:
        init_config(config_path=config_path)

        with pytest.raises(AuthenticationError) as exc_info:
            resolve_gateway_internal_token(token)

        assert exc_info.value.code == "auth.invalid_internal_token"
    finally:
        reset_config()


def test_gateway_internal_token_rejects_invalid_claim_types(tmp_path) -> None:
    secret = "test-internal-secret"
    config_path = write_config(tmp_path, gateway_jwt_secret=secret)
    token = make_gateway_token(
        secret=secret,
        account_id="acct_001",
        project_id="proj_gateway_001",
        api_key_uuid="key_gateway_001",
        scopes=["memory:read"],
        payload_overrides={"exp": "not-a-number"},
    )

    try:
        init_config(config_path=config_path)

        with pytest.raises(AuthenticationError) as exc_info:
            resolve_gateway_internal_token(token)

        assert exc_info.value.code == "auth.invalid_internal_token"
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_get_request_context_rejects_gateway_token_in_api_key_mode(tmp_path) -> None:
    api_key_path = write_api_keys(tmp_path)
    secret = "test-internal-secret"
    config_path = write_config(tmp_path, api_key_file=str(api_key_path), gateway_jwt_secret=secret)
    token = make_gateway_token(
        secret=secret,
        account_id="acct_001",
        project_id="proj_gateway_001",
        api_key_uuid="key_gateway_001",
        scopes=["memory:read"],
    )

    try:
        init_config(config_path=config_path)

        with pytest.raises(AuthenticationError) as exc_info:
            await get_request_context(authorization=f"Bearer {token}")

        assert exc_info.value.code == "auth.invalid_api_key"
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_get_request_context_rejects_api_key_in_gateway_jwt_mode(tmp_path) -> None:
    api_key_path = write_api_keys(tmp_path)
    config_path = write_config(
        tmp_path,
        auth_mode="gateway_jwt",
        api_key_file=str(api_key_path),
        gateway_jwt_secret="test-internal-secret",
    )

    try:
        init_config(config_path=config_path)

        with pytest.raises(AuthenticationError) as exc_info:
            await get_request_context(authorization="Bearer sk-dev-001")

        assert exc_info.value.code == "auth.invalid_internal_token"
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_get_internal_request_context_rejects_api_key_mode(tmp_path) -> None:
    api_key_path = write_api_keys(tmp_path)
    config_path = write_config(tmp_path, api_key_file=str(api_key_path), gateway_jwt_secret="test-internal-secret")

    try:
        init_config(config_path=config_path)

        with pytest.raises(AuthenticationError) as exc_info:
            await get_internal_request_context(authorization="Bearer sk-dev-001")

        assert exc_info.value.code == "auth.unsupported_mode"
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_get_internal_request_context_rejects_standalone_api_key_in_gateway_jwt_mode(tmp_path) -> None:
    api_key_path = write_api_keys(tmp_path)
    config_path = write_config(
        tmp_path,
        auth_mode="gateway_jwt",
        api_key_file=str(api_key_path),
        gateway_jwt_secret="test-internal-secret",
    )

    try:
        init_config(config_path=config_path)

        with pytest.raises(AuthenticationError) as exc_info:
            await get_internal_request_context(authorization="Bearer sk-dev-001")

        assert exc_info.value.code == "auth.invalid_internal_token"
    finally:
        reset_config()


def test_ensure_scopes_rejects_read_only_context_for_write_operation() -> None:
    ctx = make_context(scopes=["memory:read"])

    with pytest.raises(PermissionDeniedError) as exc_info:
        ensure_scopes(ctx, ("memory:write",))

    assert exc_info.value.status_code == 403
    assert exc_info.value.code == "auth.insufficient_scope"


def test_memory_add_route_rejects_read_only_scope() -> None:
    app = FastAPI()

    @app.exception_handler(ApiError)
    async def _handle_api_error(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "data": None},
        )

    class FakeService:
        async def add(self, ctx, payload):
            return AddPipelineSyncResult(status="ok", memories=[])

    app.include_router(router)
    app.dependency_overrides[get_request_context] = lambda: make_context(scopes=["memory:read"])
    app.dependency_overrides[get_memory_service] = lambda: FakeService()

    response = TestClient(app).post("/v1/memory/add", json={"messages": [{"text": "blocked"}]})

    assert response.status_code == 403
    assert response.json()["code"] == "auth.insufficient_scope"


def make_context(*, scopes: list[str]) -> AuthContext:
    return AuthContext(
        request_id="00000000-0000-0000-0000-000000000123",
        account_id="memory_standalone",
        project_id="proj_dev_001",
        api_key_uuid="key_dev_002",
        memory_algorithm="vanilla",
        scopes=scopes,
    )


def make_gateway_token(
    *,
    secret: str,
    account_id: str,
    project_id: str,
    api_key_uuid: str,
    scopes: list[str],
    payload_overrides: dict | None = None,
) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": "mindmemos-gateway",
        "aud": "memory-data-plane",
        "sub": account_id,
        "account_id": account_id,
        "project_id": project_id,
        "api_key_uuid": api_key_uuid,
        "memory_algorithm": "schema",
        "scopes": scopes,
        "iat": now,
        "exp": now + 60,
    }
    if payload_overrides:
        payload.update(payload_overrides)
    return make_signed_gateway_token(
        secret=secret,
        header_segment=_json_b64(header),
        body_segment=_json_b64(payload),
    )


def make_signed_gateway_token(*, secret: str, header_segment: str, body_segment: str) -> str:
    signing_input = f"{header_segment}.{body_segment}"
    sig = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(sig)}"


def _json_b64(value) -> str:
    return _b64url(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")
