"""HTTP transport: the single exit point for all SDK API calls.

Responsibilities (per ``docs/sdk/design.md``):

- Resolve the base URL and build request URLs.
- Inject ``Authorization: Bearer <api_key>``.
- Apply timeout and bounded retries.
- Parse the unified response envelope (``code`` / ``message`` / ``request_id`` /
  ``data``) and turn non-``ok`` results into :class:`ApiError`.
- Convert network failures into :class:`TransportError`.

The transport is deliberately decoupled from the on-disk config format: callers
resolve concrete values (base URL, api_key, timeouts) and pass them in, so the
config schema can evolve without touching this layer.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel

from .errors import ApiError, AuthRequiredError, TransportError

OK_CODE = "ok"
QUEUED_CODE = "queued"
SUCCESS_CODES = frozenset({OK_CODE, QUEUED_CODE})


class Envelope(BaseModel):
    """Parsed unified response envelope returned by every memory API."""

    code: str = OK_CODE
    message: str = ""
    request_id: str | None = None
    data: dict[str, Any] | None = None


class HttpTransport:
    """Synchronous HTTP transport over httpx with envelope handling."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        """Handle init."""
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max(0, max_retries)
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None

    def close(self) -> None:
        """Release underlying transport resources."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> HttpTransport:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def post_envelope(
        self,
        path: str,
        *,
        json: Any,
        headers: dict[str, str] | None = None,
        request_id: str | None = None,
    ) -> Envelope:
        """Send a POST request and return the parsed response envelope."""
        return self._request_envelope("POST", path, json=json, headers=headers, request_id=request_id)

    def get_envelope(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> Envelope:
        """Send a GET request and return the parsed response envelope."""
        return self._request_envelope("GET", path, params=params, request_id=request_id)

    def delete_envelope(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> Envelope:
        """Send a DELETE request and return the parsed response envelope."""
        return self._request_envelope("DELETE", path, params=params, request_id=request_id)

    def _request_envelope(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        request_id: str | None = None,
    ) -> Envelope:
        if not self._api_key:
            raise AuthRequiredError("No api_key configured. Run `mindmemos auth` first.")

        url = f"{self._base_url}{path}"
        request_headers = {"Authorization": f"Bearer {self._api_key}"}
        if request_id:
            request_headers["X-Request-Id"] = request_id
        if headers:
            request_headers.update(headers)

        response = self._send_with_retries(method, url, json=json, params=params, headers=request_headers)
        return self._parse_envelope(response)

    def _send_with_retries(
        self,
        method: str,
        url: str,
        *,
        json: Any | None,
        params: dict[str, Any] | None,
        headers: dict[str, str],
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._client.request(method, url, json=json, params=params, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                # Connection/timeout errors are safe to retry; other errors propagate.
                last_exc = exc
                if attempt >= self._max_retries:
                    break
        raise TransportError(f"Request to {url} failed after {self._max_retries + 1} attempt(s): {last_exc}")

    def _parse_envelope(self, response: httpx.Response) -> Envelope:
        return parse_envelope(response)


def parse_envelope(response: httpx.Response) -> Envelope:
    """Parse an HTTP response envelope and raise API errors for failures."""
    try:
        body = response.json()
    except ValueError as exc:
        raise ApiError(
            f"Non-JSON response (HTTP {response.status_code}): {response.text[:200]}",
            status_code=response.status_code,
        ) from exc

    if not isinstance(body, dict):
        raise ApiError(
            f"Unexpected response shape (HTTP {response.status_code}).",
            status_code=response.status_code,
            response_body=body,
        )

    code = body.get("code")
    message = body.get("message") or ""
    request_id = body.get("request_id")

    if response.status_code >= 400 or (code is not None and code not in SUCCESS_CODES):
        raise ApiError(
            message or f"API error (HTTP {response.status_code}, code={code}).",
            code=code,
            status_code=response.status_code,
            request_id=request_id,
            response_body=body,
        )

    return Envelope(
        code=code or OK_CODE,
        message=message,
        request_id=request_id,
        data=body.get("data"),
    )


class AsyncHttpTransport:
    """Async HTTP transport backed by httpx.AsyncClient."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float = 600.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Handle init."""
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max(0, max_retries)
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close the process-wide HTTP client."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncHttpTransport:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def post_envelope(
        self,
        path: str,
        *,
        json: Any,
        headers: dict[str, str] | None = None,
        request_id: str | None = None,
    ) -> Envelope:
        """Send a POST request and return the parsed response envelope."""
        return await self._request_envelope("POST", path, json=json, headers=headers, request_id=request_id)

    async def get_envelope(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> Envelope:
        """Send a GET request and return the parsed response envelope."""
        return await self._request_envelope("GET", path, params=params, request_id=request_id)

    async def delete_envelope(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> Envelope:
        """Send a DELETE request and return the parsed response envelope."""
        return await self._request_envelope("DELETE", path, params=params, request_id=request_id)

    async def _request_envelope(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        request_id: str | None = None,
    ) -> Envelope:
        if not self._api_key:
            raise AuthRequiredError("No api_key configured. Run `mindmemos auth` first.")

        url = f"{self._base_url}{path}"
        request_headers = {"Authorization": f"Bearer {self._api_key}"}
        if request_id:
            request_headers["X-Request-Id"] = request_id
        if headers:
            request_headers.update(headers)

        response = await self._send_with_retries(method, url, json=json, params=params, headers=request_headers)
        return parse_envelope(response)

    async def _send_with_retries(
        self,
        method: str,
        url: str,
        *,
        json: Any | None,
        params: dict[str, Any] | None,
        headers: dict[str, str],
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._client.request(method, url, json=json, params=params, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                # Connection/timeout errors are safe to retry; other errors propagate.
                last_exc = exc
                if attempt >= self._max_retries:
                    break
        raise TransportError(f"Request to {url} failed after {self._max_retries + 1} attempt(s): {last_exc}")
