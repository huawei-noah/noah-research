from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "send_async_add.py"
SPEC = importlib.util.spec_from_file_location("send_async_add", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
send_async_add = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(send_async_add)


@pytest.mark.asyncio
async def test_send_async_add_requests_posts_ten_async_payloads() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "code": "ok",
                "request_id": request.headers["x-request-id"],
                "data": {"status": "queued"},
            },
        )

    transport = httpx.MockTransport(handler)

    results = await send_async_add.send_async_add_requests(
        base_url="http://testserver",
        api_key="test-key",
        count=10,
        prefix="pytest async add",
        transport=transport,
    )

    assert len(results) == 10
    assert len(requests) == 10
    assert {request.url.path for request in requests} == {"/v1/memory/add"}
    assert all(request.headers["authorization"] == "Bearer test-key" for request in requests)

    payloads = [send_async_add.json.loads(request.content) for request in requests]
    assert all(payload["mode"] == "async" for payload in payloads)
    assert [payload["messages"][0]["text"] for payload in payloads] == [
        f"pytest async add #{index}" for index in range(1, 11)
    ]
