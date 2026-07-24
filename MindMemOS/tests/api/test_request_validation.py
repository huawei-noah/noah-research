from fastapi import FastAPI
from fastapi.testclient import TestClient
from mindmemos.api.app import register_exception_handlers
from mindmemos.api.schemas import AddRequest
from mindmemos.errors import BadRequestError


def test_request_validation_errors_return_one_message() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.post("/add")
    async def add(payload: AddRequest):
        return {"code": "ok", "data": None}

    response = TestClient(app).post("/add", json={"user_id": "u1", "messages": []})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "invalid_request"
    assert body["data"] is None
    assert "body.messages" in body["message"]


def test_api_error_returns_one_message() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/bad")
    async def bad():
        raise BadRequestError("top_k must be <= 100; value=101", code="search.top_k_too_large")

    response = TestClient(app).get("/bad")

    assert response.status_code == 400
    assert response.json() == {
        "code": "search.top_k_too_large",
        "message": "top_k must be <= 100; value=101",
        "data": None,
    }
