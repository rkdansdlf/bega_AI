from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

from app.deprecation import (
    DEPRECATED_API_OPERATIONS,
    LegacyApiDeprecationMiddleware,
)


EXPECTED_DEPRECATED_OPERATIONS = {
    ("POST", "/coach/analyze", "POST", "/ai/coach/analyze"),
    ("POST", "/coach/cache/reset", "POST", "/ai/coach/cache/reset"),
    ("POST", "/coach/analyze-legacy", "POST", "/ai/coach/analyze"),
    ("POST", "/ai/coach/analyze-legacy", "POST", "/ai/coach/analyze"),
    ("POST", "/vision/ticket", "POST", "/ai/vision/ticket"),
    (
        "POST",
        "/vision/seat-view-classify",
        "POST",
        "/ai/vision/seat-view-classify",
    ),
    ("GET", "/ai/chat/stream", "POST", "/ai/chat/stream"),
}


def _operation_tuple(operation):
    return (
        operation.method,
        operation.path,
        operation.canonical_method,
        operation.canonical_path,
    )


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(LegacyApiDeprecationMiddleware)

    async def ok() -> dict[str, bool]:
        return {"ok": True}

    for operation in DEPRECATED_API_OPERATIONS:
        app.add_api_route(operation.path, ok, methods=[operation.method])
    app.add_api_route("/ai/coach/analyze", ok, methods=["POST"])
    return app


def test_deprecated_operation_registry_is_exact() -> None:
    assert {_operation_tuple(item) for item in DEPRECATED_API_OPERATIONS} == (
        EXPECTED_DEPRECATED_OPERATIONS
    )


@pytest.mark.parametrize(
    "method,path,canonical_method,canonical_path",
    sorted(EXPECTED_DEPRECATED_OPERATIONS),
)
def test_legacy_response_has_deprecation_headers(
    method: str,
    path: str,
    canonical_method: str,
    canonical_path: str,
) -> None:
    with TestClient(_build_test_app()) as client:
        response = client.request(method, path)

    assert response.status_code == 200
    assert response.headers["Deprecation"] == "true"
    assert response.headers["X-Deprecation"] == "true"
    assert response.headers["X-Legacy-Endpoint"] == f"{method} {path}"
    assert response.headers["Link"] == (
        f'<{canonical_path}>; rel="successor-version"'
    )


def test_canonical_response_has_no_deprecation_headers() -> None:
    with TestClient(_build_test_app()) as client:
        response = client.post("/ai/coach/analyze")

    assert response.status_code == 200
    assert "Deprecation" not in response.headers
    assert "Link" not in response.headers
    assert "X-Deprecation" not in response.headers
    assert "X-Legacy-Endpoint" not in response.headers


def test_existing_legacy_endpoint_header_is_preserved() -> None:
    app = FastAPI()
    app.add_middleware(LegacyApiDeprecationMiddleware)

    @app.post("/ai/coach/analyze-legacy")
    async def legacy() -> Response:
        return Response(headers={"X-Legacy-Endpoint": "analyze-legacy"})

    with TestClient(app) as client:
        response = client.post("/ai/coach/analyze-legacy")

    assert response.headers["X-Legacy-Endpoint"] == "analyze-legacy"


def test_legacy_request_emits_structured_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="app.deprecation"):
        with TestClient(_build_test_app()) as client:
            response = client.post("/coach/analyze")

    assert response.status_code == 200
    assert (
        "deprecated_api_operation method=POST legacy_path=/coach/analyze "
        "canonical_method=POST canonical_path=/ai/coach/analyze"
        in caplog.text
    )
