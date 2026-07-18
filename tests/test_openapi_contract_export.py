from __future__ import annotations

from scripts.export_openapi_contract import render_openapi_markdown


def test_renders_operations_deterministically_with_security_and_deprecation() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Fixture AI", "version": "2.0.0"},
        "paths": {
            "/zeta": {
                "post": {
                    "tags": ["z-tag"],
                    "summary": "Create zeta",
                    "operationId": "create_zeta",
                    "deprecated": True,
                    "security": [{"InternalApiKey": []}],
                    "responses": {"204": {"description": "Created"}},
                },
                "get": {
                    "tags": ["a-tag"],
                    "summary": "Read zeta",
                    "operationId": "read_zeta",
                    "responses": {"200": {"description": "OK"}},
                },
            }
        },
        "components": {"schemas": {}},
    }

    rendered = render_openapi_markdown(
        document,
        source_path="contracts/openapi.json",
        update_command="python scripts/export_openapi_contract.py",
    )

    assert rendered.operation_count == 2
    assert rendered.schema_count == 0
    assert rendered.endpoints.startswith("# Fixture AI Endpoints\n")
    assert "Operations: **2**" in rendered.endpoints
    assert rendered.endpoints.index("## a-tag") < rendered.endpoints.index("## z-tag")
    assert "### GET `/zeta`" in rendered.endpoints
    assert "- Security: Not specified in OpenAPI" in rendered.endpoints
    assert "### POST `/zeta`" in rendered.endpoints
    assert "- Security: `InternalApiKey`" in rendered.endpoints
    assert "- Deprecated: yes" in rendered.endpoints
    assert rendered.endpoints.endswith("\n")
    assert not rendered.endpoints.endswith("\n\n")


def test_rejects_malformed_openapi_instead_of_rendering_partial_docs() -> None:
    try:
        render_openapi_markdown(
            {"openapi": "3.1.0", "info": {"title": "Broken", "version": "1"}},
            source_path="contracts/openapi.json",
            update_command="python scripts/export_openapi_contract.py",
        )
    except ValueError as error:
        assert str(error) == "OpenAPI field must be an object: paths"
    else:
        raise AssertionError("malformed OpenAPI must fail closed")
