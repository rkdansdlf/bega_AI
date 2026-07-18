from __future__ import annotations

from scripts.export_openapi_contract import render_contract_json, render_openapi_markdown


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


def test_distinguishes_explicit_public_security_from_omitted_security() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Security Fixture", "version": "1"},
        "paths": {
            "/explicit-public": {
                "get": {"tags": ["security"], "security": []},
            },
            "/omitted": {
                "get": {"tags": ["security"]},
            },
        },
        "components": {"schemas": {}},
    }

    rendered = render_openapi_markdown(
        document,
        source_path="contracts/openapi.json",
        update_command="python scripts/export_openapi_contract.py",
    )

    explicit_section = rendered.endpoints.split("### GET `/explicit-public`", 1)[1]
    omitted_section = rendered.endpoints.split("### GET `/omitted`", 1)[1]
    assert "- Security: None (explicitly public)" in explicit_section
    assert "- Security: Not specified in OpenAPI" in omitted_section


def test_renders_stable_json_method_order_and_single_trailing_newlines() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Ordering Fixture", "version": "1"},
        "paths": {
            "/ordered": {
                "trace": {"tags": ["ordered"]},
                "post": {"tags": ["ordered"]},
                "get": {"tags": ["ordered"]},
            }
        },
        "components": {"schemas": {}},
    }

    rendered = render_openapi_markdown(
        document,
        source_path="contracts/openapi.json",
        update_command="python scripts/export_openapi_contract.py",
    )

    assert render_contract_json({"z": "끝", "a": {"y": 2, "x": 1}}) == (
        "{\n  \"a\": {\n    \"x\": 1,\n    \"y\": 2\n  },\n  \"z\": \"끝\"\n}\n"
    )
    assert rendered.endpoints.index("### GET `/ordered`") < rendered.endpoints.index(
        "### POST `/ordered`"
    ) < rendered.endpoints.index("### TRACE `/ordered`")
    assert rendered.endpoints.endswith("\n")
    assert not rendered.endpoints.endswith("\n\n")
    assert rendered.schemas.endswith("\n")
    assert not rendered.schemas.endswith("\n\n")
