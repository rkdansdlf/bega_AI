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


def test_renders_endpoint_details_and_component_schemas_without_synthetic_examples() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Fixture AI", "version": "2.0.0"},
        "paths": {
            "/widgets/{id}": {
                "parameters": [
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "description": "Widget ID",
                        "schema": {"type": "string"},
                        "example": "w-1",
                    }
                ],
                "post": {
                    "tags": ["widgets"],
                    "parameters": [
                        {
                            "name": "verbose",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "boolean", "default": False},
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/WidgetRequest"
                                },
                                "examples": {
                                    "valid": {"value": {"name": "sample"}}
                                },
                            }
                        },
                    },
                    "responses": {
                        "201": {
                            "description": "Created",
                            "headers": {
                                "Location": {"schema": {"type": "string"}}
                            },
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/WidgetResponse"
                                    },
                                    "example": {"id": "w-1"},
                                }
                            },
                        },
                        "default": {"description": "Unexpected error"},
                    },
                },
            }
        },
        "components": {
            "schemas": {
                "SearchResult": {
                    "description": "Search result union",
                    "oneOf": [
                        {"$ref": "#/components/schemas/WidgetResponse"},
                        {"type": "string"},
                    ],
                },
                "WidgetRequest": {
                    "type": "object",
                    "description": "Widget creation request",
                    "required": ["name"],
                    "properties": {
                        "enabled": {"type": "boolean", "default": False},
                        "name": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 80,
                            "pattern": "^[A-Z]",
                            "enum": ["ACTIVE", "INACTIVE"],
                            "example": "WIDGET",
                        },
                    },
                },
                "WidgetResponse": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "format": "uuid"}
                    },
                },
            }
        },
    }

    rendered = render_openapi_markdown(
        document,
        source_path="contracts/openapi.json",
        update_command="python scripts/export_openapi_contract.py",
    )

    assert '| `id` | path | yes | `string` | Widget ID | `"w-1"` |' in rendered.endpoints
    assert "Required: **yes**" in rendered.endpoints
    assert "[WidgetRequest](api-schemas.md#widgetrequest)" in rendered.endpoints
    assert "#### Example: valid" in rendered.endpoints
    assert '"name": "sample"' in rendered.endpoints
    assert "### Response `201`" in rendered.endpoints
    assert "[WidgetResponse](api-schemas.md#widgetresponse)" in rendered.endpoints
    assert "Example: false" not in rendered.endpoints
    assert "Example: `false`" not in rendered.endpoints
    assert rendered.schemas.startswith("# Fixture AI Schemas\n")
    assert "Schemas: **3**" in rendered.schemas
    assert "Required properties: `name`" in rendered.schemas
    assert "| `name` | yes | `string` |" in rendered.schemas
    assert "minLength=1" in rendered.schemas
    assert "Default: `false`" in rendered.schemas
    assert 'Example: `"WIDGET"`' in rendered.schemas
    assert "Enum: `ACTIVE`, `INACTIVE`" in rendered.schemas
    assert "### Composition" in rendered.schemas
    assert "`oneOf`" in rendered.schemas


def test_merges_parameters_and_renders_only_explicit_parameter_examples() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Parameters", "version": "1"},
        "paths": {
            "/widgets/{id}": {
                "parameters": [
                    {
                        "name": "id",
                        "in": "path",
                        "example": "path-value",
                        "schema": {"type": "string"},
                    },
                    {
                        "name": "search",
                        "in": "query",
                        "description": "left|right",
                        "schema": {"type": "string", "default": "never-an-example"},
                    },
                ],
                "get": {
                    "tags": ["parameters"],
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "example": "operation-value",
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "trace",
                            "in": "header",
                            "examples": {"request": {"value": "shown"}},
                            "schema": {"type": "string", "enum": ["hidden"]},
                        },
                    ],
                    "responses": {"default": {"description": "Fallback"}, "404": {"description": "Missing"}, "201": {"description": "Created"}},
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

    assert rendered.endpoints.index("| `id` | path") < rendered.endpoints.index("| `search` | query") < rendered.endpoints.index("| `trace` | header")
    assert '`"operation-value"`' in rendered.endpoints
    assert "path-value" not in rendered.endpoints
    assert "left\\|right" in rendered.endpoints
    assert "#### Example: request" in rendered.endpoints
    assert '"shown"' in rendered.endpoints
    assert "never-an-example" not in rendered.endpoints
    assert "hidden" not in rendered.endpoints
    assert rendered.endpoints.index("### Response `201`") < rendered.endpoints.index("### Response `404`") < rendered.endpoints.index("### Response `default`")


def test_preserves_schema_examples_items_and_unsupported_fields_stably() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Schema edges", "version": "1"},
        "paths": {},
        "components": {
            "schemas": {
                "WidgetList": {
                    "type": "array",
                    "items": {"$ref": "#/components/schemas/Widget"},
                    "examples": {"empty": [], "one": [{"id": "w-1"}]},
                    "contains": {"type": "string"},
                },
                "Widget": {"type": "object"},
            }
        },
    }

    rendered = render_openapi_markdown(
        document,
        source_path="contracts/openapi.json",
        update_command="python scripts/export_openapi_contract.py",
    )

    assert "- Items: [Widget](api-schemas.md#widget)" in rendered.schemas
    assert "- Examples:" in rendered.schemas
    assert '"empty": []' in rendered.schemas
    assert '"contains": {' in rendered.schemas
    assert '"type": "string"' in rendered.schemas


def test_preserves_media_encoding_and_component_references() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Reference edges", "version": "1"},
        "paths": {
            "/upload": {
                "post": {
                    "tags": ["upload"],
                    "requestBody": {
                        "$ref": "#/components/requestBodies/UploadRequest"
                    },
                    "responses": {
                        "200": {
                            "$ref": "#/components/responses/Uploaded",
                            "headers": {
                                "X-Request": {"$ref": "#/components/headers/RequestId"}
                            },
                            "content": {
                                "multipart/form-data": {
                                    "schema": {"type": "object"},
                                    "encoding": {"file": {"contentType": "image/png"}},
                                }
                            },
                        }
                    },
                }
            }
        },
        "components": {"schemas": {}},
    }

    rendered = render_openapi_markdown(
        document,
        source_path="contracts/openapi.json",
        update_command="python scripts/export_openapi_contract.py",
    )

    assert "#/components/requestBodies/UploadRequest" in rendered.endpoints
    assert "#/components/responses/Uploaded" in rendered.endpoints
    assert "#/components/headers/RequestId" in rendered.endpoints
    assert '"encoding": {' in rendered.endpoints
