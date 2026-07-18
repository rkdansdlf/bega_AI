from __future__ import annotations

import importlib
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

import scripts.export_openapi_contract as exporter
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


def test_uses_collision_safe_schema_anchor_registry_for_headings_and_references() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Anchor edges", "version": "1"},
        "paths": {
            "/anchors": {
                "get": {
                    "tags": ["anchors"],
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Foo"}
                                }
                            }
                        }
                    },
                }
            }
        },
        "components": {
            "schemas": {
                "Foo": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/components/schemas/foo"}},
                },
                "foo": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/components/schemas/Foo!"}},
                },
                "Foo!": {"type": "object"},
            }
        },
    }

    rendered = render_openapi_markdown(
        document,
        source_path="contracts/openapi.json",
        update_command="python scripts/export_openapi_contract.py",
    )

    anchors = {
        name: anchor
        for anchor, name in re.findall(
            r'<a id="([^"]+)"></a>\n## ([^\n]+)', rendered.schemas
        )
    }
    assert set(anchors) == {"Foo", "foo", "Foo!"}
    assert len(set(anchors.values())) == 3
    assert rendered.endpoints.count(f"[Foo](api-schemas.md#{anchors['Foo']})") == 1
    assert f"[foo](api-schemas.md#{anchors['foo']})" in rendered.schemas
    assert f"[Foo!](api-schemas.md#{anchors['Foo!']})" in rendered.schemas
    for name, anchor in anchors.items():
        assert rendered.schemas.count(f'<a id="{anchor}"></a>') == 1


def test_renders_parameter_content_without_synthesizing_examples() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Parameter content", "version": "1"},
        "paths": {
            "/search": {
                "get": {
                    "tags": ["parameters"],
                    "parameters": [
                        {
                            "name": "filter",
                            "in": "query",
                            "content": {
                                "text/plain": {
                                    "schema": {"type": "string", "default": "not-an-example"},
                                    "example": "explicit text",
                                },
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Filter"},
                                    "examples": {"valid": {"value": {"team": "LG"}}},
                                },
                            },
                        }
                    ],
                }
            }
        },
        "components": {"schemas": {"Filter": {"type": "object"}}},
    }

    rendered = render_openapi_markdown(
        document,
        source_path="contracts/openapi.json",
        update_command="python scripts/export_openapi_contract.py",
    )

    assert "#### Parameter content: `filter`" in rendered.endpoints
    assert rendered.endpoints.index("#### Media type: `application/json`") < rendered.endpoints.index("#### Media type: `text/plain`")
    assert "[Filter](api-schemas.md#filter)" in rendered.endpoints
    assert "#### Example: valid" in rendered.endpoints
    assert '"team": "LG"' in rendered.endpoints
    assert '"explicit text"' in rendered.endpoints
    assert "not-an-example" not in rendered.endpoints


def test_renders_response_links_and_preserves_only_unrendered_response_metadata() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Response links", "version": "1"},
        "paths": {
            "/widgets": {
                "post": {
                    "tags": ["responses"],
                    "responses": {
                        "201": {
                            "description": "Created",
                            "links": {
                                "z-next": {"operationId": "get_widget", "parameters": {"id": "$response.body#/id"}},
                                "a-related": {"operationRef": "#/paths/~1widgets/get"},
                            },
                            "x-trace": {"requestId": "abc"},
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

    assert rendered.endpoints.index("##### Link: a-related") < rendered.endpoints.index("##### Link: z-next")
    assert '"operationId": "get_widget"' in rendered.endpoints
    assert '"operationRef": "#/paths/~1widgets/get"' in rendered.endpoints
    assert '"x-trace": {' in rendered.endpoints
    assert '"links": {' not in rendered.endpoints


def test_retains_nested_property_schema_metadata_and_facets() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Property edges", "version": "1"},
        "paths": {},
        "components": {
            "schemas": {
                "Tag": {"type": "string"},
                "Widget": {
                    "type": "object",
                    "properties": {
                        "tags": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"$ref": "#/components/schemas/Tag"},
                            "contains": {"type": "string", "pattern": "^[A-Z]"},
                            "x-ui": {"compact": True},
                        },
                        "invalid_items": {"type": "array", "items": False},
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

    assert "| `tags` | no | `array` |  | minItems=1 |" in rendered.schemas
    assert "Items: [Tag](api-schemas.md#tag)" in rendered.schemas
    assert '"contains": {' in rendered.schemas
    assert '"x-ui": {' in rendered.schemas
    assert '"compact": true' in rendered.schemas
    assert '"items": false' in rendered.schemas


def test_renders_header_content_with_explicit_examples_only() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Header content", "version": "1"},
        "paths": {
            "/widgets": {
                "get": {
                    "tags": ["headers"],
                    "responses": {
                        "200": {
                            "headers": {
                                "X-Widget": {
                                    "content": {
                                        "text/plain": {
                                            "schema": {"type": "string", "default": "not-an-example"},
                                            "example": "explicit text",
                                        },
                                        "application/json": {
                                            "schema": {"$ref": "#/components/schemas/WidgetHeader"},
                                            "examples": {"valid": {"value": {"id": "w-1"}}},
                                        },
                                    }
                                }
                            }
                        }
                    },
                }
            }
        },
        "components": {"schemas": {"WidgetHeader": {"type": "object"}}},
    }

    rendered = render_openapi_markdown(
        document,
        source_path="contracts/openapi.json",
        update_command="python scripts/export_openapi_contract.py",
    )

    assert "#### Header content: `X-Widget`" in rendered.endpoints
    assert rendered.endpoints.index("#### Media type: `application/json`") < rendered.endpoints.index("#### Media type: `text/plain`")
    assert "[WidgetHeader](api-schemas.md#widgetheader)" in rendered.endpoints
    assert "#### Example: valid" in rendered.endpoints
    assert '"id": "w-1"' in rendered.endpoints
    assert '"explicit text"' in rendered.endpoints
    assert "not-an-example" not in rendered.endpoints


def test_retains_unrendered_parameter_metadata_without_known_field_duplicates() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Parameter metadata", "version": "1"},
        "paths": {
            "/widgets": {
                "get": {
                    "tags": ["parameters"],
                    "parameters": [{
                        "name": "mode",
                        "in": "query",
                        "required": True,
                        "description": "Mode",
                        "schema": {"type": "string"},
                        "style": "form",
                        "explode": False,
                        "allowReserved": True,
                        "deprecated": True,
                        "x-ui": {"compact": True},
                    }],
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

    assert '"allowReserved": true' in rendered.endpoints
    assert '"deprecated": true' in rendered.endpoints
    assert '"explode": false' in rendered.endpoints
    assert '"style": "form"' in rendered.endpoints
    assert '"x-ui": {' in rendered.endpoints
    assert '"name": "mode"' not in rendered.endpoints
    assert '"in": "query"' not in rendered.endpoints
    assert '"required": true' not in rendered.endpoints
    assert '"description": "Mode"' not in rendered.endpoints


def test_retains_unrendered_request_body_metadata_without_known_field_duplicates() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Request metadata", "version": "1"},
        "paths": {
            "/widgets": {
                "post": {
                    "tags": ["requests"],
                    "requestBody": {
                        "required": True,
                        "description": "Widget input",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                        "x-policy": {"audit": True},
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

    assert '"x-policy": {' in rendered.endpoints
    assert '"audit": true' in rendered.endpoints
    assert '"required": true' not in rendered.endpoints
    assert '"description": "Widget input"' not in rendered.endpoints
    assert '"content": {' not in rendered.endpoints


def test_retains_unrendered_header_metadata_without_known_field_duplicates() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Header metadata", "version": "1"},
        "paths": {
            "/widgets": {
                "get": {
                    "tags": ["headers"],
                    "responses": {
                        "200": {
                            "headers": {
                                "X-Mode": {
                                    "description": "Mode header",
                                    "schema": {"type": "string"},
                                    "deprecated": True,
                                    "style": "simple",
                                    "explode": False,
                                    "allowReserved": True,
                                    "allowEmptyValue": False,
                                    "x-ui": {"compact": True},
                                }
                            }
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

    for field in ("allowReserved", "allowEmptyValue", "deprecated", "explode", "style", "x-ui"):
        assert f'"{field}"' in rendered.endpoints
    assert '"description": "Mode header"' not in rendered.endpoints
    assert '"schema": {' not in rendered.endpoints


def test_retains_malformed_response_reference_in_stable_metadata() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Malformed response ref", "version": "1"},
        "paths": {
            "/widgets": {
                "get": {
                    "tags": ["responses"],
                    "responses": {"200": {"$ref": 42}},
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

    assert '"$ref": 42' in rendered.endpoints
    assert "- Reference:" not in rendered.endpoints


def test_retains_malformed_references_in_parameter_request_media_and_schema_paths() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Malformed reference audit", "version": "1"},
        "paths": {
            "/widgets": {
                "post": {
                    "tags": ["references"],
                    "parameters": [{
                        "name": "filter",
                        "in": "query",
                        "schema": {"type": "string"},
                        "$ref": 1,
                    }],
                    "requestBody": {
                        "$ref": 2,
                        "content": {
                            "application/json": {"schema": {"$ref": 3, "type": "string"}}
                        },
                    },
                    "responses": {
                        "200": {
                            "headers": {"X-Trace": {"$ref": 4}},
                            "content": {
                                "application/json": {"schema": {"$ref": 5, "type": "string"}}
                            },
                        }
                    },
                }
            }
        },
        "components": {"schemas": {"Broken": {"$ref": 6, "type": "string"}}},
    }

    rendered = render_openapi_markdown(
        document,
        source_path="contracts/openapi.json",
        update_command="python scripts/export_openapi_contract.py",
    )

    for value in range(1, 7):
        assert f'"$ref": {value}' in rendered.endpoints or f'"$ref": {value}' in rendered.schemas


def test_retains_valid_reference_siblings_for_parameter_header_and_media_schemas() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Reference siblings", "version": "1"},
        "paths": {
            "/widgets": {
                "get": {
                    "tags": ["references"],
                    "parameters": [{
                        "name": "filter",
                        "in": "query",
                        "schema": {
                            "$ref": "#/components/schemas/S",
                            "description": "parameter keep",
                            "default": "not-an-example",
                        },
                    }],
                    "responses": {
                        "200": {
                            "headers": {
                                "X-Shape": {
                                    "schema": {
                                        "$ref": "#/components/schemas/S",
                                        "x-header": {"keep": True},
                                    }
                                }
                            },
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/S",
                                        "x-media": {"keep": True},
                                    }
                                },
                                "text/plain": {
                                    "schema": {"$ref": "#/components/schemas/S"}
                                },
                            },
                        }
                    },
                }
            }
        },
        "components": {"schemas": {"S": {"type": "string"}}},
    }

    rendered = render_openapi_markdown(
        document,
        source_path="contracts/openapi.json",
        update_command="python scripts/export_openapi_contract.py",
    )

    assert rendered.endpoints.count("[S](api-schemas.md#s)") == 4
    assert '"description": "parameter keep"' in rendered.endpoints
    assert '"default": "not-an-example"' in rendered.endpoints
    assert '"x-header": {' in rendered.endpoints
    assert '"x-media": {' in rendered.endpoints
    assert rendered.endpoints.count('"x-header": {') == 1
    assert rendered.endpoints.count('"x-media": {') == 1
    assert 'Example: `"not-an-example"`' not in rendered.endpoints
    assert rendered.endpoints.count("Unsupported media schema siblings:") == 1


def test_retains_reference_siblings_in_component_and_nested_property_schemas() -> None:
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Schema reference siblings", "version": "1"},
        "paths": {},
        "components": {
            "schemas": {
                "S": {"type": "string"},
                "Wrapper": {
                    "$ref": "#/components/schemas/S",
                    "title": "root title",
                    "description": "root keep",
                    "default": False,
                    "x-root": {"keep": True},
                },
                "Container": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "$ref": "#/components/schemas/S",
                            "format": "nested-format",
                            "description": "nested keep",
                            "default": "nested default",
                            "x-nested": {"keep": True},
                        }
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

    assert "root keep" in rendered.schemas
    assert "nested keep" in rendered.schemas
    assert "Default: `false`" in rendered.schemas
    assert 'Default: `"nested default"`' in rendered.schemas
    assert '"x-root": {' in rendered.schemas
    assert '"x-nested": {' in rendered.schemas
    assert '"title": "root title"' in rendered.schemas
    assert '"format": "nested-format"' in rendered.schemas
    assert '"$ref": "#/components/schemas/S"' not in rendered.schemas
    assert 'Example: `"nested default"`' not in rendered.schemas
    assert rendered.schemas.count("root keep") == 1
    assert rendered.schemas.count("nested keep") == 1
    assert rendered.schemas.count("Default: `false`") == 1
    assert rendered.schemas.count('Default: `"nested default"`') == 1
    assert rendered.schemas.count('"x-root": {') == 1
    assert rendered.schemas.count('"x-nested": {') == 1


def test_build_contract_document_uses_documentation_settings_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_environment = {
        key: f"caller-{key.lower()}" for key in exporter.DOCUMENTATION_ENV
    }
    for key, value in caller_environment.items():
        monkeypatch.setenv(key, value)

    document = exporter.build_contract_document()

    operations = [
        operation
        for path_item in document["paths"].values()
        for method, operation in path_item.items()
        if method in exporter.HTTP_METHODS
    ]
    rendered = render_openapi_markdown(
        document,
        source_path="contracts/openapi.json",
        update_command=exporter.UPDATE_COMMAND,
    )

    assert operations
    assert rendered.operation_count == len(operations)
    assert "security" not in document["paths"]["/health"]["get"]
    assert document["paths"]["/ai/chat/completion"]["post"]["security"] == [
        {"InternalApiKey": []}
    ]
    assert document["components"]["securitySchemes"]["InternalApiKey"]["name"] == (
        "X-Internal-Api-Key"
    )
    assert {key: os.environ[key] for key in caller_environment} == caller_environment

    document["paths"].clear()
    assert exporter.build_contract_document()["paths"]


def test_build_contract_document_restores_environment_and_settings_cache_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main as app_main
    from app.config import get_settings

    caller_environment = {
        key: f"before-failure-{key.lower()}" for key in exporter.DOCUMENTATION_ENV
    }
    for key, value in caller_environment.items():
        monkeypatch.setenv(key, value)

    def fail_to_create_app() -> object:
        raise RuntimeError("expected app construction failure")

    monkeypatch.setattr(app_main, "create_app", fail_to_create_app)

    with pytest.raises(RuntimeError, match="expected app construction failure"):
        exporter.build_contract_document()

    assert {key: os.environ[key] for key in caller_environment} == caller_environment
    assert get_settings.cache_info().currsize == 0


def test_cli_writes_all_artifacts_and_check_reports_every_stale_path_without_writing(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact_paths = {
        "contract": tmp_path / "contracts" / "openapi.json",
        "endpoints": tmp_path / "docs" / "api-endpoints.md",
        "schemas": tmp_path / "docs" / "api-schemas.md",
    }
    monkeypatch.setattr(exporter, "CONTRACT_PATH", artifact_paths["contract"])
    monkeypatch.setattr(exporter, "ENDPOINTS_PATH", artifact_paths["endpoints"])
    monkeypatch.setattr(exporter, "SCHEMAS_PATH", artifact_paths["schemas"])

    assert exporter.main([]) == 0
    generated_bytes = {
        name: path.read_bytes() for name, path in artifact_paths.items()
    }
    assert all(content.endswith(b"\n") and not content.endswith(b"\n\n") for content in generated_bytes.values())
    assert exporter.main(["--check"]) == 0

    artifact_paths["endpoints"].write_bytes(b"stale endpoint bytes\n")
    artifact_paths["schemas"].unlink()
    stale_bytes = artifact_paths["endpoints"].read_bytes()

    assert exporter.main(["--check"]) == 1
    output = capsys.readouterr().out
    assert str(artifact_paths["endpoints"].relative_to(tmp_path)) in output
    assert str(artifact_paths["schemas"].relative_to(tmp_path)) in output
    assert artifact_paths["endpoints"].read_bytes() == stale_bytes
    assert not artifact_paths["schemas"].exists()


def _production_subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "APP_ENV": "production",
            "AI_INTERNAL_TOKEN": "production-regression-token",
            "AI_DOCS_ENABLED": "false",
            "AI_METRICS_ENABLED": "false",
            "AI_DIRECT_BROWSER_ACCESS_ENABLED": "false",
            "CORS_ORIGINS": "[]",
        }
    )
    environment.pop("PYTEST_CURRENT_TEST", None)
    environment.pop("BEGA_SKIP_APP_INIT", None)
    return environment


def test_build_contract_document_does_not_leak_documentation_package_to_clean_process() -> None:
    script = """
import importlib
import sys

assert not any(name == "app" or name.startswith("app.") for name in sys.modules)

from scripts.export_openapi_contract import build_contract_document

document = build_contract_document()
assert document["paths"]
assert not any(name == "app" or name.startswith("app.") for name in sys.modules)
assert "BEGA_SKIP_APP_INIT" not in __import__("os").environ

import app
assert app.app is app.get_app()
assert app.app.docs_url is None
assert app.app.redoc_url is None
assert app.app.openapi_url is None

app_main = importlib.import_module("app.main")
assert app_main.app.docs_url is None
assert app_main.app.redoc_url is None
assert app_main.app.openapi_url is None
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=_production_subprocess_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_build_contract_document_cleans_package_tree_after_failure_in_clean_process() -> None:
    script = """
import importlib
import sys

from scripts import export_openapi_contract as exporter

def fail_validation(parent, field):
    raise RuntimeError("expected validation failure")

exporter._require_mapping = fail_validation
try:
    exporter.build_contract_document()
except RuntimeError as error:
    assert str(error) == "expected validation failure"
else:
    raise AssertionError("generation should fail")

assert not any(name == "app" or name.startswith("app.") for name in sys.modules)

import app
assert app.app is app.get_app()
assert app.app.docs_url is None
assert app.app.openapi_url is None
app_main = importlib.import_module("app.main")
assert app_main.app.docs_url is None
assert app_main.app.openapi_url is None
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=_production_subprocess_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_build_contract_document_preserves_preimported_package_module_and_caches() -> None:
    script = """
import importlib
import sys

import app
app_main = importlib.import_module("app.main")
from scripts.export_openapi_contract import build_contract_document

modules_before = {
    name: module
    for name, module in sys.modules.items()
    if name == "app" or name.startswith("app.")
}
package_app = app.app
get_app = app.get_app
cache_info = get_app.cache_info()
main_app = app_main.app

document = build_contract_document()
assert document["paths"]

modules_after = {
    name: module
    for name, module in sys.modules.items()
    if name == "app" or name.startswith("app.")
}
assert set(modules_after) == set(modules_before)
assert all(modules_after[name] is module for name, module in modules_before.items())
assert app.app is package_app
assert app.get_app is get_app
assert get_app.cache_info() == cache_info
assert app_main.app is main_app
assert package_app.docs_url is None
assert package_app.openapi_url is None
assert main_app.docs_url is None
assert main_app.openapi_url is None
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=_production_subprocess_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_build_contract_document_serializes_concurrent_clean_process_builds() -> None:
    script = """
import importlib
import os
import sys
import threading

from scripts import export_openapi_contract as exporter

original_environment = {
    key: os.environ.get(key) for key in exporter.DOCUMENTATION_ENV
}
original_update = type(os.environ).update
original_require_mapping = exporter._require_mapping
first_documentation_env = threading.Event()
second_documentation_env = threading.Event()
second_validation = threading.Event()
allow_first = threading.Event()
allow_second = threading.Event()
first_finished = threading.Event()
results = {}
errors = []

def controlled_update(self, values, *args, **kwargs):
    if self is os.environ and values == exporter.DOCUMENTATION_ENV:
        original_update(self, values, *args, **kwargs)
        if threading.current_thread().name == "contract-one":
            first_documentation_env.set()
            assert allow_first.wait(15)
        elif threading.current_thread().name == "contract-two":
            second_documentation_env.set()
        return None
    return original_update(self, values, *args, **kwargs)

def controlled_require_mapping(parent, field):
    if threading.current_thread().name == "contract-two" and not second_validation.is_set():
        second_validation.set()
        assert allow_second.wait(15)
    return original_require_mapping(parent, field)

type(os.environ).update = controlled_update
exporter._require_mapping = controlled_require_mapping

def build(name):
    try:
        results[name] = exporter.build_contract_document()
    except BaseException as error:
        errors.append(error)
    finally:
        if name == "contract-one":
            first_finished.set()

first = threading.Thread(target=build, args=("contract-one",), name="contract-one")
second = threading.Thread(target=build, args=("contract-two",), name="contract-two")
first.start()
assert first_documentation_env.wait(5)
second.start()

overlapped = second_documentation_env.wait(2)
if overlapped:
    assert second_validation.wait(15)
    allow_first.set()
    assert first_finished.wait(15)
    allow_second.set()
else:
    allow_second.set()
    allow_first.set()

first.join(20)
second.join(20)
type(os.environ).update = original_update
exporter._require_mapping = original_require_mapping

assert not first.is_alive()
assert not second.is_alive()
assert not errors
assert set(results) == {"contract-one", "contract-two"}
assert all(document["paths"] for document in results.values())
assert {key: os.environ.get(key) for key in original_environment} == original_environment
assert not any(name == "app" or name.startswith("app.") for name in sys.modules)
assert overlapped is False

import app
assert app.app is app.get_app()
assert app.app.docs_url is None
assert app.app.openapi_url is None
app_main = importlib.import_module("app.main")
assert app_main.app.docs_url is None
assert app_main.app.openapi_url is None
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=_production_subprocess_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
