"""Export complete FastAPI OpenAPI JSON and deterministic Markdown docs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONTRACT_PATH = ROOT / "contracts" / "openapi.json"
ENDPOINTS_PATH = ROOT / "docs" / "api-endpoints.md"
SCHEMAS_PATH = ROOT / "docs" / "api-schemas.md"
UPDATE_COMMAND = "python scripts/export_openapi_contract.py"
HTTP_METHODS = ("get", "post", "put", "patch", "delete", "options", "head", "trace")
DOCUMENTATION_ENV = {
    "APP_ENV": "local",
    "AI_INTERNAL_TOKEN": "openapi-contract-generation-token",
    "AI_DOCS_ENABLED": "true",
    "AI_METRICS_ENABLED": "false",
    "AI_DIRECT_BROWSER_ACCESS_ENABLED": "false",
    "CORS_ORIGINS": "[]",
    "BEGA_SKIP_APP_INIT": "1",
}


@dataclass(frozen=True)
class RenderedDocuments:
    endpoints: str
    schemas: str
    operation_count: int
    schema_count: int


@dataclass(frozen=True)
class _Operation:
    path: str
    method: str
    definition: Mapping[str, Any]
    tag: str


def render_contract_json(document: Mapping[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_openapi_markdown(
    document: Mapping[str, Any],
    source_path: str,
    update_command: str,
) -> RenderedDocuments:
    paths = _require_mapping(document, "paths")
    components = document.get("components", {})
    if not isinstance(components, Mapping):
        raise ValueError("OpenAPI field must be an object: components")
    schemas = _require_mapping(components, "schemas")
    schema_anchors = _schema_anchor_registry(schemas)
    operations = _collect_operations(paths)
    endpoints = _render_endpoints(
        document, source_path, update_command, paths, operations, schema_anchors
    )
    schema_docs = _render_schemas(
        document, source_path, update_command, schemas, schema_anchors
    )
    return RenderedDocuments(endpoints, schema_docs, len(operations), len(schemas))


def _require_mapping(parent: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = parent.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"OpenAPI field must be an object: {field}")
    return value


def _collect_operations(paths: Mapping[str, Any]) -> list[_Operation]:
    operations: list[_Operation] = []
    method_order = {method: index for index, method in enumerate(HTTP_METHODS)}

    for path, path_item in paths.items():
        if not isinstance(path, str) or not isinstance(path_item, Mapping):
            raise ValueError("OpenAPI paths entries must be objects")
        for method in HTTP_METHODS:
            definition = path_item.get(method)
            if definition is None:
                continue
            if not isinstance(definition, Mapping):
                raise ValueError(f"OpenAPI operation must be an object: {method} {path}")
            tags = definition.get("tags", [])
            if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
                raise ValueError(f"OpenAPI operation tags must be an array of strings: {method} {path}")
            operations.append(
                _Operation(path=path, method=method, definition=definition, tag=tags[0] if tags else "untagged")
            )

    return sorted(
        operations,
        key=lambda operation: (
            operation.tag.lower(),
            operation.path,
            method_order[operation.method],
        ),
    )


def _render_endpoints(
    document: Mapping[str, Any],
    source_path: str,
    update_command: str,
    paths: Mapping[str, Any],
    operations: list[_Operation],
    schema_anchors: Mapping[str, str],
) -> str:
    title, version = _title_and_version(document)
    lines = [
        f"# {title} Endpoints",
        "",
        "> This file is generated. Do not edit directly.",
        f"> Source: `{source_path}`",
        f"> Regenerate with: `{update_command}`",
        "",
        f"Version: `{version}`",
        f"Paths: **{len(paths)}**",
        f"Operations: **{len(operations)}**",
    ]

    current_tag: str | None = None
    for operation in operations:
        if operation.tag != current_tag:
            lines.extend(["", f"## {operation.tag}"])
            current_tag = operation.tag
        definition = operation.definition
        lines.extend(["", f"### {operation.method.upper()} `{operation.path}`"])
        if isinstance(definition.get("summary"), str):
            lines.append(definition["summary"])
        if isinstance(definition.get("description"), str):
            lines.extend(["", definition["description"]])
        operation_id = definition.get("operationId")
        if isinstance(operation_id, str):
            lines.append(f"- Operation ID: `{operation_id}`")
        tags = definition.get("tags", [])
        if tags:
            lines.append(f"- Tags: {', '.join(f'`{tag}`' for tag in tags)}")
        lines.append(f"- Security: {_security_label(definition)}")
        lines.append(f"- Deprecated: {'yes' if definition.get('deprecated') is True else 'no'}")

        path_item = paths[operation.path]
        if not isinstance(path_item, Mapping):
            raise ValueError("OpenAPI paths entries must be objects")
        _append_parameters(
            lines, _merged_parameters(path_item, definition), schema_anchors
        )
        _append_request_body(lines, definition.get("requestBody"), schema_anchors)
        _append_responses(lines, definition.get("responses"), schema_anchors)

    return _finish(lines)


def _security_label(definition: Mapping[str, Any]) -> str:
    if "security" not in definition:
        return "Not specified in OpenAPI"
    security = definition["security"]
    if not isinstance(security, list):
        raise ValueError("OpenAPI operation security must be an array")
    if not security:
        return "None (explicitly public)"

    requirements: list[str] = []
    for requirement in security:
        if not isinstance(requirement, Mapping):
            raise ValueError("OpenAPI security requirement must be an object")
        schemes: list[str] = []
        for name in sorted(requirement):
            scopes = requirement[name]
            if not isinstance(name, str) or not isinstance(scopes, list) or any(
                not isinstance(scope, str) for scope in scopes
            ):
                raise ValueError("OpenAPI security requirement must map schemes to scope arrays")
            suffix = f" ({', '.join(scopes)})" if scopes else ""
            schemes.append(f"`{name}`{suffix}")
        requirements.append(" + ".join(schemes) if schemes else "(none)")
    return " OR ".join(requirements)


def _merged_parameters(
    path_item: Mapping[str, Any], operation: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    merged: dict[tuple[str, str], Mapping[str, Any]] = {}
    for owner in (path_item, operation):
        parameters = owner.get("parameters", [])
        if not isinstance(parameters, list):
            raise ValueError("OpenAPI parameters must be an array")
        for parameter in parameters:
            if not isinstance(parameter, Mapping):
                raise ValueError("OpenAPI parameter must be an object")
            location = parameter.get("in")
            name = parameter.get("name")
            if not isinstance(location, str) or not isinstance(name, str):
                raise ValueError("OpenAPI parameter must include string in and name")
            merged[(location, name)] = parameter
    location_order = {"path": 0, "query": 1, "header": 2, "cookie": 3}
    return [
        parameter
        for _, parameter in sorted(
            merged.items(),
            key=lambda item: (
                location_order.get(item[0][0], len(location_order)),
                item[0][0].lower(),
                item[0][1].lower(),
                item[0][1],
            ),
        )
    ]


def _append_parameters(
    lines: list[str],
    parameters: list[Mapping[str, Any]],
    schema_anchors: Mapping[str, str],
) -> None:
    if not parameters:
        return
    lines.extend(["", "#### Parameters", "", "| Name | In | Required | Schema | Description | Example |", "| --- | --- | --- | --- | --- | --- |"])
    for parameter in parameters:
        name = parameter["name"]
        location = parameter["in"]
        description = parameter.get("description", "")
        example = f"`{_inline_json(parameter['example'])}`" if "example" in parameter else ""
        lines.append(
            "| "
            f"`{_escape_code(name)}` | {_escape_cell(location)} | "
            f"{'yes' if parameter.get('required') is True else 'no'} | "
            f"{_schema_label(parameter.get('schema'), schema_anchors=schema_anchors)} | "
            f"{_escape_cell(description) if isinstance(description, str) else _escape_cell(_stable_json(description))} | "
            f"{_escape_cell(example)} |"
        )
        _append_reference_siblings(
            lines, f"parameter {_escape_code(name)} schema siblings", parameter.get("schema")
        )
        _append_examples(lines, parameter)
        if "content" in parameter:
            lines.extend(["", f"#### Parameter content: `{_escape_code(name)}`"])
            _append_content(lines, parameter["content"], schema_anchors)
        represented = {"name", "in", "required", "description", "example", "examples"}
        if "schema" in parameter and _schema_is_rendered(parameter["schema"]):
            represented.add("schema")
        if "content" in parameter and parameter["content"] is not None:
            represented.add("content")
        metadata = {
            key: value for key, value in parameter.items() if key not in represented
        }
        if metadata:
            _append_unsupported(
                lines, f"parameter {_escape_code(name)} metadata", metadata
            )


def _append_request_body(
    lines: list[str], request_body: object, schema_anchors: Mapping[str, str]
) -> None:
    if request_body is None:
        return
    lines.extend(["", "### Request body"])
    if not isinstance(request_body, Mapping):
        _append_unsupported(lines, "request body", request_body)
        return
    lines.append(f"- Required: **{'yes' if request_body.get('required') is True else 'no'}**")
    represented: set[str] = set()
    if isinstance(request_body.get("required"), bool):
        represented.add("required")
    if isinstance(request_body.get("$ref"), str):
        lines.append(
            f"- Reference: {_schema_label({'$ref': request_body['$ref']}, schema_anchors=schema_anchors)}"
        )
        represented.add("$ref")
    if isinstance(request_body.get("description"), str):
        lines.append(request_body["description"])
        represented.add("description")
    _append_content(lines, request_body.get("content"), schema_anchors)
    if "content" in request_body and request_body["content"] is not None:
        represented.add("content")
    metadata = {
        key: value for key, value in request_body.items() if key not in represented
    }
    if metadata:
        _append_unsupported(lines, "request body metadata", metadata)


def _append_responses(
    lines: list[str], responses: object, schema_anchors: Mapping[str, str]
) -> None:
    if responses is None:
        return
    lines.extend(["", "### Responses"])
    if not isinstance(responses, Mapping):
        _append_unsupported(lines, "responses", responses)
        return
    for status, response in sorted(responses.items(), key=_response_sort_key):
        label = str(status)
        lines.extend(["", f"### Response `{_escape_code(label)}`"])
        if not isinstance(response, Mapping):
            _append_unsupported(lines, "response", response)
            continue
        description = response.get("description")
        if isinstance(description, str):
            lines.append(description)
        elif description is not None:
            _append_unsupported(lines, "response description", description)
        if isinstance(response.get("$ref"), str):
            lines.append(
                f"- Reference: {_schema_label({'$ref': response['$ref']}, schema_anchors=schema_anchors)}"
            )
        represented: set[str] = set()
        if isinstance(description, str) or description is not None:
            represented.add("description")
        if isinstance(response.get("$ref"), str):
            represented.add("$ref")
        _append_headers(lines, response.get("headers"), schema_anchors)
        if "headers" in response and response["headers"] is not None:
            represented.add("headers")
        _append_content(lines, response.get("content"), schema_anchors)
        if "content" in response and response["content"] is not None:
            represented.add("content")
        _append_response_links(lines, response.get("links"))
        if "links" in response and response["links"] is not None:
            represented.add("links")
        metadata = {
            key: value for key, value in response.items() if key not in represented
        }
        if metadata:
            _append_unsupported(lines, "response metadata", metadata)


def _response_sort_key(item: tuple[object, object]) -> tuple[int, int, str]:
    status = str(item[0])
    if status.isdigit():
        return (0, int(status), status)
    if status == "default":
        return (1, 0, status)
    return (2, 0, status)


def _append_headers(
    lines: list[str], headers: object, schema_anchors: Mapping[str, str]
) -> None:
    if headers is None:
        return
    lines.extend(["", "#### Headers"])
    if not isinstance(headers, Mapping):
        _append_unsupported(lines, "response headers", headers)
        return
    lines.extend(["", "| Name | Required | Schema | Description | Example |", "| --- | --- | --- | --- | --- |"])
    for name, header in sorted(headers.items(), key=lambda item: str(item[0]).lower()):
        if not isinstance(header, Mapping):
            lines.append(f"| `{_escape_code(str(name))}` |  |  | {_escape_cell(_stable_json(header))} |  |")
            continue
        description = header.get("description", "")
        example = f"`{_inline_json(header['example'])}`" if "example" in header else ""
        header_schema = header.get("schema")
        if header_schema is None and isinstance(header.get("$ref"), str):
            header_schema = {"$ref": header["$ref"]}
        lines.append(
            "| "
            f"`{_escape_code(str(name))}` | {'yes' if header.get('required') is True else 'no'} | "
            f"{_schema_label(header_schema, schema_anchors=schema_anchors)} | "
            f"{_escape_cell(description) if isinstance(description, str) else _escape_cell(_stable_json(description))} | "
            f"{_escape_cell(example)} |"
        )
        _append_reference_siblings(
            lines, f"header {_escape_code(str(name))} schema siblings", header_schema
        )
        _append_examples(lines, header)
        if "content" in header:
            lines.extend(["", f"#### Header content: `{_escape_code(str(name))}`"])
            _append_content(lines, header["content"], schema_anchors)
        represented = {"description", "required", "example", "examples"}
        if "schema" in header and _schema_is_rendered(header["schema"]):
            represented.add("schema")
        if "content" in header and header["content"] is not None:
            represented.add("content")
        if isinstance(header.get("$ref"), str):
            represented.add("$ref")
        metadata = {key: value for key, value in header.items() if key not in represented}
        if metadata:
            _append_unsupported(
                lines, f"header {_escape_code(str(name))} metadata", metadata
            )


def _append_content(
    lines: list[str], content: object, schema_anchors: Mapping[str, str]
) -> None:
    if content is None:
        return
    if not isinstance(content, Mapping):
        _append_unsupported(lines, "content", content)
        return
    for media_type, media in sorted(content.items(), key=lambda item: str(item[0])):
        lines.extend(["", f"#### Media type: `{_escape_code(str(media_type))}`"])
        if not isinstance(media, Mapping):
            _append_unsupported(lines, "media type", media)
            continue
        if "schema" in media:
            lines.append(
                f"- Schema: {_schema_label(media['schema'], schema_anchors=schema_anchors)}"
            )
            _append_reference_siblings(
                lines, "media schema siblings", media["schema"]
            )
        _append_examples(lines, media)
        represented = {"example", "examples"}
        if "schema" in media and _schema_is_rendered(media["schema"]):
            represented.add("schema")
        unsupported = {key: value for key, value in media.items() if key not in represented}
        if unsupported:
            _append_unsupported(lines, "media type fields", unsupported)


def _append_response_links(lines: list[str], links: object) -> None:
    if links is None:
        return
    lines.extend(["", "#### Links"])
    if not isinstance(links, Mapping):
        _append_unsupported(lines, "response links", links)
        return
    for name, link in sorted(links.items(), key=lambda item: str(item[0])):
        lines.extend(
            [
                "",
                f"##### Link: {_escape_cell(str(name))}",
                "",
                "```json",
                _stable_json(link),
                "```",
            ]
        )


def _append_examples(lines: list[str], owner: Mapping[str, Any]) -> None:
    if "example" in owner:
        lines.extend(["", "#### Example", "", "```json", _stable_json(owner["example"]), "```"])
    if "examples" not in owner:
        return
    examples = owner["examples"]
    if not isinstance(examples, Mapping):
        _append_unsupported(lines, "examples", examples)
        return
    for name, example in sorted(examples.items(), key=lambda item: str(item[0])):
        if not isinstance(example, Mapping):
            _append_unsupported(lines, f"example {_escape_code(str(name))}", example)
            continue
        if "value" in example:
            lines.extend(
                [
                    "",
                    f"#### Example: {_escape_cell(str(name))}",
                    "",
                    "```json",
                    _stable_json(example["value"]),
                    "```",
                ]
            )
        else:
            _append_unsupported(lines, f"example {_escape_code(str(name))}", example)


def _schema_label(
    schema: object,
    schema_document: str = "api-schemas.md",
    schema_anchors: Mapping[str, str] | None = None,
) -> str:
    if not isinstance(schema, Mapping):
        return "-" if schema is None else f"`{_escape_code(_stable_json(schema))}`"
    reference = schema.get("$ref")
    if isinstance(reference, str):
        prefix = "#/components/schemas/"
        if reference.startswith(prefix):
            name = reference.removeprefix(prefix)
            anchor = schema_anchors.get(name, _anchor(name)) if schema_anchors else _anchor(name)
            return f"[{_escape_cell(name)}]({schema_document}#{anchor})"
        return f"`{_escape_code(reference)}`"
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        label = schema_type
        if isinstance(schema.get("format"), str):
            label = f"{label} ({schema['format']})"
        return f"`{_escape_code(label)}`"
    return f"`{_escape_code(_stable_json(schema))}`"


def _schema_is_rendered(schema: object) -> bool:
    if schema is None:
        return False
    if isinstance(schema, Mapping) and "$ref" in schema:
        return isinstance(schema["$ref"], str)
    return True


def _append_reference_siblings(
    lines: list[str],
    label: str,
    schema: object,
    rendered_keys: set[str] | frozenset[str] = frozenset(),
) -> bool:
    if not isinstance(schema, Mapping) or not isinstance(schema.get("$ref"), str):
        return False
    siblings = {
        key: value
        for key, value in schema.items()
        if key != "$ref" and key not in rendered_keys
    }
    if not siblings:
        return False
    _append_unsupported(lines, label, siblings)
    return True


def _root_schema_rendered_keys(schema: Mapping[str, Any]) -> set[str]:
    rendered: set[str] = set()
    if isinstance(schema.get("description"), str):
        rendered.add("description")
    if "required" in schema and schema["required"] is not None:
        rendered.add("required")
    rendered.update(
        key
        for key in (
            "minLength", "maxLength", "pattern", "minimum", "maximum",
            "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minItems",
            "maxItems", "uniqueItems", "minProperties", "maxProperties",
        )
        if key in schema
    )
    rendered.update(
        key
        for key in ("default", "enum", "const", "example", "examples", "items")
        if key in schema
    )
    rendered.update(
        key
        for key in ("nullable", "readOnly", "writeOnly", "deprecated")
        if schema.get(key) is True
    )
    rendered.update(
        key
        for key in (
            "properties", "allOf", "anyOf", "oneOf", "not", "discriminator",
            "additionalProperties",
        )
        if key in schema
    )
    if not isinstance(schema.get("$ref"), str):
        rendered.update(key for key in ("type", "format") if key in schema)
    return rendered


def _stable_json(value: object) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True)


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return f"<unsupported {type(value).__name__}>"


def _inline_json(value: object) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True)


def _escape_code(value: object) -> str:
    return str(value).replace("`", "\\`").replace("\n", " ")


def _escape_cell(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _append_unsupported(lines: list[str], label: str, value: object) -> None:
    lines.extend(["", f"Unsupported {label}:", "", "```json", _stable_json(value), "```"])


def _render_schemas(
    document: Mapping[str, Any],
    source_path: str,
    update_command: str,
    schemas: Mapping[str, Any],
    schema_anchors: Mapping[str, str],
) -> str:
    title, version = _title_and_version(document)
    lines = [
        f"# {title} Schemas",
        "",
        "> This file is generated. Do not edit directly.",
        f"> Source: `{source_path}`",
        f"> Regenerate with: `{update_command}`",
        "",
        f"Version: `{version}`",
        f"Schemas: **{len(schemas)}**",
    ]
    for name, schema in sorted(schemas.items(), key=lambda item: str(item[0]).lower()):
        _append_schema(lines, str(name), schema, schema_anchors)
    return _finish(lines)


def _append_schema(
    lines: list[str], name: str, schema: object, schema_anchors: Mapping[str, str]
) -> None:
    lines.extend(["", f'<a id="{schema_anchors[name]}"></a>', f"## {name}"])
    if not isinstance(schema, Mapping):
        _append_unsupported(lines, "schema", schema)
        return
    if isinstance(schema.get("description"), str):
        lines.extend(["", schema["description"]])
    lines.append(f"- Type: {_schema_label(schema, schema_anchors=schema_anchors)}")
    required = schema.get("required")
    if isinstance(required, list) and all(isinstance(item, str) for item in required):
        if required:
            lines.append(f"- Required properties: {', '.join(f'`{_escape_code(item)}`' for item in sorted(required))}")
    elif required is not None:
        _append_unsupported(lines, "required properties", required)
    constraints = _constraints_label(schema)
    if constraints:
        lines.append(f"- Constraints: {constraints}")
    if "default" in schema:
        lines.append(f"- Default: `{_inline_json(schema['default'])}`")
    if "enum" in schema:
        enum = schema["enum"]
        if isinstance(enum, list):
            lines.append(f"- Enum: {', '.join(f'`{_enum_value(item)}`' for item in enum)}")
        else:
            _append_unsupported(lines, "enum", enum)
    if "const" in schema:
        lines.append(f"- Const: `{_inline_json(schema['const'])}`")
    flags = [name for name in ("nullable", "readOnly", "writeOnly", "deprecated") if schema.get(name) is True]
    if flags:
        lines.append(f"- Flags: {', '.join(f'`{flag}`' for flag in flags)}")
    if "example" in schema:
        lines.append(f"- Example: `{_inline_json(schema['example'])}`")
    if "examples" in schema:
        lines.extend(["- Examples:", "", "```json", _stable_json(schema["examples"]), "```"])
    if "items" in schema:
        items = schema["items"]
        if isinstance(items, (Mapping, list)):
            if isinstance(items, list):
                lines.append(
                    f"- Items: {', '.join(_schema_label(item, schema_anchors=schema_anchors) for item in items)}"
                )
            else:
                lines.append(
                    f"- Items: {_schema_label(items, schema_anchors=schema_anchors)}"
                )
        else:
            _append_unsupported(lines, "items", items)
    _append_property_table(lines, schema, schema_anchors)
    _append_composition(lines, schema, schema_anchors)
    if "additionalProperties" in schema:
        value = schema["additionalProperties"]
        if isinstance(value, bool):
            lines.append(f"- Additional properties: {'allowed' if value else 'not allowed'}")
        elif isinstance(value, Mapping):
            lines.append(
                f"- Additional properties: {_schema_label(value, schema_anchors=schema_anchors)}"
            )
        else:
            _append_unsupported(lines, "additionalProperties", value)
    known = {
        "type", "format", "title", "description", "required", "properties",
        "minLength", "maxLength", "pattern", "minimum", "maximum", "exclusiveMinimum",
        "exclusiveMaximum", "multipleOf", "minItems", "maxItems", "uniqueItems", "minProperties",
        "maxProperties", "default", "enum", "const", "nullable", "readOnly", "writeOnly",
        "deprecated", "example", "examples", "allOf", "anyOf", "oneOf", "not", "discriminator",
        "additionalProperties", "items",
    }
    if _schema_is_rendered(schema) and isinstance(schema.get("$ref"), str):
        known.add("$ref")
    if _append_reference_siblings(
        lines,
        "schema reference siblings",
        schema,
        _root_schema_rendered_keys(schema),
    ):
        known.update(schema)
    unsupported = {key: value for key, value in schema.items() if key not in known}
    if unsupported:
        _append_unsupported(lines, "schema fields", unsupported)


def _append_property_table(
    lines: list[str], schema: Mapping[str, Any], schema_anchors: Mapping[str, str]
) -> None:
    if "properties" not in schema:
        return
    properties = schema["properties"]
    if not isinstance(properties, Mapping):
        _append_unsupported(lines, "properties", properties)
        return
    required = schema.get("required", [])
    required_names = set(required) if isinstance(required, list) and all(isinstance(item, str) for item in required) else set()
    lines.extend(["", "### Properties", "", "| Name | Required | Schema | Description | Constraints |", "| --- | --- | --- | --- | --- |"])
    for name, property_schema in sorted(properties.items(), key=lambda item: str(item[0]).lower()):
        description = property_schema.get("description", "") if isinstance(property_schema, Mapping) else ""
        constraints = _constraints_label(property_schema) if isinstance(property_schema, Mapping) else ""
        lines.append(
            "| "
            f"`{_escape_code(str(name))}` | {'yes' if name in required_names else 'no'} | "
            f"{_schema_label(property_schema, schema_anchors=schema_anchors)} | "
            f"{_escape_cell(description) if isinstance(description, str) else _escape_cell(_stable_json(description))} | "
            f"{_escape_cell(constraints)} |"
        )
        if isinstance(property_schema, Mapping):
            _append_property_metadata(lines, str(name), property_schema, schema_anchors)


def _append_property_metadata(
    lines: list[str],
    name: str,
    schema: Mapping[str, Any],
    schema_anchors: Mapping[str, str],
) -> None:
    metadata: list[str] = []
    represented = {
        "type", "format", "description", "minLength", "maxLength",
        "pattern", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "multipleOf", "minItems", "maxItems", "uniqueItems", "minProperties",
        "maxProperties",
    }
    if _schema_is_rendered(schema) and isinstance(schema.get("$ref"), str):
        represented.add("$ref")
        represented.discard("type")
        represented.discard("format")
    if "default" in schema:
        metadata.append(f"Default: `{_inline_json(schema['default'])}`")
        represented.add("default")
    if "enum" in schema and isinstance(schema["enum"], list):
        metadata.append(f"Enum: {', '.join(f'`{_enum_value(item)}`' for item in schema['enum'])}")
        represented.add("enum")
    if "const" in schema:
        metadata.append(f"Const: `{_inline_json(schema['const'])}`")
        represented.add("const")
    flags = [flag for flag in ("nullable", "readOnly", "writeOnly", "deprecated") if schema.get(flag) is True]
    if flags:
        metadata.append(f"Flags: {', '.join(f'`{flag}`' for flag in flags)}")
        represented.update(flags)
    if "example" in schema:
        metadata.append(f"Example: `{_inline_json(schema['example'])}`")
        represented.add("example")
    if "items" in schema:
        items = schema["items"]
        if isinstance(items, list):
            metadata.append(
                "Items: "
                + ", ".join(
                    _schema_label(item, schema_anchors=schema_anchors) for item in items
                )
            )
            represented.add("items")
        elif isinstance(items, Mapping):
            metadata.append(
                f"Items: {_schema_label(items, schema_anchors=schema_anchors)}"
            )
            represented.add("items")
    if metadata:
        lines.append(f"- `{_escape_code(name)}`: {'; '.join(metadata)}")
    if _append_reference_siblings(
        lines,
        f"property {_escape_code(name)} schema siblings",
        schema,
        represented,
    ):
        represented.update(schema)
    remaining = {key: value for key, value in schema.items() if key not in represented}
    if remaining:
        _append_unsupported(lines, f"property {_escape_code(name)} metadata", remaining)


def _constraints_label(schema: Mapping[str, Any]) -> str:
    constraints: list[str] = []
    for key in (
        "minLength", "maxLength", "pattern", "minimum", "maximum", "exclusiveMinimum",
        "exclusiveMaximum", "multipleOf", "minItems", "maxItems", "uniqueItems", "minProperties",
        "maxProperties",
    ):
        if key in schema:
            constraints.append(f"{key}={_inline_json(schema[key])}")
    return ", ".join(constraints)


def _append_composition(
    lines: list[str], schema: Mapping[str, Any], schema_anchors: Mapping[str, str]
) -> None:
    composition_keys = ("allOf", "anyOf", "oneOf", "not", "discriminator")
    if not any(key in schema for key in composition_keys):
        return
    lines.extend(["", "### Composition"])
    for key in composition_keys:
        if key not in schema:
            continue
        value = schema[key]
        if key in {"allOf", "anyOf", "oneOf"} and isinstance(value, list):
            lines.append(
                f"- `{key}`: {', '.join(_schema_label(item, schema_anchors=schema_anchors) for item in value)}"
            )
        elif key == "not" and isinstance(value, Mapping):
            lines.append(
                f"- `not`: {_schema_label(value, schema_anchors=schema_anchors)}"
            )
        else:
            _append_unsupported(lines, key, value)


def _enum_value(value: object) -> str:
    return value if isinstance(value, str) else _inline_json(value)


def _schema_anchor_registry(schemas: Mapping[str, Any]) -> dict[str, str]:
    names = sorted((str(name) for name in schemas), key=lambda name: (name.lower(), name))
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(_anchor(name), []).append(name)

    anchors: dict[str, str] = {}
    for base, group in groups.items():
        if len(group) == 1:
            anchors[group[0]] = base
            continue
        for name in group:
            digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
            anchors[name] = f"{base}--{digest}"
    return anchors


def _anchor(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "schema"


def _title_and_version(document: Mapping[str, Any]) -> tuple[str, str]:
    info = _require_mapping(document, "info")
    title = info.get("title")
    version = info.get("version")
    if not isinstance(title, str) or not isinstance(version, str):
        raise ValueError("OpenAPI info title and version must be strings")
    return title, version


def _finish(lines: list[str]) -> str:
    return "\n".join(lines).rstrip() + "\n"


def _snapshot_app_modules() -> dict[str, object]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "app" or name.startswith("app.")
    }


def _snapshot_prometheus_collectors() -> tuple[object, tuple[object, ...]] | None:
    try:
        from prometheus_client import REGISTRY
    except ImportError:
        return None
    collectors = getattr(REGISTRY, "_names_to_collectors", {})
    if not isinstance(collectors, Mapping):
        return None
    return REGISTRY, tuple(collectors.values())


def _remove_transient_prometheus_collectors(
    snapshot: tuple[object, tuple[object, ...]] | None,
) -> None:
    if snapshot is None:
        return
    registry, existing_collectors = snapshot
    collectors = getattr(registry, "_names_to_collectors", {})
    if not isinstance(collectors, Mapping):
        return
    existing_ids = {id(collector) for collector in existing_collectors}
    transient_collectors = {
        id(collector): collector
        for collector in collectors.values()
        if id(collector) not in existing_ids
    }
    for collector in transient_collectors.values():
        try:
            registry.unregister(collector)
        except KeyError:
            continue


def _remove_transient_app_modules(existing_modules: Mapping[str, object]) -> None:
    transient_modules = {
        name: module
        for name, module in _snapshot_app_modules().items()
        if name not in existing_modules
    }
    for name in sorted(transient_modules, key=lambda value: value.count("."), reverse=True):
        module = transient_modules[name]
        if sys.modules.get(name) is not module:
            continue
        sys.modules.pop(name, None)
        parent_name, _, attribute = name.rpartition(".")
        if not parent_name:
            continue
        parent = sys.modules.get(parent_name)
        if getattr(parent, attribute, None) is module:
            delattr(parent, attribute)


def build_contract_document() -> dict[str, Any]:
    """Build a detached OpenAPI document with fixed documentation settings."""

    existing_app_modules = _snapshot_app_modules()
    prometheus_collectors = _snapshot_prometheus_collectors()
    previous = {key: os.environ.get(key) for key in DOCUMENTATION_ENV}
    os.environ.update(DOCUMENTATION_ENV)
    try:
        from app.config import get_settings
        from app.main import create_app

        get_settings.cache_clear()
        document = json.loads(json.dumps(create_app().openapi(), ensure_ascii=False))
        if not isinstance(document, dict):
            raise ValueError("OpenAPI document must be an object")
        paths = _require_mapping(document, "paths")
        components = _require_mapping(document, "components")
        _require_mapping(components, "schemas")
        if not isinstance(paths, Mapping):
            raise AssertionError("validated OpenAPI paths must be a mapping")
        return document
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        from app.config import get_settings

        get_settings.cache_clear()
        _remove_transient_app_modules(existing_app_modules)
        _remove_transient_prometheus_collectors(prometheus_collectors)


def _render_artifacts() -> dict[Path, str]:
    document = build_contract_document()
    rendered = render_openapi_markdown(
        document,
        source_path="contracts/openapi.json",
        update_command=UPDATE_COMMAND,
    )
    return {
        CONTRACT_PATH: render_contract_json(document),
        ENDPOINTS_PATH: rendered.endpoints,
        SCHEMAS_PATH: rendered.schemas,
    }


def _write_atomic(path: Path, content: str) -> None:
    """Atomically replace a UTF-8 artifact after writing one complete document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    """Update generated API artifacts, or check them without writing."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    artifacts = _render_artifacts()

    if args.check:
        stale_paths = [
            path
            for path, expected in artifacts.items()
            if not path.exists() or path.read_text(encoding="utf-8") != expected
        ]
        if stale_paths:
            for path in stale_paths:
                print(
                    "AI OpenAPI artifact is missing or out of date: "
                    f"{_display_path(path)}"
                )
            print(f"Run {UPDATE_COMMAND}.")
            return 1
        print("AI OpenAPI artifacts are current.")
        return 0

    for path, content in artifacts.items():
        _write_atomic(path, content)
        print(f"Wrote {_display_path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
