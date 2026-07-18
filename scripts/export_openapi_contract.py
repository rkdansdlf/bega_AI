"""Export complete FastAPI OpenAPI JSON and deterministic Markdown docs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts" / "openapi.json"
ENDPOINTS_PATH = ROOT / "docs" / "api-endpoints.md"
SCHEMAS_PATH = ROOT / "docs" / "api-schemas.md"
UPDATE_COMMAND = "python scripts/export_openapi_contract.py"
HTTP_METHODS = ("get", "post", "put", "patch", "delete", "options", "head", "trace")


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
    operations = _collect_operations(paths)
    endpoints = _render_endpoints(
        document, source_path, update_command, paths, operations
    )
    schema_docs = _render_schemas(document, source_path, update_command, schemas)
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

    return _finish(lines)


def _security_label(definition: Mapping[str, Any]) -> str:
    security = definition.get("security")
    if security is None or security == []:
        return "Not specified in OpenAPI"
    if not isinstance(security, list):
        raise ValueError("OpenAPI operation security must be an array")

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


def _render_schemas(
    document: Mapping[str, Any],
    source_path: str,
    update_command: str,
    schemas: Mapping[str, Any],
) -> str:
    title, version = _title_and_version(document)
    return _finish(
        [
            f"# {title} Schemas",
            "",
            "> This file is generated. Do not edit directly.",
            f"> Source: `{source_path}`",
            f"> Regenerate with: `{update_command}`",
            "",
            f"Version: `{version}`",
            f"Schemas: **{len(schemas)}**",
        ]
    )


def _title_and_version(document: Mapping[str, Any]) -> tuple[str, str]:
    info = _require_mapping(document, "info")
    title = info.get("title")
    version = info.get("version")
    if not isinstance(title, str) or not isinstance(version, str):
        raise ValueError("OpenAPI info title and version must be strings")
    return title, version


def _finish(lines: list[str]) -> str:
    return "\n".join(lines).rstrip() + "\n"
