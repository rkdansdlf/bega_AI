from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence


def build_search_result(**fields: Any) -> dict[str, Any]:
    return {
        "found": False,
        "error": None,
        **fields,
    }


def apply_list_results(
    result: dict[str, Any],
    *,
    field: str,
    rows: Sequence[Any],
    total_field: str | None = None,
    row_mapper: Callable[[Any], Any] | None = None,
    extra_updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    mapper = row_mapper or (lambda row: row)
    items = [mapper(row) for row in rows]
    result[field] = items
    result["found"] = bool(items)
    if total_field is not None:
        result[total_field] = len(items)
    if extra_updates:
        result.update(dict(extra_updates))
    return result


def apply_error(result: dict[str, Any], message: str) -> dict[str, Any]:
    result["error"] = message
    return result
