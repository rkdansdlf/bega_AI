from __future__ import annotations

import logging
from typing import Any, Callable

from app.tools.team_code_resolver import TeamCodeResolver

logger = logging.getLogger(__name__)

_DEFAULT_TEAM_RESOLVER = TeamCodeResolver()


def resolve_team_display_name(
    team_value: Any,
    team_name_resolver: Callable[[str], str] | None = None,
) -> Any:
    if not isinstance(team_value, str) or not team_value:
        return team_value

    if team_name_resolver is not None:
        try:
            resolved_name = team_name_resolver(team_value)
        except Exception as exc:
            logger.debug(
                "[TeamDisplay] Resolver failed for %s: %s",
                team_value,
                exc,
            )
        else:
            if isinstance(resolved_name, str) and resolved_name.strip() and resolved_name != team_value:
                return resolved_name

    fallback_name = _DEFAULT_TEAM_RESOLVER.display_name(team_value)
    if isinstance(fallback_name, str) and fallback_name.strip():
        return fallback_name
    return team_value


def replace_team_codes(
    data: Any,
    team_name_resolver: Callable[[str], str] | None = None,
) -> Any:
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if "team" in key and isinstance(value, str):
                new_dict[key] = resolve_team_display_name(value, team_name_resolver)
            else:
                new_dict[key] = replace_team_codes(value, team_name_resolver)
        return new_dict

    if isinstance(data, list):
        return [replace_team_codes(item, team_name_resolver) for item in data]

    if isinstance(data, str):
        return resolve_team_display_name(data, team_name_resolver)

    return data
