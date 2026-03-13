"""Thread-safe last-good snapshot storage for team mapping rows."""

from __future__ import annotations

from threading import RLock
from typing import Any, Dict, List, Optional

_snapshot_lock = RLock()
_last_good_team_mapping_rows: Optional[List[Dict[str, Any]]] = None


def update_team_mapping_snapshot(rows: List[Dict[str, Any]]) -> None:
    """Store a copy of the latest successful OCI mapping rows."""
    copied_rows = [dict(row) for row in rows]
    with _snapshot_lock:
        global _last_good_team_mapping_rows
        _last_good_team_mapping_rows = copied_rows


def load_team_mapping_snapshot() -> Optional[List[Dict[str, Any]]]:
    """Return a copy of the last successful OCI mapping rows, if any."""
    with _snapshot_lock:
        if _last_good_team_mapping_rows is None:
            return None
        return [dict(row) for row in _last_good_team_mapping_rows]
