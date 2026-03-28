"""Shared immutable or effectively immutable AI-service resources."""

from __future__ import annotations

from functools import lru_cache

from ..tools.latest_baseball import LatestBaseballSearchTool


@lru_cache(maxsize=1)
def get_shared_latest_baseball_tool() -> LatestBaseballSearchTool:
    """Return a process-wide LatestBaseballSearchTool instance."""
    return LatestBaseballSearchTool()
