from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Dict, Optional


class TeamResolutionMetrics:
    """Thread-safe in-memory counters for team resolution monitoring."""

    def __init__(self, emit_every: int = 100) -> None:
        self._lock = threading.RLock()
        self._emit_every = max(1, emit_every)

        self._event_total = 0
        self._last_emitted_event = 0

        self._resolution_total = 0
        self._outside_window_total = 0
        self._outside_window_fallback_total = 0
        self._mode_counts: Dict[str, int] = defaultdict(int)
        self._year_bucket_counts: Dict[str, int] = defaultdict(int)

        self._query_total = 0
        self._query_miss_total = 0
        self._source_totals: Dict[str, int] = defaultdict(int)
        self._source_misses: Dict[str, int] = defaultdict(int)

    @staticmethod
    def _year_bucket(season_year: Optional[int]) -> str:
        if season_year is None:
            return "unknown"
        if 2021 <= season_year <= 2025:
            return "window_2021_2025"
        return "outside_window"

    def record_resolution_event(
        self,
        *,
        season_year: Optional[int],
        query_mode: str,
        outside_window: bool,
        fallback_used: bool,
    ) -> None:
        with self._lock:
            self._event_total += 1
            self._resolution_total += 1
            self._mode_counts[query_mode] += 1
            self._year_bucket_counts[self._year_bucket(season_year)] += 1

            if outside_window:
                self._outside_window_total += 1
            if fallback_used:
                self._outside_window_fallback_total += 1

    def record_query_result(
        self,
        *,
        source: str,
        season_year: Optional[int],
        found: bool,
        error: Optional[str],
    ) -> None:
        with self._lock:
            self._event_total += 1
            self._query_total += 1
            self._source_totals[source] += 1

            miss = (not found) or bool(error)
            if miss:
                self._query_miss_total += 1
                self._source_misses[source] += 1

            self._year_bucket_counts[self._year_bucket(season_year)] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            miss_rate = (
                self._query_miss_total / self._query_total
                if self._query_total > 0
                else 0.0
            )
            fallback_rate = (
                self._outside_window_fallback_total / self._outside_window_total
                if self._outside_window_total > 0
                else 0.0
            )

            return {
                "resolution_total": self._resolution_total,
                "outside_window_total": self._outside_window_total,
                "outside_window_fallback_total": self._outside_window_fallback_total,
                "outside_window_fallback_rate": fallback_rate,
                "query_total": self._query_total,
                "query_miss_total": self._query_miss_total,
                "team_resolution_miss_rate": miss_rate,
                "mode_counts": dict(self._mode_counts),
                "year_bucket_counts": dict(self._year_bucket_counts),
                "source_totals": dict(self._source_totals),
                "source_misses": dict(self._source_misses),
            }

    def maybe_log(self, logger: Any, source: str) -> None:
        with self._lock:
            if (self._event_total - self._last_emitted_event) < self._emit_every:
                return
            self._last_emitted_event = self._event_total
            snapshot = self.snapshot()

        logger.info(
            "[TeamResolutionMetrics] source=%s resolution_total=%s query_total=%s "
            "team_resolution_miss_rate=%.6f outside_window_fallback_rate=%.6f "
            "outside_window_total=%s outside_window_fallback_total=%s",
            source,
            snapshot["resolution_total"],
            snapshot["query_total"],
            snapshot["team_resolution_miss_rate"],
            snapshot["outside_window_fallback_rate"],
            snapshot["outside_window_total"],
            snapshot["outside_window_fallback_total"],
        )

    def reset(self) -> None:
        with self._lock:
            self._event_total = 0
            self._last_emitted_event = 0
            self._resolution_total = 0
            self._outside_window_total = 0
            self._outside_window_fallback_total = 0
            self._mode_counts.clear()
            self._year_bucket_counts.clear()
            self._query_total = 0
            self._query_miss_total = 0
            self._source_totals.clear()
            self._source_misses.clear()


_TEAM_RESOLUTION_METRICS = TeamResolutionMetrics()


def get_team_resolution_metrics() -> TeamResolutionMetrics:
    return _TEAM_RESOLUTION_METRICS
