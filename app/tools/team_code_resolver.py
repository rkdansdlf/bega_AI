from __future__ import annotations

from collections import defaultdict
import logging
import os
from typing import Any, Dict, Iterable, List
from app.tools.team_resolution_metrics import get_team_resolution_metrics

logger = logging.getLogger(__name__)


CANONICAL_CODES = {
    "SS",
    "LT",
    "LG",
    "DB",
    "KIA",
    "KH",
    "HH",
    "SSG",
    "NC",
    "KT",
}

LEGACY_TO_CANONICAL = {
    "HT": "KIA",
    "DO": "DB",
    "OB": "DB",
    "KI": "KH",
    "NX": "KH",
    "WO": "KH",
    "KW": "KH",
    "SK": "SSG",
    "SL": "SSG",
    "BE": "HH",
    "MBC": "LG",
    "LOT": "LT",
}


class TeamCodeResolver:
    """Canonical + legacy dual-read team resolver."""

    def __init__(self) -> None:
        self.team_resolution_metrics = get_team_resolution_metrics()
        self.read_mode = (
            os.getenv("TEAM_CODE_READ_MODE", "canonical_only").strip().lower()
        )
        self.canonical_window_start = self._read_int_env(
            "TEAM_CODE_CANONICAL_WINDOW_START", 2021
        )
        self.canonical_window_end = self._read_int_env(
            "TEAM_CODE_CANONICAL_WINDOW_END", 2025
        )
        self.outside_window_mode = (
            os.getenv("TEAM_CODE_OUTSIDE_WINDOW_MODE", "dual").strip().lower()
        )
        if self.read_mode not in {"dual", "canonical_only"}:
            self.read_mode = "canonical_only"
        if self.outside_window_mode not in {"dual", "canonical_only"}:
            self.outside_window_mode = "dual"
        self.name_to_canonical: Dict[str, str] = {
            "KIA": "KIA",
            "기아": "KIA",
            "KIA 타이거즈": "KIA",
            "HT": "KIA",
            "해태": "KIA",
            "LG": "LG",
            "LG 트윈스": "LG",
            "MBC": "LG",
            "SSG": "SSG",
            "SK": "SSG",
            "SSG 랜더스": "SSG",
            "NC": "NC",
            "NC 다이노스": "NC",
            "두산": "DB",
            "두산 베어스": "DB",
            "DB": "DB",
            "DO": "DB",
            "OB": "DB",
            "KT": "KT",
            "KT 위즈": "KT",
            "롯데": "LT",
            "롯데 자이언츠": "LT",
            "LT": "LT",
            "삼성": "SS",
            "삼성 라이온즈": "SS",
            "SS": "SS",
            "한화": "HH",
            "한화 이글스": "HH",
            "HH": "HH",
            "키움": "KH",
            "키움 히어로즈": "KH",
            "넥센": "KH",
            "KH": "KH",
            "KI": "KH",
            "WO": "KH",
            "NX": "KH",
            "우리": "KH",
            "빙그레": "HH",
            "현대": "HU",  # Historical dissolved
            "태평양": "TP",
            "청보": "CB",
            "삼미": "SM",
            "쌍방울": "SL",
        }

        self.team_variants: Dict[str, List[str]] = {
            "KIA": ["KIA", "HT"],
            "SSG": ["SSG", "SK"],
            "DB": ["DB", "OB", "DO"],
            "KH": ["KH", "WO", "NX", "KI", "KW"],
            "HH": ["HH", "BE"],
            "LG": ["LG", "MBC"],
            "LT": ["LT", "LOT"],
            "SS": ["SS"],
            "NC": ["NC"],
            "KT": ["KT"],
        }

        # Franchise brand history for season-aware resolution
        # Format: { canonical_code: [(end_year, brand_code), ...] } sorted by end_year
        self.brand_history: Dict[str, List[tuple[int | None, str]]] = {
            "KIA": [(2000, "HT"), (None, "KIA")],
            "SSG": [(2020, "SK"), (None, "SSG")],
            "DB": [(1998, "OB"), (None, "DB")],
            "KH": [(2009, "WO"), (2018, "NX"), (None, "KH")],
            "HH": [(1993, "BE"), (None, "HH")],
            "LG": [(1989, "MBC"), (None, "LG")],
        }

        self.code_to_name: Dict[str, str] = {
            "KIA": "KIA 타이거즈",
            "HT": "해태 타이거즈",
            "LG": "LG 트윈스",
            "MBC": "MBC 청룡",
            "SSG": "SSG 랜더스",
            "SK": "SK 와이번스",
            "NC": "NC 다이노스",
            "DB": "두산 베어스",
            "OB": "OB 베어스",
            "KT": "KT 위즈",
            "LT": "롯데 자이언츠",
            "SS": "삼성 라이온즈",
            "HH": "한화 이글스",
            "BE": "빙그레 이글스",
            "KH": "키움 히어로즈",
            "NX": "넥센 히어로즈",
            "WO": "우리 히어로즈",
        }

    @staticmethod
    def _clean(team_input: str) -> str:
        return team_input.strip().upper()

    @staticmethod
    def _read_int_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return int(raw.strip())
        except (TypeError, ValueError):
            return default

    def is_in_canonical_window(self, season_year: int | None) -> bool:
        if season_year is None:
            return False
        start = min(self.canonical_window_start, self.canonical_window_end)
        end = max(self.canonical_window_start, self.canonical_window_end)
        return start <= season_year <= end

    def _query_mode_for_year(self, season_year: int | None) -> str:
        if self.read_mode != "canonical_only":
            return "dual"
        if season_year is None:
            return "dual"
        if self.is_in_canonical_window(season_year):
            return "canonical_only"
        return self.outside_window_mode

    def resolve_canonical(self, team_input: str, season_year: int | None = None) -> str:
        if not team_input:
            return team_input
        cleaned = self._clean(team_input)

        # 1. Start with name/initial code mapping to canonical
        canonical = self.name_to_canonical.get(
            team_input
        ) or self.name_to_canonical.get(cleaned)
        if not canonical:
            canonical = LEGACY_TO_CANONICAL.get(cleaned, cleaned)

        # 2. If year is provided, find the EXACT brand code for that year
        if season_year and canonical in self.brand_history:
            for end_year, brand in self.brand_history[canonical]:
                if end_year is None or season_year <= end_year:
                    return brand

        return canonical

    def query_variants(
        self, team_input: str, season_year: int | None = None
    ) -> List[str]:
        canonical = self.resolve_canonical(team_input)
        if not canonical:
            return []

        query_mode = self._query_mode_for_year(season_year)
        outside_window = (
            self.read_mode == "canonical_only"
            and season_year is not None
            and not self.is_in_canonical_window(season_year)
        )
        fallback_used = outside_window and query_mode == "dual"
        self.team_resolution_metrics.record_resolution_event(
            season_year=season_year,
            query_mode=query_mode,
            outside_window=outside_window,
            fallback_used=fallback_used,
        )
        self.team_resolution_metrics.maybe_log(
            logger, "TeamCodeResolver.query_variants"
        )

        if query_mode == "canonical_only":
            target = self.resolve_canonical(team_input, season_year)
            return [target] if target else []

        ordered_variants: List[str] = []
        if season_year is not None:
            seasonal_code = self.resolve_canonical(team_input, season_year)
            if seasonal_code:
                ordered_variants.append(seasonal_code)

        if canonical in self.team_variants:
            ordered_variants.extend(self.team_variants[canonical])
        else:
            ordered_variants.append(canonical)

        deduped: List[str] = []
        for code in ordered_variants:
            if code and code not in deduped:
                deduped.append(code)
        return deduped

    def variants(self, team_input: str, season_year: int | None = None) -> List[str]:
        return self.query_variants(team_input, season_year)

    def display_name(self, team_code: str) -> str:
        if not team_code:
            return team_code
        cleaned = self._clean(team_code)
        return self.code_to_name.get(cleaned, team_code)

    def sync_from_team_rows(self, rows: Iterable[Dict[str, Any]]) -> None:
        """Sync resolver mappings from teams + team_franchises rows."""
        by_franchise: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_franchise[row["franchise_id"]].append(row)

        for members in by_franchise.values():
            if not members:
                continue

            preferred = members[0]
            current_code = preferred.get("current_code")
            canonical = self.resolve_canonical(current_code or preferred["team_id"])
            if canonical not in CANONICAL_CODES:
                canonical = self.resolve_canonical(preferred["team_id"])

            modern_name = preferred["team_name"]
            variant_codes: List[str] = [canonical]
            for member in members:
                team_id = str(member["team_id"]).upper()
                if team_id not in variant_codes:
                    variant_codes.append(team_id)

            for legacy in self.team_variants.get(canonical, []):
                if legacy not in variant_codes:
                    variant_codes.append(legacy)
            self.team_variants[canonical] = variant_codes

            for member in members:
                team_id = str(member["team_id"]).upper()
                team_name = member["team_name"]
                self.name_to_canonical[team_id] = canonical
                self.name_to_canonical[team_name] = canonical
                short_name = team_name.split()[0]
                self.name_to_canonical[short_name] = canonical
                # Don't overwrite display name if we already have a specific one (like HT/SK)
                if team_id not in self.code_to_name:
                    self.code_to_name[team_id] = modern_name

            self.code_to_name[canonical] = modern_name

    @property
    def query_mode(self) -> str:
        return "canonical_only" if self.read_mode == "canonical_only" else "dual"
