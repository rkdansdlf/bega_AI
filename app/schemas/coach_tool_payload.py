"""Coach 컨텍스트 전용 도구 호출 결과 페이로드.

기존 ``DatabaseQueryTool``/``GameQueryTool``의 도구 함수들은 광범위 dict를
반환하여 Coach Fact Sheet 빌더 단계에서 미사용 필드까지 흘러간다. 이 모듈은
Coach가 실제로 사용하는 핵심 필드만 담는 dataclass를 정의하고, 기존 빌더가
기대하는 dict 구조와 호환되도록 ``to_factsheet_dict()`` 어댑터 메서드를
제공한다.

설계 원칙:
- ``slots=True, frozen=True``: 메모리 절약 + 불변
- top_batters/top_pitchers는 도구 단계에서 이미 ``top_n`` (기본 3)으로 절단
- ``to_factsheet_dict()``가 반환하는 dict는 ``_append_team_fact_lines`` 등
  기존 코드가 읽는 키와 동일하므로 호출자 변경을 점진적으로 진행 가능
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True, frozen=True)
class CoachPlayerLine:
    """Coach 컨텍스트용 핵심 선수 라인."""

    player_name: str
    role: Optional[str] = None
    # 타자 지표
    ops: Optional[float] = None
    avg: Optional[float] = None
    obp: Optional[float] = None
    slg: Optional[float] = None
    home_runs: Optional[int] = None
    rbi: Optional[int] = None
    plate_appearances: Optional[int] = None
    # 투수 지표
    era: Optional[float] = None
    whip: Optional[float] = None
    wins: Optional[int] = None
    losses: Optional[int] = None
    saves: Optional[int] = None
    holds: Optional[int] = None
    innings_pitched: Optional[float] = None
    strikeouts: Optional[int] = None
    games_started: Optional[int] = None
    # 부가
    team_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """기존 dict 스키마와 호환되는 직렬화."""
        result: Dict[str, Any] = {"player_name": self.player_name}
        for key in (
            "role",
            "ops",
            "avg",
            "obp",
            "slg",
            "home_runs",
            "rbi",
            "plate_appearances",
            "era",
            "whip",
            "wins",
            "losses",
            "saves",
            "holds",
            "innings_pitched",
            "strikeouts",
            "games_started",
            "team_name",
        ):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


@dataclass(slots=True, frozen=True)
class CoachFormSignal:
    """선수 폼 시그널 (배터 또는 투수)."""

    player_name: str
    form_status: Optional[str] = None  # 'hot' | 'steady' | 'cold' | 'insufficient'
    form_score: Optional[float] = None
    season_metrics: Dict[str, Any] = field(default_factory=dict)
    recent_metrics: Dict[str, Any] = field(default_factory=dict)
    clutch_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_name": self.player_name,
            "form_status": self.form_status,
            "form_score": self.form_score,
            "season_metrics": dict(self.season_metrics),
            "recent_metrics": dict(self.recent_metrics),
            "clutch_metrics": dict(self.clutch_metrics),
        }


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _player_dict_to_line(item: Dict[str, Any]) -> CoachPlayerLine:
    """광범위 player dict → ``CoachPlayerLine`` 변환."""
    return CoachPlayerLine(
        player_name=str(item.get("player_name") or ""),
        role=item.get("role"),
        ops=_coerce_float(item.get("ops")),
        avg=_coerce_float(item.get("avg")),
        obp=_coerce_float(item.get("obp")),
        slg=_coerce_float(item.get("slg")),
        home_runs=_coerce_int(item.get("home_runs")),
        rbi=_coerce_int(item.get("rbi")),
        plate_appearances=_coerce_int(item.get("plate_appearances")),
        era=_coerce_float(item.get("era")),
        whip=_coerce_float(item.get("whip")),
        wins=_coerce_int(item.get("wins")),
        losses=_coerce_int(item.get("losses")),
        saves=_coerce_int(item.get("saves")),
        holds=_coerce_int(item.get("holds")),
        innings_pitched=_coerce_float(item.get("innings_pitched")),
        strikeouts=_coerce_int(item.get("strikeouts")),
        games_started=_coerce_int(item.get("games_started")),
        team_name=item.get("team_name"),
    )


def _form_dict_to_signal(item: Dict[str, Any]) -> CoachFormSignal:
    return CoachFormSignal(
        player_name=str(item.get("player_name") or ""),
        form_status=item.get("form_status"),
        form_score=_coerce_float(item.get("form_score")),
        season_metrics=dict(item.get("season_metrics") or {}),
        recent_metrics=dict(item.get("recent_metrics") or {}),
        clutch_metrics=dict(item.get("clutch_metrics") or {}),
    )


@dataclass(slots=True, frozen=True)
class CoachTeamPayload:
    """Coach 분석에 필요한 한 팀의 모든 핵심 정보를 담는 페이로드.

    ``to_factsheet_dict()``가 반환하는 dict는 ``_append_team_fact_lines``,
    ``_recent_summary``, ``_advanced_metrics``, ``_player_form_signals``가
    기대하는 동일한 키 경로를 갖는다.
    """

    team_id: str
    team_name: str
    found: bool = True
    error: Optional[str] = None
    # 선수 라인 (이미 top_n으로 절단됨)
    top_batters: List[CoachPlayerLine] = field(default_factory=list)
    top_pitchers: List[CoachPlayerLine] = field(default_factory=list)
    # 최근 폼 (recent.summary에 들어가는 핵심 4개 키)
    recent_wins: int = 0
    recent_losses: int = 0
    recent_draws: int = 0
    recent_run_diff: int = 0
    # advanced metrics 중 fact sheet가 사용하는 부분만
    batting_ops: Optional[float] = None
    bullpen_share: Optional[Any] = None  # 원본 형식 보존 (str 또는 number)
    # 추가 advanced metrics (옵션 - 일부 컴포저가 사용)
    advanced_extra: Dict[str, Any] = field(default_factory=dict)
    # 폼 시그널 (top-1 batter, top-1 pitcher 권장)
    form_signals_batters: List[CoachFormSignal] = field(default_factory=list)
    form_signals_pitchers: List[CoachFormSignal] = field(default_factory=list)

    @classmethod
    def from_team_data_dict(
        cls,
        team_data: Dict[str, Any],
        *,
        team_id: str,
        team_name_fallback: str = "",
        top_n: int = 3,
        form_signals_top_n: int = 1,
    ) -> "CoachTeamPayload":
        """``_execute_coach_tools_parallel`` 결과의 한 팀 dict → 압축 페이로드.

        ``team_data`` 형식:
            {
                "summary": {"top_batters": [...], "top_pitchers": [...], ...},
                "advanced": {"metrics": {"batting": {...}, "pitching": {...}},
                             "fatigue_index": {"bullpen_share": ...},
                             "rankings": {...}},
                "recent": {"summary": {"wins": int, "losses": int,
                                        "draws": int, "run_diff": int}},
                "player_form_signals": {"batters": [...], "pitchers": [...]},
            }
        """
        summary_raw: Dict[str, Any] = team_data.get("summary") or {}
        advanced_raw: Dict[str, Any] = team_data.get("advanced") or {}
        recent_raw: Dict[str, Any] = team_data.get("recent") or {}
        form_raw: Dict[str, Any] = team_data.get("player_form_signals") or {}

        team_name = str(
            summary_raw.get("team_name")
            or recent_raw.get("team_name")
            or team_name_fallback
            or team_id
        )

        top_batters = [
            _player_dict_to_line(item)
            for item in (summary_raw.get("top_batters") or [])
            if isinstance(item, dict)
        ][:top_n]
        top_pitchers = [
            _player_dict_to_line(item)
            for item in (summary_raw.get("top_pitchers") or [])
            if isinstance(item, dict)
        ][:top_n]

        def _sorted_form(items: Any) -> List[CoachFormSignal]:
            valid = [x for x in (items or []) if isinstance(x, dict)]
            valid.sort(key=lambda x: float(x.get("form_score") or -1.0), reverse=True)
            return [_form_dict_to_signal(x) for x in valid[:form_signals_top_n]]

        recent_summary = recent_raw.get("summary") or {}
        if not isinstance(recent_summary, dict):
            recent_summary = {}

        adv_metrics = advanced_raw.get("metrics") or {}
        batting_metrics = adv_metrics.get("batting") or {}
        fatigue_index = advanced_raw.get("fatigue_index") or {}

        advanced_extra: Dict[str, Any] = {}
        if "rankings" in advanced_raw:
            advanced_extra["rankings"] = advanced_raw["rankings"]
        pitching_metrics = adv_metrics.get("pitching")
        if pitching_metrics:
            advanced_extra.setdefault("metrics", {})["pitching"] = pitching_metrics

        found = bool(summary_raw.get("found")) or bool(recent_raw.get("found"))
        error = (
            summary_raw.get("error")
            or recent_raw.get("error")
            or advanced_raw.get("error")
            or team_data.get("error")
        )

        return cls(
            team_id=team_id,
            team_name=team_name,
            found=found,
            error=error,
            top_batters=top_batters,
            top_pitchers=top_pitchers,
            recent_wins=_coerce_int(recent_summary.get("wins")) or 0,
            recent_losses=_coerce_int(recent_summary.get("losses")) or 0,
            recent_draws=_coerce_int(recent_summary.get("draws")) or 0,
            recent_run_diff=_coerce_int(recent_summary.get("run_diff")) or 0,
            batting_ops=_coerce_float(batting_metrics.get("ops")),
            bullpen_share=fatigue_index.get("bullpen_share"),
            advanced_extra=advanced_extra,
            form_signals_batters=_sorted_form(form_raw.get("batters")),
            form_signals_pitchers=_sorted_form(form_raw.get("pitchers")),
        )

    def to_factsheet_dict(self) -> Dict[str, Any]:
        """``_build_coach_fact_sheet`` 경로가 읽는 dict 구조로 직렬화.

        키 경로: ``team_data["summary"]["top_batters"]``,
        ``team_data["recent"]["summary"]``, ``team_data["advanced"]["metrics"]
        ["batting"]["ops"]``, ``team_data["advanced"]["fatigue_index"]
        ["bullpen_share"]``, ``team_data["player_form_signals"]["batters"]``
        등.
        """
        advanced: Dict[str, Any] = {
            "metrics": {"batting": {"ops": self.batting_ops}},
            "fatigue_index": {"bullpen_share": self.bullpen_share},
        }
        if self.advanced_extra:
            for key, value in self.advanced_extra.items():
                if key in advanced:
                    if isinstance(advanced[key], dict) and isinstance(value, dict):
                        merged = dict(value)
                        merged.update(advanced[key])
                        advanced[key] = merged
                    continue
                advanced[key] = value

        return {
            "team_name": self.team_name,
            "team_id": self.team_id,
            "found": self.found,
            "error": self.error,
            "summary": {
                "team_name": self.team_name,
                "year": None,
                "top_batters": [p.to_dict() for p in self.top_batters],
                "top_pitchers": [p.to_dict() for p in self.top_pitchers],
                "found": self.found,
                "error": self.error,
            },
            "recent": {
                "summary": {
                    "wins": self.recent_wins,
                    "losses": self.recent_losses,
                    "draws": self.recent_draws,
                    "run_diff": self.recent_run_diff,
                }
            },
            "advanced": advanced,
            "player_form_signals": {
                "batters": [s.to_dict() for s in self.form_signals_batters],
                "pitchers": [s.to_dict() for s in self.form_signals_pitchers],
            },
        }


@dataclass(slots=True, frozen=True)
class ClutchMomentLine:
    """승부처 한 장면."""

    inning_label: str = "이닝 미상"
    outs: int = 0
    bases_before: str = "-"
    batter_name: str = "타자 미상"
    wpa_delta_pct: Any = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inning_label": self.inning_label,
            "outs": self.outs,
            "bases_before": self.bases_before,
            "batter_name": self.batter_name,
            "wpa_delta_pct": self.wpa_delta_pct,
            "description": self.description,
        }


@dataclass(slots=True, frozen=True)
class CoachMatchupPayload:
    """양 팀 매치업 정보."""

    home_team_id: str
    away_team_id: str
    head_to_head_summary: Optional[str] = None
    head_to_head_recent: List[Dict[str, Any]] = field(default_factory=list)
    clutch_moments: List[ClutchMomentLine] = field(default_factory=list)

    def to_clutch_dict(self) -> Dict[str, Any]:
        """``_clutch_moments`` 헬퍼가 읽는 dict 형식.

        키 경로: ``tool_results["clutch_moments"]["moments"]``.
        """
        return {"moments": [m.to_dict() for m in self.clutch_moments]}

    def to_head_to_head_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.head_to_head_summary,
            "recent": list(self.head_to_head_recent),
        }


__all__ = [
    "CoachPlayerLine",
    "CoachFormSignal",
    "CoachTeamPayload",
    "ClutchMomentLine",
    "CoachMatchupPayload",
]
