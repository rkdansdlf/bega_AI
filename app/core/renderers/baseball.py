"""KBO 시즌 데이터를 위한 야구 전용 청크 렌더"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

NUMBER_SENTINELS = {"", None, "null", "None"}


def _has_final_consonant(word: str) -> bool:
    if not word:
        return False
    char = word[-1]
    code = ord(char)
    if not 0xAC00 <= code <= 0xD7A3:
        return False
    return (code - 0xAC00) % 28 != 0


def josa(word: str, pair: Tuple[str, str] = ("은", "는")) -> str:
    """Return word + correct particle."""

    return f"{word}{pair[0] if _has_final_consonant(word) else pair[1]}"


def _format_float(value: Any, digits: int) -> Optional[str]:
    if value in NUMBER_SENTINELS:
        return None
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return None


def _format_ip(value: Any) -> Optional[str]:
    if value in NUMBER_SENTINELS:
        return None
    try:
        ip = float(value)
    except (TypeError, ValueError):
        return None
    whole = int(ip)
    fraction = round((ip - whole) * 10)
    if fraction == 0:
        return f"{whole}이닝"
    return f"{whole}.{fraction}이닝"


def _format_percentage(value: Any, digits: int = 1) -> Optional[str]:
    if value in NUMBER_SENTINELS:
        return None
    try:
        pct = float(value) * 100
    except (TypeError, ValueError):
        return None
    return f"{pct:.{digits}f}%"


def _format_count(value: Any, suffix: str = "") -> Optional[str]:
    if value in NUMBER_SENTINELS:
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return f"{number}{suffix}"


def make_meta(row: Dict[str, Any], *, kind: str, aliases: Iterable[str]) -> str:
    meta = {
        "kind": kind,
        "season": row.get("season_year") or row.get("season"),
        "team": row.get("team_name") or row.get("team_code"),
        "player_id": row.get("player_id"),
        "aliases": [alias for alias in aliases if alias],
        "primary_stats": {},
    }
    for key in ("era", "avg", "ops", "whip", "ip"):
        if row.get(key) not in NUMBER_SENTINELS:
            meta["primary_stats"][key] = row[key]
    return json.dumps(meta, ensure_ascii=False)


def _today_fallback(today_str: Optional[str]) -> str:
    if today_str:
        return today_str
    return datetime.utcnow().strftime("%Y-%m-%d")


def _build_aliases(row: Dict[str, Any]) -> List[str]:
    aliases = [
        row.get("player_name_eng"),
        row.get("player_name"),
        row.get("team_name"),
        row.get("team_code"),
        row.get("player_id"),
    ]
    return [str(alias) for alias in aliases if alias not in NUMBER_SENTINELS]


def _render_header(row: Dict[str, Any], *, season_field: str = "season_year") -> Tuple[str, str]:
    season = row.get(season_field) or row.get("season")
    team = row.get("team_name") or row.get("team_code")
    player = row.get("player_name")
    if not player:
        player = f"선수 {row.get('player_id', '정보 미상')}"
    header = []
    if season:
        header.append(f"{season}년")
    if team:
        header.append(f"{team}")
    header_text = " ".join(header) if header else "2025년 KBO"
    subject = josa(player, ("은", "는"))
    return header_text, subject


def _tl_dr(pieces: List[str]) -> str:
    return "[TL;DR] " + " ".join(pieces)


def _detailed_lines(lines: Iterable[str]) -> str:
    valid = [line for line in lines if line]
    if not valid:
        return "[상세] 제공된 추가 세부 기록이 없습니다."
    bullet_lines = "\n".join(f"- {line}" for line in valid)
    return "[상세]\n" + bullet_lines


def render_pitching_season(
    row: Dict[str, Any],
    *,
    league_avg: Optional[Dict[str, Any]] = None,
    percentiles: Optional[Dict[str, Any]] = None,
    today_str: Optional[str],
) -> str:
    today = _today_fallback(today_str)
    header_text, subject = _render_header(row)
    ip = _format_ip(row.get("innings_pitched") or row.get("ip"))
    era = _format_float(row.get("era"), 2)
    whip = _format_float(row.get("whip"), 2)
    wins = _format_count(row.get("win") or row.get("wins"), "승")
    losses = _format_count(row.get("loss") or row.get("losses"), "패")
    saves = _format_count(row.get("save") or row.get("saves"), "세이브")
    holds = _format_count(row.get("hold") or row.get("holds"), "홀드")

    headline_parts = [header_text, subject]
    main_stats = []
    if ip:
        main_stats.append(f"{ip}")
    if era:
        main_stats.append(f"평균자책점 {era}")
    if whip:
        main_stats.append(f"WHIP {whip}")
    if wins or losses or saves:
        wl = " ".join(filter(None, [wins, losses, saves]))
        main_stats.append(wl.strip())
    tl_dr = _tl_dr([p for p in main_stats if p])
    if not main_stats:
        tl_dr = f"[TL;DR] {header_text} {subject} 공식 기록이 집계되지 않았습니다."

    core_sentences = []
    first_sentence_parts = [header_text, subject]
    if main_stats:
        first_sentence_parts.append(" ".join(main_stats))
    else:
        first_sentence_parts.append("등록된 시즌 주요 기록이 없습니다.")
    core_sentences.append("[핵심 문장] " + " ".join(first_sentence_parts) + "입니다.")

    strikeouts = _format_count(row.get("so") or row.get("strikeouts"), "탈삼진")
    hits_allowed = _format_count(row.get("h") or row.get("hits"), "피안타")
    walk = _format_count(row.get("bb") or row.get("walks"), "볼넷")
    opponent_avg = _format_float(row.get("opponent_avg") or row.get("baopp"), 3)
    detailed = []
    if strikeouts:
        detailed.append(f"{strikeouts}을 기록했습니다.")
    if hits_allowed:
        detailed.append(f"{hits_allowed}을 허용했습니다.")
    if walk:
        detailed.append(f"{walk}을 내줬습니다.")
    if opponent_avg:
        detailed.append(f"상대 타율은 {opponent_avg}입니다.")

    advanced = []
    if percentiles:
        era_pct = percentiles.get("era")
        if era_pct not in NUMBER_SENTINELS:
            advanced.append(f"ERA 백분위는 상위 {era_pct}%입니다.")
    detailed_section = _detailed_lines(detailed + advanced)

    meta = make_meta(
        row,
        kind="pitching_season",
        aliases=_build_aliases(row),
    )

    source = row.get("source_table", "player_season_pitching")
    row_id = row.get("source_row_id", "")
    source_line = (
        f"[출처] 2025 KBO 공식 기록 ({source}{'#' if row_id else ''}{row_id}) / 기준일 {today}"
    )

    return "\n".join(
        [
            tl_dr,
            core_sentences[0],
            detailed_section,
            f"[META] {meta}",
            source_line,
        ]
    )


def render_batting_season(
    row: Dict[str, Any],
    *,
    league_avg: Optional[Dict[str, Any]] = None,
    percentiles: Optional[Dict[str, Any]] = None,
    today_str: Optional[str],
) -> str:
    today = _today_fallback(today_str)
    header_text, subject = _render_header(row)
    avg = _format_float(row.get("avg") or row.get("batting_avg"), 3)
    ops = _format_float(row.get("ops"), 3)
    obp = _format_float(row.get("obp"), 3)
    slg = _format_float(row.get("slg"), 3)
    games = _format_count(row.get("g") or row.get("games"), "경기")
    plate = _format_count(row.get("pa") or row.get("plate_appearances"), "타석")
    home_runs = _format_count(row.get("hr") or row.get("home_runs"), "홈런")
    rbi = _format_count(row.get("rbi"), "타점")

    headline = [header_text, subject]
    main = []
    if games:
        main.append(games)
    if avg:
        main.append(f"타율 {avg}")
    if ops:
        main.append(f"OPS {ops}")
    if home_runs:
        main.append(home_runs)
    if rbi:
        main.append(rbi)
    tl_dr = _tl_dr(main if main else ["주요 기록이 제공되지 않았습니다."])

    first_sentence = [header_text, subject]
    if main:
        first_sentence.append(" ".join(main))
    else:
        first_sentence.append("등록된 시즌 주요 기록이 없습니다.")
    core = "[핵심 문장] " + " ".join(first_sentence) + "입니다."

    detailed = []
    if plate:
        detailed.append(f"{plate}에 나섰습니다.")
    if obp:
        detailed.append(f"출루율은 {obp}입니다.")
    if slg:
        detailed.append(f"장타율은 {slg}입니다.")
    if league_avg and avg and league_avg.get("avg") not in NUMBER_SENTINELS:
        league_avg_avg = _format_float(league_avg.get("avg"), 3)
        if league_avg_avg:
            detailed.append(f"리그 평균 타율 {league_avg_avg}와 비교해 주세요.")
    advanced = []
    if percentiles:
        ops_pct = percentiles.get("ops")
        if ops_pct not in NUMBER_SENTINELS:
            advanced.append(f"OPS 백분위는 상위 {ops_pct}%입니다.")
    detail_block = _detailed_lines(detailed + advanced)

    meta = make_meta(
        row,
        kind="batting_season",
        aliases=_build_aliases(row),
    )

    source = row.get("source_table", "player_season_batting")
    row_id = row.get("source_row_id", "")
    source_line = (
        f"[출처] 2025 KBO 공식 기록 ({source}{'#' if row_id else ''}{row_id}) / 기준일 {today}"
    )

    return "\n".join(
        [
            tl_dr,
            core,
            detail_block,
            f"[META] {meta}",
            source_line,
        ]
    )


if __name__ == "__main__":  # pragma: no cover - simple manual sanity check
    sample_pitcher = {
        "season_year": 2025,
        "team_name": "한화 이글스",
        "player_name": "김재영",
        "innings_pitched": 9,
        "era": 13.0,
        "whip": 1.85,
        "win": 1,
        "loss": 1,
        "so": 7,
        "player_id": "62754",
    }
    sample_batter = {
        "season_year": 2025,
        "team_name": "SSG 랜더스",
        "player_name": "최정",
        "avg": 0.345,
        "ops": 1.012,
        "obp": 0.420,
        "slg": 0.592,
        "g": 45,
        "pa": 180,
        "hr": 12,
        "rbi": 38,
        "player_id": "60003",
    }
    print(render_pitching_season(sample_pitcher, today_str="2025-05-01"))
    print("---")
    print(render_batting_season(sample_batter, today_str="2025-05-01"))
