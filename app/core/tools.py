"""자주 쓰는 통계 질의를 SQL로 처리하는 도우미 모듈."""

import re
from typing import Any, Dict, List, Optional

from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import RealDictCursor


OPS_TOP = re.compile(
    r"(20\d{2}).*?(LG|SSG|두산|삼성|롯데|KIA|한화|키움|NC|KT).*(OPS).*(상위|TOP)\s*(\d+)",
    re.I,
)

AVG_TOP = re.compile(
    r"(20\d{2}).*?(?:시즌)?\s*(?:KBO)?\s*(타율|평균).*?(?:상위|TOP)?\s*(\d+)?",
    re.I,
)

HR_TOP = re.compile(
    r"(20\d{2}).*?(?:시즌)?\s*(?:KBO)?\s*(홈런|HR).*?(?:상위|TOP)?\s*(\d+)?",
    re.I,
)

TEAM_MAP = {
    "LG": "LG",
    "두산": "OB",
    "삼성": "SS",
    "롯데": "LT",
    "KIA": "HT",
    "한화": "HH",
    "키움": "WO",
    "NC": "NC",
    "KT": "KT",
    "SSG": "SK",
}


def select_rows(conn: PgConnection, sql: str, params) -> list[Dict[str, Any]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def _fmt_stat(value: Any, ndigits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def _player_label(row: Dict[str, Any]) -> str:
    name = row.get("player_name")
    pid = row.get("player_id")
    if name and pid:
        return f"{name} ({pid})"
    return name or str(pid) or "-"


def _build_table(title: str, headers: List[str], rows: List[List[str]]) -> str:
    lines = [f"## {title}", ""]
    lines.append(" | ".join(headers))
    lines.append(" | ".join([":---" if i else ":--:" for i in range(len(headers))]))
    for row in rows:
        lines.append(" | ".join(row))
    return "\n".join(lines)


def _format_league_avg(rows, year: str) -> tuple[str, list[Dict[str, Any]]]:
    table_rows: List[List[str]] = []
    citations: list[Dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        player_label = _player_label(row)
        table_rows.append(
            [
                str(idx),
                player_label,
                row["team_id"],
                _fmt_stat(row.get("avg")),
                _fmt_stat(row.get("pa"), ndigits=0),
            ]
        )
        citations.append(
            {
                "id": row["id"],
                "title": f"{year} {row['team_id']} {player_label}",
            }
        )
    text = _build_table(
        f"{year} KBO 타율 상위 {len(rows)}",
        ["순위", "선수", "팀", "타율", "PA"],
        table_rows,
    )
    return text, citations


def try_answer_with_sql(
    conn: PgConnection, user_query: str, filters: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    match = OPS_TOP.search(user_query)
    if match:
        year, team_kr, _, _, topk = match.groups()
        team_id = TEAM_MAP.get(team_kr)
        if not team_id:
            return None
        sql = """
          SELECT p.player_name,
                 bs.player_id,
                 bs.team_id,
                 bs.avg,
                 bs.obp,
                 bs.slg,
                 bs.ops,
                 bs.hr,
                 bs.rbi,
                 bs.pa,
                 bs.ab,
                 bs.id
          FROM player_season_batting bs
          LEFT JOIN player_basic p ON p.player_id = bs.player_id
          WHERE bs.season_year = %s AND bs.team_id = %s
          ORDER BY bs.ops DESC
          LIMIT %s
        """
        rows = select_rows(conn, sql, [year, team_id, int(topk)])
        if not rows:
            return None
        table_rows: List[List[str]] = []
        citations = []
        for idx, row in enumerate(rows, 1):
            player_label = _player_label(row)
            table_rows.append(
                [
                    str(idx),
                    player_label,
                    _fmt_stat(row.get("team_id")),
                    _fmt_stat(row.get("ops")),
                    _fmt_stat(row.get("obp")),
                    _fmt_stat(row.get("slg")),
                    _fmt_stat(row.get("avg")),
                    _fmt_stat(row.get("hr"), ndigits=0),
                    _fmt_stat(row.get("rbi"), ndigits=0),
                    _fmt_stat(row.get("pa"), ndigits=0),
                ]
            )
            citations.append(
                {"id": row["id"], "title": f"{year} {team_kr} {player_label}"}
            )
        text = _build_table(
            f"{year} {team_kr} OPS 상위 {topk}",
            ["순위", "선수", "팀", "OPS", "OBP", "SLG", "AVG", "HR", "RBI", "PA"],
            table_rows,
        )
        return {
            "text": text,
            "citations": citations,
            "confidence": 0.9,
        }

    match = HR_TOP.search(user_query)
    if match:
        year, _, topk_str = match.groups()
        topk = int(topk_str) if topk_str else 5
        sql = """
          SELECT p.player_name,
                 bs.player_id,
                 bs.team_id,
                 bs.avg,
                 bs.obp,
                 bs.slg,
                 bs.ops,
                 bs.hr,
                 bs.rbi,
                 bs.pa,
                 bs.id
          FROM player_season_batting bs
          LEFT JOIN player_basic p ON p.player_id = bs.player_id
          WHERE bs.season_year = %s
          ORDER BY bs.hr DESC, bs.pa DESC
          LIMIT %s
        """
        rows = select_rows(conn, sql, [year, topk])
        if not rows:
            return None
        table_rows = []
        citations = []
        for idx, row in enumerate(rows, 1):
            player_label = _player_label(row)
            table_rows.append(
                [
                    str(idx),
                    player_label,
                    row["team_id"],
                    _fmt_stat(row.get("avg")),
                    _fmt_stat(row.get("obp")),
                    _fmt_stat(row.get("slg")),
                    _fmt_stat(row.get("ops")),
                    _fmt_stat(row.get("hr"), ndigits=0),
                    _fmt_stat(row.get("rbi"), ndigits=0),
                    _fmt_stat(row.get("pa"), ndigits=0),
                ]
            )
            citations.append(
                {
                    "id": row["id"],
                    "title": f"{year} {row['team_id']} {player_label}",
                }
            )
        text = _build_table(
            f"{year} KBO 홈런 상위 {len(rows)}",
            [
                "순위",
                "선수",
                "팀",
                "AVG",
                "OBP",
                "SLG",
                "OPS",
                "HR",
                "RBI",
                "PA",
            ],
            table_rows,
        )
        return {
            "text": text,
            "citations": citations,
            "confidence": 0.9,
        }

    match = AVG_TOP.search(user_query)
    if match:
        year, _, topk_str = match.groups()
        topk = int(topk_str) if topk_str else 5
        sql = """
          SELECT p.player_name, bs.player_id, bs.team_id, bs.avg, bs.pa, bs.id
          FROM player_season_batting bs
          LEFT JOIN player_basic p ON p.player_id = bs.player_id
          WHERE bs.season_year = %s
          ORDER BY bs.avg DESC
          LIMIT %s
        """
        rows = select_rows(conn, sql, [year, topk])
        if not rows:
            return None
        text, citations = _format_league_avg(rows, year)
        return {
            "text": text,
            "citations": citations,
            "confidence": 0.85,
        }

    return None
