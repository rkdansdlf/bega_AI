import textwrap

from app.core.renderers.baseball import (
    render_batting_season,
    render_pitching_season,
)


def _normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def test_render_pitching_season_snapshot():
    row = {
        "season_year": 2025,
        "team_name": "한화 이글스",
        "player_name": "김재영",
        "innings_pitched": 9,
        "era": 13.0,
        "whip": 1.85,
        "win": 1,
        "loss": 1,
        "save": 0,
        "so": 7,
        "player_id": "62754",
        "source_table": "player_season_pitching",
        "source_row_id": "player_id=62754|season_year=2025",
    }
    rendered = render_pitching_season(row, today_str="2025-05-01")
    snapshot = _normalize(rendered)
    assert snapshot.startswith("[TL;DR]"), "must start with TL;DR section"
    sections = [section.strip() for section in rendered.split("\n") if section.strip()]
    assert any(line.startswith("[핵심 문장]") for line in sections), "core sentence missing"
    assert any(section.startswith("[상세]") for section in sections), "detail section missing"
    assert any(section.startswith("[META]") for section in sections), "meta section missing"
    assert sections[-1].startswith("[출처]"), "source section missing"
    assert "평균자책점 13.00" in rendered
    assert "WHIP 1.85" in rendered
    assert "1승 1패" in rendered or "1승 1패 0세이브" in rendered


def test_render_batting_season_snapshot():
    row = {
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
        "source_table": "player_season_batting",
        "source_row_id": "player_id=60003|season_year=2025",
    }
    rendered = render_batting_season(row, today_str="2025-05-01")
    snapshot = _normalize(rendered)
    assert snapshot.startswith("[TL;DR]")
    sections = [section.strip() for section in rendered.split("\n") if section.strip()]
    assert any(line.startswith("[핵심 문장]") for line in sections)
    assert any(section.startswith("[상세]") for section in sections)
    assert any(section.startswith("[META]") for section in sections)
    assert sections[-1].startswith("[출처]")
    assert "타율 0.345" in rendered
    assert "OPS 1.012" in rendered
    assert "홈런 12" in rendered
