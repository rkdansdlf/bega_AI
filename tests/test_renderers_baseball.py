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
    assert any(
        line.startswith("[핵심 문장]") for line in sections
    ), "core sentence missing"
    assert any(
        section.startswith("[상세]") for section in sections
    ), "detail section missing"
    assert any(
        section.startswith("[META]") for section in sections
    ), "meta section missing"
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


def test_render_batting_metrics_precalc():
    """타자 메트릭(wRC+, WAR, Score) 선계산 검증"""
    row = {
        "season_year": 2024,
        "team_name": "KIA",
        "player_name": "Kim",
        "avg": 0.350,
        "ops": 1.100,
        "g": 100,
        "pa": 400,
        "hr": 30,
        "id": "test_batter_1",
        # Added counting stats for wOBA/wRC+ calculation
        "hits": 120,
        "at_bats": 340,
        "walks": 50,
        "hbp": 5,
        "doubles": 25,
        "triples": 2,
        "sacrifice_flies": 5,
    }

    # render 함수 내부에서 make_meta 호출 시 extra_stats가 주입되는지 확인
    rendered = render_batting_season(row, today_str="2024-10-01")

    # [META] {json} 파싱
    import json

    meta_line = [line for line in rendered.splitlines() if line.startswith("[META]")][0]
    meta_json = meta_line.replace("[META] ", "")
    meta_data = json.loads(meta_json)

    # 선계산된 필드가 존재해야 함
    assert "wrc_plus" in meta_data
    assert "war" in meta_data
    assert "score" in meta_data

    # OPS 1.100이면 wRC+는 100보다 훨씬 커야 함
    if meta_data["wrc_plus"] is not None:
        assert meta_data["wrc_plus"] > 100
    if meta_data["score"]:
        assert meta_data["score"] > 90  # High score expected


def test_render_pitching_metrics_precalc():
    """투수 메트릭(FIP, ERA-, Score) 선계산 검증"""
    row = {
        "season_year": 2024,
        "team_name": "KIA",
        "player_name": "Yang",
        "innings_pitched": 150.0,
        "era": 2.50,  # Excellent ERA
        "whip": 1.05,
        "strikeouts": 150,
        "walks_allowed": 30,
        "home_runs_allowed": 10,
        "hit_batters": 5,
        "games_started": 25,
        "id": "test_pitcher_1",
    }

    rendered = render_pitching_season(row, today_str="2024-10-01")

    import json

    meta_line = [line for line in rendered.splitlines() if line.startswith("[META]")][0]
    meta_json = meta_line.replace("[META] ", "")
    meta_data = json.loads(meta_json)

    assert "era_minus" in meta_data
    assert "fip_minus" in meta_data
    assert "score" in meta_data
    assert meta_data.get("role") == "SP"

    # ERA 2.50 -> ERA- should be low (e.g. < 80 check context dependent but <100 is sure)
    if meta_data["era_minus"] is not None:
        assert meta_data["era_minus"] < 100
    if meta_data["score"]:
        assert meta_data["score"] > 90  # High score expected
