from app.core.renderers.baseball import (
    render_batting_season,
    render_game_flow_summary,
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


def _parse_meta(rendered: str) -> dict:
    import json

    meta_line = [line for line in rendered.splitlines() if line.startswith("[META]")][0]
    return json.loads(meta_line.replace("[META] ", ""))


def test_render_game_flow_summary_late_comeback_metrics() -> None:
    row = {
        "game_id": "20250501LGHH0",
        "game_date": "2025-05-01",
        "home_team": "LG",
        "away_team": "HH",
        "home_team_name": "LG 트윈스",
        "away_team_name": "한화 이글스",
        "home_score": 3,
        "away_score": 2,
        "winning_team": "LG",
        "inning_lines_json": [
            {"inning": 1, "home_runs": 0, "away_runs": 2, "is_extra": False},
            {"inning": 5, "home_runs": 3, "away_runs": 0, "is_extra": False},
        ],
        "source_table": "game_flow_summary",
        "source_row_id": "game_id=20250501LGHH0",
    }

    rendered = render_game_flow_summary(row, today_str="2025-05-01")
    meta = _parse_meta(rendered)

    assert rendered.startswith("[TL;DR]")
    assert "5회말" in rendered
    assert meta["lead_changes"] == 1
    assert meta["tie_events"] == 0
    assert meta["decisive_half"] == "5회말"
    assert len(rendered) <= 800


def test_render_game_flow_summary_wire_to_wire_game() -> None:
    row = {
        "game_id": "20250502LGHH0",
        "game_date": "2025-05-02",
        "home_team": "LG",
        "away_team": "HH",
        "home_team_name": "LG 트윈스",
        "away_team_name": "한화 이글스",
        "home_score": 1,
        "away_score": 4,
        "winning_team": "HH",
        "inning_lines_json": [
            {"inning": 1, "home_runs": 0, "away_runs": 2, "is_extra": False},
            {"inning": 4, "home_runs": 0, "away_runs": 1, "is_extra": False},
            {"inning": 7, "home_runs": 0, "away_runs": 1, "is_extra": False},
            {"inning": 8, "home_runs": 1, "away_runs": 0, "is_extra": False},
        ],
    }

    rendered = render_game_flow_summary(row, today_str="2025-05-02")
    meta = _parse_meta(rendered)

    assert meta["lead_changes"] == 0
    assert meta["tie_events"] == 0
    assert meta["decisive_half"] == "1회초"


def test_render_game_flow_summary_extra_innings_bucket_totals() -> None:
    row = {
        "game_id": "20250503LGHH0",
        "game_date": "2025-05-03",
        "home_team": "LG",
        "away_team": "HH",
        "home_team_name": "LG 트윈스",
        "away_team_name": "한화 이글스",
        "home_score": 4,
        "away_score": 5,
        "winning_team": "HH",
        "inning_lines_json": [
            {"inning": 2, "home_runs": 1, "away_runs": 0, "is_extra": False},
            {"inning": 6, "home_runs": 2, "away_runs": 3, "is_extra": False},
            {"inning": 9, "home_runs": 1, "away_runs": 1, "is_extra": False},
            {"inning": 10, "home_runs": 0, "away_runs": 1, "is_extra": True},
        ],
    }

    rendered = render_game_flow_summary(row, today_str="2025-05-03")
    meta = _parse_meta(rendered)

    assert meta["bucket_totals"]["away"]["extra"] == 1
    assert meta["bucket_totals"]["home"]["extra"] == 0
    assert "연장" in rendered


def test_render_game_flow_summary_draw_has_no_decisive_half() -> None:
    row = {
        "game_id": "20250504LGHH0",
        "game_date": "2025-05-04",
        "home_team": "LG",
        "away_team": "HH",
        "home_team_name": "LG 트윈스",
        "away_team_name": "한화 이글스",
        "home_score": 3,
        "away_score": 3,
        "winning_team": None,
        "inning_lines_json": [
            {"inning": 1, "home_runs": 0, "away_runs": 1, "is_extra": False},
            {"inning": 1, "home_runs": 1, "away_runs": 1, "is_extra": False},
            {"inning": 8, "home_runs": 0, "away_runs": 2, "is_extra": False},
            {"inning": 8, "home_runs": 2, "away_runs": 2, "is_extra": False},
        ],
    }

    rendered = render_game_flow_summary(row, today_str="2025-05-04")
    meta = _parse_meta(rendered)

    assert "무승부" in rendered
    assert meta["decisive_half"] is None


def test_render_game_flow_summary_resolves_team_codes_without_name_fields() -> None:
    row = {
        "game_id": "20250505SKHH0",
        "game_date": "2025-05-05",
        "home_team": "SK",
        "away_team": "HH",
        "home_score": 4,
        "away_score": 2,
        "winning_team": "SK",
        "inning_lines_json": [
            {"inning": 1, "home_runs": 1, "away_runs": 0, "is_extra": False},
            {"inning": 8, "home_runs": 3, "away_runs": 2, "is_extra": False},
        ],
    }

    rendered = render_game_flow_summary(row, today_str="2025-05-05")

    assert "SK 와이번스" in rendered
    assert "한화 이글스" in rendered
