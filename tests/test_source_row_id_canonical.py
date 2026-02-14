from scripts.ingest_from_kbo import build_canonical_source_row_id, build_source_row_id


def test_build_canonical_source_row_id_game() -> None:
    row = {"id": 54498, "game_id": "20251031LGHH0"}
    assert build_canonical_source_row_id(row, "game") == "id=54498"


def test_build_canonical_source_row_id_game_metadata() -> None:
    row = {"game_id": "20250301SSLG0", "stadium_code": "JAMSIL"}
    assert (
        build_canonical_source_row_id(row, "game_metadata")
        == "game_id=20250301SSLG0"
    )


def test_build_canonical_source_row_id_stadium_alias() -> None:
    row = {"stadium_id": "CHANGWON"}
    assert build_canonical_source_row_id(row, "stadiums") == "stadium_id=NCPARK"


def test_build_canonical_source_row_id_player_movements() -> None:
    row = {"id": 1000, "date": "2024-07-09", "player_name": "홍길동"}
    assert build_canonical_source_row_id(row, "player_movements") == "id=1000"


def test_build_source_row_id_fallback_when_canonical_missing() -> None:
    row = {"team_code": "LG", "season": 2025}
    built = build_source_row_id(
        row=row,
        table="team_history",
        pk_columns=(),
        pk_hint=("team_code", "season"),
    )
    assert built == "team_code=LG|season=2025"
