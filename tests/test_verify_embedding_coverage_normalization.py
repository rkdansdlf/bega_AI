from scripts.verify_embedding_coverage import normalize_actual_source_row_id


def test_normalize_actual_source_row_id_game_legacy_composite() -> None:
    raw = "id=54498|game_id=20251031LGHH0"
    assert normalize_actual_source_row_id(raw, "game") == "id=54498"


def test_normalize_actual_source_row_id_game_metadata_prefers_game_id() -> None:
    raw = "id=123|game_id=20250301SSLG0"
    assert normalize_actual_source_row_id(raw, "game_metadata") == "game_id=20250301SSLG0"


def test_normalize_actual_source_row_id_stadium_alias_and_part_suffix() -> None:
    raw = "stadium_id=CHANGWON#part1"
    assert normalize_actual_source_row_id(raw, "stadiums") == "stadium_id=NCPARK"


def test_normalize_actual_source_row_id_unknown_table_keeps_base() -> None:
    raw = "foo=bar#part2"
    assert normalize_actual_source_row_id(raw, "unknown_table") == "foo=bar"


def test_normalize_actual_source_row_id_player_movements_id_only() -> None:
    raw = "id=1000|date=2024-07-09|player_name=홍길동"
    assert normalize_actual_source_row_id(raw, "player_movements") == "id=1000"


def test_normalize_actual_source_row_id_uses_meta_compat_key_for_game() -> None:
    raw = "game_id=20251031LGHH0"
    meta = {"id": 54498}
    assert normalize_actual_source_row_id(raw, "game", meta) == "id=54498"


def test_normalize_actual_source_row_id_uses_meta_compat_key_for_game_summary() -> None:
    raw = "game_id=20250415HHSK0|summary_type=폭투|detail_text=폰세(2회)"
    meta = {
        "id": 150464,
    }
    expected = "id=150464"
    assert normalize_actual_source_row_id(raw, "game_summary", meta) == expected


def test_normalize_actual_source_row_id_legacy_alias_maps_to_canonical() -> None:
    raw = "game_id=20251031LGHH0"
    alias = {"game_id=20251031LGHH0": "id=54498"}
    assert normalize_actual_source_row_id(raw, "game", legacy_aliases=alias) == "id=54498"


def test_normalize_actual_source_row_id_legacy_alias_not_found_keeps_legacy() -> None:
    raw = "game_id=20251031LGHH0"
    assert normalize_actual_source_row_id(raw, "game") == "game_id=20251031LGHH0"
