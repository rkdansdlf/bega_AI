from scripts.verify_embedding_coverage import (
    _status_for_counts,
    CoverageTarget,
    build_actual_rows_query,
    build_targets,
)


def test_status_for_counts_handles_missing_and_extra() -> None:
    assert _status_for_counts(0, 0) == "OK"
    assert _status_for_counts(2, 0) == "MISSING"
    assert _status_for_counts(0, 3) == "EXTRA"
    assert _status_for_counts(1, 1) == "MISSING+EXTRA"


def test_build_targets_includes_game_flow_summary() -> None:
    targets = build_targets("seasonal", 2025, 2025)
    assert any(target.table == "game_flow_summary" for target in targets)


def test_build_targets_includes_static_source_file_profiles() -> None:
    targets = build_targets("static", 2025, 2025)

    assert any(
        target.table.startswith("markdown_docs_")
        and target.source_table == "markdown_docs"
        for target in targets
    )
    assert any(
        target.table.startswith("kbo_regulations_")
        and target.source_table == "kbo_regulations"
        for target in targets
    )
    assert any(target.source_table == "kbo_definitions" for target in targets)


def test_build_actual_rows_query_scopes_static_source_file_target() -> None:
    query, params = build_actual_rows_query(
        CoverageTarget(
            table="markdown_docs_rules_terms",
            year=0,
            source_table="markdown_docs",
        )
    )

    assert "source_table = %s" in query
    assert "source_row_id = %s OR source_row_id LIKE %s" in query
    assert params[0] == "markdown_docs"
    assert params[1].startswith("markdown_docs:")
    assert params[2].startswith(params[1])


def test_build_actual_rows_query_uses_season_for_seasonal_target() -> None:
    query, params = build_actual_rows_query(
        CoverageTarget(table="game", year=2021, source_table="game")
    )

    assert "season_year = %s" in query
    assert params == ("game", 2021)
