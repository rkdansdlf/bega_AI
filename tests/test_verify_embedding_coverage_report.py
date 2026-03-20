from scripts.verify_embedding_coverage import _status_for_counts, build_targets


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
