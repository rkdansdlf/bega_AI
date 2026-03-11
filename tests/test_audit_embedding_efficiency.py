from scripts.audit_embedding_efficiency import (
    base_source_row_id,
    normalize_content,
    strip_source_line,
)


def test_base_source_row_id_strips_chunk_suffix() -> None:
    assert base_source_row_id("game_id=20250301SSLG0#part3") == "game_id=20250301SSLG0"


def test_base_source_row_id_keeps_non_chunk_source() -> None:
    assert base_source_row_id("id=54498") == "id=54498"


def test_normalize_content_collapses_whitespace_and_casefolds() -> None:
    raw = "  Alpha\n\nBETA\tGamma  "
    assert normalize_content(raw) == "alpha beta gamma"


def test_strip_source_line_removes_source_footer_only() -> None:
    raw = "요약: 테스트\n출처: game#id=10"
    assert strip_source_line(raw) == "요약: 테스트"
