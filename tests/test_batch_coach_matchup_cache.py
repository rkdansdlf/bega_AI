from scripts.batch_coach_matchup_cache import _extract_error_message


def test_extract_error_message_reads_error_field() -> None:
    payload = '{"error": "rate limit", "message": "fallback"}'
    assert _extract_error_message(payload) == "rate limit"


def test_extract_error_message_ignores_empty_error_field() -> None:
    assert _extract_error_message('{"error": "", "message": "   "}') is None


def test_extract_error_message_ignores_empty_payload() -> None:
    assert _extract_error_message("   ") is None


def test_extract_error_message_uses_message_when_error_empty() -> None:
    payload = '{"error": "", "message": "detail available"}'
    assert _extract_error_message(payload) == "detail available"


def test_extract_error_message_fallback_to_raw_non_json() -> None:
    assert _extract_error_message("service unavailable") == "service unavailable"
