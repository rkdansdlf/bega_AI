from app.core.chat_cache_key import has_temporal_keyword


def test_series_query_without_temporal_context_is_cacheable() -> None:
    assert has_temporal_keyword("한국시리즈 역대 MVP 알려줘") is False


def test_series_query_with_today_context_bypasses_cache() -> None:
    assert has_temporal_keyword("오늘 시리즈 현황 알려줘") is True


def test_series_query_with_realtime_context_bypasses_cache() -> None:
    assert has_temporal_keyword("실시간 시리즈 중계 알려줘") is True


def test_existing_temporal_keyword_still_bypasses_cache() -> None:
    assert has_temporal_keyword("오늘 경기 결과 알려줘") is True
