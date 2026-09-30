"""KST business-clock regression tests."""

from datetime import datetime, timezone

from app.core.entity_extractor import extract_year
from app.core.time import kst_now, kst_today


def test_kst_date_rolls_over_nine_hours_before_utc_date() -> None:
    instant = datetime(2026, 8, 7, 15, 5, tzinfo=timezone.utc)

    assert kst_today(instant).isoformat() == "2026-08-08"
    assert kst_now(instant).hour == 0


def test_kst_clock_rejects_naive_injected_datetime() -> None:
    try:
        kst_now(datetime(2026, 8, 8, 0, 0))
    except ValueError as exc:
        assert str(exc) == "now must be timezone-aware"
    else:
        raise AssertionError("naive datetimes must not be accepted")


def test_current_season_synonyms_resolve_to_kst_current_year() -> None:
    expected_year = kst_today().year

    for query in ("올해 기록", "금년 기록", "이번 시즌 기록", "올시즌 기록"):
        assert extract_year(query) == expected_year
