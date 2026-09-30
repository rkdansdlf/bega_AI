from datetime import date, datetime

from app.routers import coach_auto_brief_ops


class _UtcBoundaryDate:
    @classmethod
    def today(cls) -> date:
        return date(2026, 7, 29)


class _KstBoundaryDateTime:
    @classmethod
    def now(cls, tz) -> datetime:
        assert getattr(tz, "key", None) == "Asia/Seoul"
        return datetime(2026, 7, 30, 3, 0, 0, tzinfo=tz)


def test_today_and_tomorrow_windows_use_kbo_business_timezone(monkeypatch) -> None:
    monkeypatch.setattr(coach_auto_brief_ops, "date", _UtcBoundaryDate)
    monkeypatch.setattr(coach_auto_brief_ops, "datetime", _KstBoundaryDateTime)

    today = coach_auto_brief_ops._resolve_requested_window(
        window="today",
        start_date=None,
        end_date=None,
    )
    tomorrow = coach_auto_brief_ops._resolve_requested_window(
        window="tomorrow",
        start_date=None,
        end_date=None,
    )

    assert today == (date(2026, 7, 30), date(2026, 7, 30))
    assert tomorrow == (date(2026, 7, 31), date(2026, 7, 31))


def test_explicit_date_windows_are_not_shifted_by_business_timezone(
    monkeypatch,
) -> None:
    monkeypatch.setattr(coach_auto_brief_ops, "datetime", _KstBoundaryDateTime)

    explicit_range = coach_auto_brief_ops._resolve_requested_window(
        window="custom",
        start_date=date(2026, 7, 29),
        end_date=date(2026, 7, 31),
    )
    explicit_single_day = coach_auto_brief_ops._resolve_requested_window(
        window="custom",
        start_date=date(2026, 8, 1),
        end_date=None,
    )

    assert explicit_range == (date(2026, 7, 29), date(2026, 7, 31))
    assert explicit_single_day == (date(2026, 8, 1), date(2026, 8, 1))
