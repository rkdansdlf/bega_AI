"""KBO business-clock helpers.

Baseball dates are interpreted in Korea Standard Time regardless of the host or
container timezone.  Tests may pass an aware datetime to make boundary cases
deterministic.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

KBO_BUSINESS_TIMEZONE = ZoneInfo("Asia/Seoul")


def kst_now(now: datetime | None = None) -> datetime:
    """Return an aware datetime normalized to the KBO business timezone."""

    if now is None:
        return datetime.now(KBO_BUSINESS_TIMEZONE)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(KBO_BUSINESS_TIMEZONE)


def kst_today(now: datetime | None = None) -> date:
    """Return the current KBO business date."""

    return kst_now(now).date()
