"""AI 서비스 보안 이벤트 카운터 유틸리티."""

import logging
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_events: dict[tuple[str, str, str], int] = defaultdict(int)


def record_security_event(
    event: str,
    *,
    endpoint: str = "",
    detail: str = "",
) -> int:
    """기본 보안 이벤트 카운터를 증가시킵니다."""
    key = (event, endpoint, detail)
    with _lock:
        _events[key] += 1
        count = _events[key]

    logger.warning(
        "security_event=%s endpoint=%s detail=%s count=%d",
        event,
        endpoint or "unknown",
        detail,
        count,
    )
    return count


def get_security_event_counters() -> dict[str, int]:
    """테스트/모니터링용 보안 이벤트 스냅샷을 반환합니다."""
    with _lock:
        return {
            f"{event}|{endpoint}|{detail}": count for (event, endpoint, detail), count in _events.items()
        }
