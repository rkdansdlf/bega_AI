"""InMemoryRateLimiter / 클라이언트 키 추출 테스트."""

import asyncio

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers

from app.core import ratelimit
from app.core.ratelimit import OVERFLOW_KEY, InMemoryRateLimiter, _extract_client_key


class _Clock:
    """time.time 을 대체하는 수동 시계."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(ratelimit.time, "time", fake)
    return fake


def _check(limiter: InMemoryRateLimiter, key: str) -> None:
    asyncio.run(limiter.check(key))


def _is_rejected(limiter: InMemoryRateLimiter, key: str) -> bool:
    try:
        _check(limiter, key)
    except HTTPException as exc:
        assert exc.status_code == 429
        return True
    return False


# --- 기존 동작 고정 ---------------------------------------------------------


def test_limit_is_enforced_per_key(clock):
    limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60)

    for _ in range(3):
        _check(limiter, "a")
    assert _is_rejected(limiter, "a") is True
    # 다른 키는 영향을 받지 않는다.
    _check(limiter, "b")


def test_rejection_carries_retry_after_header(clock):
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
    _check(limiter, "a")
    clock.now += 10

    with pytest.raises(HTTPException) as caught:
        _check(limiter, "a")

    assert caught.value.status_code == 429
    assert caught.value.headers["Retry-After"] == "50"


def test_requests_are_allowed_again_after_the_window(clock):
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
    _check(limiter, "a")
    assert _is_rejected(limiter, "a") is True

    clock.now += 61

    _check(limiter, "a")


# --- 메모리 보호 ------------------------------------------------------------


def test_expired_keys_are_swept(clock):
    limiter = InMemoryRateLimiter(max_requests=5, window_seconds=60)
    for index in range(50):
        _check(limiter, f"client-{index}")
    assert limiter.tracked_keys == 50

    clock.now += 61
    _check(limiter, "fresh")

    # 만료된 50개는 사라지고 새 키만 남는다.
    assert limiter.tracked_keys == 1


def test_active_keys_survive_the_sweep(clock):
    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)
    _check(limiter, "busy")  # t=1000
    clock.now += 30
    _check(limiter, "busy")  # t=1030 (가장 최근 요청)
    _check(limiter, "idle")  # t=1030

    clock.now += 31  # t=1061: busy 의 첫 요청(1000)은 만료, 최근 요청(1030)은 윈도우 안
    _check(limiter, "trigger-sweep")

    # 최근 요청이 윈도우 안에 있는 키는 키 단위 정리에서 제거되지 않는다.
    assert "busy" in limiter._hits
    # 키 안의 만료된 기록은 그 키가 다시 호출될 때 기존 방식대로 걷어낸다.
    _check(limiter, "busy")
    assert list(limiter._hits["busy"]) == [1030.0, 1061.0]


def test_distinct_keys_are_capped_and_share_an_overflow_bucket(clock):
    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60, max_keys=5)

    # 위조된 키가 계속 들어오는 상황: 메모리는 상한에서 멈춰야 한다.
    outcomes = []
    for index in range(100):
        outcomes.append(_is_rejected(limiter, f"spoofed-{index}"))

    assert limiter.tracked_keys <= 5 + 1  # 상한 + overflow 버킷
    assert OVERFLOW_KEY in limiter._hits
    # 상한을 넘은 키들은 하나의 버킷을 공유하므로 제한(2회)을 피할 수 없다.
    assert any(outcomes)
    assert outcomes.count(False) <= 5 + 2


def test_known_keys_keep_their_own_bucket_when_at_capacity(clock):
    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60, max_keys=3)
    for name in ("legit-1", "legit-2", "legit-3"):
        _check(limiter, name)

    # 가득 찬 뒤에도 기존 키는 자기 버킷으로 계속 정상 처리된다(공유 버킷에 휘말리지 않음).
    _check(limiter, "legit-1")
    assert _is_rejected(limiter, "legit-1") is True
    _check(limiter, "legit-2")


def test_capacity_is_recovered_by_emergency_sweep(clock):
    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60, max_keys=3)
    for name in ("a", "b", "c"):
        _check(limiter, name)

    clock.now += 61  # 모두 만료 -> 긴급 정리로 새 키를 정상 수용
    _check(limiter, "d")

    assert OVERFLOW_KEY not in limiter._hits
    assert "d" in limiter._hits


# --- 클라이언트 키 추출 -----------------------------------------------------


class _Client:
    host = "10.0.0.7"


class _Request:
    """starlette 의 실제 Headers 를 쓰는 최소 요청. 여러 XFF 줄도 표현할 수 있다."""

    def __init__(self, *xff_lines: str, client=_Client):
        self.headers = Headers(
            raw=[(b"x-forwarded-for", line.encode()) for line in xff_lines]
        )
        self.client = client() if client else None


def test_default_ignores_forwarded_for_header():
    # 기본값에서는 클라이언트가 보낸 X-Forwarded-For 로 키를 바꿀 수 없어야 한다.
    request = _Request("1.2.3.4")

    assert _extract_client_key(request) == "10.0.0.7"


def test_client_key_falls_back_to_unknown_without_peer():
    assert _extract_client_key(_Request(client=None)) == "unknown"


def _trusted(request, count=1) -> str:
    return _extract_client_key(
        request, trust_x_forwarded_for=True, trusted_proxy_count=count
    )


def test_trusted_xff_uses_the_entry_appended_by_the_proxy_not_the_spoofed_one():
    # 클라이언트가 "6.6.6.6" 을 보내고, 프록시가 실제 연결 주소를 오른쪽에 덧붙인 경우.
    request = _Request("6.6.6.6, 203.0.113.9")

    assert _trusted(request) == "203.0.113.9"


def test_rotating_a_spoofed_xff_does_not_change_the_key():
    # 위조 값을 바꿔도 프록시가 덧붙인 값이 같으면 같은 키(같은 버킷)다.
    keys = {_trusted(_Request(f"6.6.6.{index}, 203.0.113.9")) for index in range(50)}

    assert keys == {"203.0.113.9"}


def test_trusted_proxy_count_selects_the_nth_entry_from_the_right():
    # client -> proxy1 -> proxy2 -> app : XFF = "spoof, client, proxy1"
    request = _Request("6.6.6.6, 203.0.113.9, 198.51.100.2")

    assert _trusted(request, count=2) == "203.0.113.9"


def test_single_value_xff_from_an_overwriting_proxy_still_works():
    assert _trusted(_Request("203.0.113.9")) == "203.0.113.9"


def test_multiple_xff_header_lines_are_treated_as_one_chain():
    request = _Request("6.6.6.6", "203.0.113.9")

    assert _trusted(request) == "203.0.113.9"


def test_trusted_xff_falls_back_to_peer_when_chain_is_shorter_than_expected():
    assert _trusted(_Request("203.0.113.9"), count=2) == "10.0.0.7"
    assert _trusted(_Request()) == "10.0.0.7"


def test_trusted_xff_falls_back_to_peer_for_non_ip_values():
    # 임의 문자열이 키가 되어 메모리를 낭비하거나 버킷을 위조하지 못하게 한다.
    assert _trusted(_Request("6.6.6.6, not-an-ip")) == "10.0.0.7"
    assert _trusted(_Request("x" * 5000)) == "10.0.0.7"


def test_trusted_xff_normalizes_ipv6():
    assert _trusted(_Request("2001:DB8:0:0:0:0:0:1")) == "2001:db8::1"


@pytest.mark.parametrize(
    "raw, expected",
    [(None, 1), ("", 1), ("1", 1), ("2", 2), (" 3 ", 3), ("0", 1), ("-4", 1), ("abc", 1)],
)
def test_trusted_proxy_count_parsing(raw, expected):
    assert ratelimit._parse_trusted_proxy_count(raw) == expected
