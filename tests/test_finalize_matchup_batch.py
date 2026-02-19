from scripts.finalize_matchup_batch import _ensure_completed


def test_ensure_completed_ok() -> None:
    ok, reason = _ensure_completed({"cases": 90, "failed": 0, "in_progress": 0}, 90)
    assert ok
    assert reason == "ok"


def test_ensure_completed_case_mismatch() -> None:
    ok, reason = _ensure_completed({"cases": 80, "failed": 0, "in_progress": 0}, 90)
    assert not ok
    assert reason == "cases 80 != expected 90"


def test_ensure_completed_failed_exists() -> None:
    ok, reason = _ensure_completed({"cases": 90, "failed": 1, "in_progress": 0}, 90)
    assert not ok
    assert reason == "failed 1 != 0"


def test_ensure_completed_in_progress() -> None:
    ok, reason = _ensure_completed({"cases": 90, "failed": 0, "in_progress": 1}, 90)
    assert not ok
    assert reason == "in_progress remains"
