import json

import pytest
from fastapi.testclient import TestClient

from app.eval.event_mining import build_candidates, merge_feedback
from app.routers import feedback as fb
from scripts import mine_retrieval_events as miner


class _Cur:
    async def fetchone(self):
        return (42,)


class _Conn:
    def __init__(self):
        self.calls = []

    async def execute(self, sql, params):
        self.calls.append((sql, params))
        return _Cur()


class _Pool:
    def __init__(self):
        self.conn = _Conn()

    def connection(self):
        conn = self.conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *a):
                return False

        return _Ctx()


def _client(monkeypatch, pool):
    from app import main

    monkeypatch.setattr(fb, "get_connection_pool", lambda: pool)
    app = main.create_app()
    app.dependency_overrides[main.require_ai_internal_token] = lambda: None
    return TestClient(app)


def test_feedback_requires_internal_token(monkeypatch):
    from app import main

    monkeypatch.setattr(fb, "get_connection_pool", lambda: _Pool())
    res = TestClient(main.create_app()).post(
        "/ai/chat/feedback", json={"question": "q", "rating": "DOWN"}
    )
    assert res.status_code in (401, 403)


def test_feedback_is_stored_with_fingerprint(monkeypatch):
    pool = _Pool()
    res = _client(monkeypatch, pool).post(
        "/ai/chat/feedback",
        json={
            "question": "2025 LG 성적",
            "rating": "DOWN",
            "request_id": "req-12345678",
            "corrected_fact": "실제는 다르다",
            "fingerprint": {"prompt_hash": "abc"},
        },
    )
    assert res.status_code == 201 and res.json() == {"feedback_id": 42}
    sql, params = pool.conn.calls[0]
    assert "INSERT INTO rag_answer_feedback" in sql
    assert params[3] == "DOWN" and json.loads(params[6]) == {"prompt_hash": "abc"}


@pytest.mark.parametrize(
    "body",
    [
        {"question": "q", "rating": "MAYBE"},
        {"question": "", "rating": "UP"},
        {"rating": "UP"},
    ],
)
def test_feedback_validation(monkeypatch, body):
    res = _client(monkeypatch, _Pool()).post("/ai/chat/feedback", json=body)
    assert res.status_code == 422


def test_feedback_store_failure_is_503_without_leaking_details(monkeypatch):
    class _Broken:
        def connection(self):
            raise RuntimeError("password=secret host=db")

    res = _client(monkeypatch, _Broken()).post(
        "/ai/chat/feedback", json={"question": "q", "rating": "UP"}
    )
    assert res.status_code == 503
    assert "secret" not in res.text


def test_negative_feedback_becomes_golden_candidates():
    report = build_candidates([])
    merged = merge_feedback(
        report,
        [
            {"question": "q1", "rating": "DOWN", "corrected_fact": "fact"},
            {"question": "q1", "rating": "DOWN"},
            {"question": "q2", "rating": "UP"},
        ],
    )
    assert [c["question"] for c in merged["candidates"]] == ["q1"]
    cand = merged["candidates"][0]
    assert cand["observed_count"] == 2 and cand["status"] == "needs_label"
    assert "negative_feedback" in cand["failure_modes"]
    assert cand["corrected_facts"] == ["fact"]
    assert merged["summary"]["by_failure_mode"]["negative_feedback"] == 2


def test_cli_merges_feedback_file(tmp_path):
    events = tmp_path / "e.json"
    events.write_text("[]", encoding="utf-8")
    fbk = tmp_path / "f.json"
    fbk.write_text(json.dumps([{"question": "q", "rating": "DOWN"}]), encoding="utf-8")
    out = tmp_path / "c.jsonl"
    assert (
        miner.main(
            [
                "--events-file",
                str(events),
                "--feedback-file",
                str(fbk),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert (
        json.loads(out.read_text(encoding="utf-8").splitlines()[0])["question"] == "q"
    )
