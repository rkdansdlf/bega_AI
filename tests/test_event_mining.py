import json

from app.eval.event_mining import build_candidates, classify_event
from scripts import mine_retrieval_events as miner


def ev(**over):
    base = {
        "user_query": "2025 LG 성적",
        "intent": "stats_lookup",
        "metadata_filter": {"season_year": 2025, "team_id": "LG"},
        "retrieved_chunk_ids": [1, 2],
        "scores": [{"id": 1, "similarity": 0.8}],
        "success": True,
    }
    base.update(over)
    return base


def test_healthy_event_has_no_labels():
    assert classify_event(ev()) == []


def test_each_failure_mode_is_detected():
    assert classify_event(ev(success=False)) == ["error"]
    assert "zero_hit" in classify_event(ev(retrieved_chunk_ids=[]))
    assert "fallback" in classify_event(
        ev(
            metadata_filter={
                "fallback_used": True,
                "fallback_stage": "without_source_table",
            }
        )
    )
    assert "constraint_relaxed" in classify_event(
        ev(metadata_filter={"constraint_relaxed": True, "relaxed_fields": ["team_id"]})
    )
    assert "bad_source" in classify_event(
        ev(metadata_filter={"relevance_guard_dropped": 2})
    )
    assert "low_similarity" in classify_event(ev(scores=[{"similarity": 0.2}]))


def test_json_string_columns_are_parsed():
    e = ev(
        metadata_filter=json.dumps({"fallback_used": True}),
        retrieved_chunk_ids="[]",
        scores="[]",
    )
    assert {"zero_hit", "fallback"} <= set(classify_event(e))


def test_candidates_are_grouped_ranked_and_unlabeled():
    events = [
        ev(user_query="q-common", retrieved_chunk_ids=[]),
        ev(user_query="q-common", retrieved_chunk_ids=[]),
        ev(user_query="q-rare", retrieved_chunk_ids=[]),
        ev(user_query="healthy"),
    ]
    report = build_candidates(events)
    assert report["summary"]["events"] == 4
    assert report["summary"]["by_failure_mode"]["zero_hit"] == 3
    cands = report["candidates"]
    assert [c["question"] for c in cands] == ["q-common", "q-rare"]
    assert cands[0]["observed_count"] == 2
    assert cands[0]["status"] == "needs_label"
    assert cands[0]["relevant_doc_keys"] == []
    assert cands[0]["filters"] == {"season_year": 2025, "team_id": "LG"}
    assert (
        build_candidates(events, min_occurrences=2)["candidates"][0]["question"]
        == "q-common"
    )


def test_candidate_ids_are_stable():
    a = build_candidates([ev(retrieved_chunk_ids=[])])["candidates"][0]["id"]
    b = build_candidates([ev(retrieved_chunk_ids=[])])["candidates"][0]["id"]
    assert a == b and a.startswith("mined-")


def test_cli_offline_mode_writes_jsonl(tmp_path):
    events = tmp_path / "events.json"
    events.write_text(
        json.dumps([ev(user_query="q", retrieved_chunk_ids=[])]), encoding="utf-8"
    )
    out = tmp_path / "cand.jsonl"
    assert miner.main(["--events-file", str(events), "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["question"] == "q" and rows[0]["status"] == "needs_label"
