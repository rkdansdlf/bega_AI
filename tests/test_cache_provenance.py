from app.core.cache_provenance import (
    build_cache_provenance,
    is_cacheable_provenance,
    restore_cache_provenance,
)
from app.routers.chat_stream import _build_cached_completion_payload


def _meta(**over):
    base = {
        "verified": False,
        "data_sources": [{"id": "chunk-1"}, {"id": "chunk-2"}, {"id": "chunk-3"}],
        "answer_sources": ["chunk-1"],
        "as_of_date": "2025-09-01",
        "grounding_mode": "db_fast_path",
        "source_tier": "canonical_db",
    }
    base.update(over)
    return base


def test_unverified_answer_stays_unverified_on_hit():
    prov = build_cache_provenance(_meta(verified=False))
    payload = _build_cached_completion_payload(
        {"response_text": "답변", "provenance": prov}, cache_key="k" * 16
    )
    assert payload["verified"] is False


def test_sources_and_as_of_roundtrip():
    prov = build_cache_provenance(_meta(verified=True))
    payload = _build_cached_completion_payload(
        {"response_text": "답변", "provenance": prov},
        cache_key="k" * 16,
        semantic_cached=True,
    )
    assert payload["verified"] is True
    assert [s["id"] for s in payload["data_sources"]] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    assert payload["answer_sources"] == ["chunk-1"]
    assert payload["as_of_date"] == "2025-09-01"
    assert payload["origin_grounding_mode"] == "db_fast_path"
    assert payload["grounding_mode"] == "semantic_cache"


def test_legacy_row_without_provenance_is_unverified():
    payload = _build_cached_completion_payload(
        {"response_text": "답변"}, cache_key="k" * 16
    )
    assert payload["verified"] is False
    assert payload["fallback_reason"] == "cache_provenance_missing"


def test_json_string_provenance_is_parsed():
    import json

    prov = build_cache_provenance(_meta(verified=True))
    assert restore_cache_provenance(json.dumps(prov))["verified"] is True
    assert restore_cache_provenance("not json")["verified"] is False


def test_fallback_answers_are_not_cacheable():
    assert not is_cacheable_provenance(None)
    assert not is_cacheable_provenance(
        build_cache_provenance(_meta(fallback_answer_used=True))
    )
    assert is_cacheable_provenance(build_cache_provenance(_meta()))
