from types import SimpleNamespace

from app.core import fingerprint
from app.core.cache_provenance import build_cache_provenance, restore_cache_provenance

SETTINGS = SimpleNamespace(
    rag_rerank_enabled=True,
    rag_reranker_provider="http",
    rag_reranker_model="bge-m3",
    embed_provider="openrouter",
    embed_dim=1536,
    openrouter_model="vendor/model-x",
)


def test_fingerprint_has_every_required_field():
    fp = fingerprint.build_response_fingerprint(SETTINGS, {})
    for key in (
        "prompt_version",
        "prompt_hash",
        "planner_version",
        "retrieval_version",
        "reranker_version",
        "model",
        "embedding_signature",
    ):
        assert key in fp
    assert fp["reranker_version"] == "http:bge-m3"
    assert fp["model"] == "vendor/model-x"
    assert fp["embedding_signature"]
    assert fp["prompt_version"] == f"prompts-{fp['prompt_hash']}"


def test_model_prefers_recorded_usage():
    fp = fingerprint.build_response_fingerprint(
        SETTINGS, {"model_usage": [{"model": "used-model"}]}
    )
    assert fp["model"] == "used-model"


def test_reranker_version_when_disabled():
    off = SimpleNamespace(rag_rerank_enabled=False)
    assert fingerprint.build_response_fingerprint(off)["reranker_version"] == "none"


def test_prompt_hash_changes_when_a_prompt_changes(monkeypatch):
    from app.core import prompts

    before = fingerprint.prompt_hash()
    fingerprint.prompt_hash.cache_clear()
    monkeypatch.setattr(prompts, "SYSTEM_PROMPT", prompts.SYSTEM_PROMPT + " edited")
    try:
        assert fingerprint.prompt_hash() != before
    finally:
        fingerprint.prompt_hash.cache_clear()


def test_fingerprint_round_trips_through_cache_provenance():
    fp = fingerprint.build_response_fingerprint(SETTINGS, {})
    prov = build_cache_provenance({"verified": True}, fp)
    assert restore_cache_provenance(prov)["fingerprint"] == fp
    assert restore_cache_provenance(None)["fingerprint"] is None
