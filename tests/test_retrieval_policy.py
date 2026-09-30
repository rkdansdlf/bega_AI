import asyncio

import pytest

from app.core.retrieval_policy import (
    InvalidRetrievalFilter,
    annotate_relaxation,
    protected_filter_keys,
    resolve_filter_column,
)


@pytest.mark.parametrize("key", ["team_id", "season_year", "player_id", "source_table"])
def test_allowed_columns(key):
    assert resolve_filter_column(key) == (key, None)


def test_meta_json_key_maps_to_metadata():
    assert resolve_filter_column("meta.league") == ("metadata", "league")


@pytest.mark.parametrize(
    "key",
    [
        "team_id) OR TRUE --",
        "unknown_column",
        "meta.league') OR ('1'='1",
        "content_tsv",
        "evil.key",
        "meta.",
    ],
)
def test_rejects_unknown_or_injected_keys(key):
    with pytest.raises(InvalidRetrievalFilter):
        resolve_filter_column(key)


def test_similarity_search_rejects_bad_filter_before_query(monkeypatch):
    from app.core import retrieval

    async def _exists(*a, **k):
        return True

    monkeypatch.setattr(retrieval, "_rag_chunks_exists", _exists)

    class Conn:
        async def execute(self, *a, **k):  # pragma: no cover
            raise AssertionError("query must not run")

    with pytest.raises(InvalidRetrievalFilter):
        asyncio.run(
            retrieval.similarity_search(
                Conn(), [0.1] * 4, limit=3, filters={"team_id) OR TRUE --": "LG"}
            )
        )


def test_factual_query_protects_core_entities():
    f = {"season_year": 2025, "team_id": "KIA", "source_table": "x"}
    assert protected_filter_keys(f, intent="stats_lookup") == {"season_year", "team_id"}
    assert protected_filter_keys(f, intent="freeform") == {"season_year", "team_id"}


def test_explainer_and_regulation_may_relax():
    f = {"season_year": 2025, "team_id": "KIA"}
    assert protected_filter_keys(f, intent="knowledge_explanation") == frozenset()
    assert (
        protected_filter_keys(f, intent="freeform", is_regulation=True) == frozenset()
    )


def test_annotate_relaxation():
    assert annotate_relaxation(
        {"team_id": "A", "source_table": "t"}, {"team_id": "A"}
    ) == {
        "constraint_relaxed": True,
        "relaxed_fields": ["source_table"],
    }


def test_fallback_never_drops_team_or_season(monkeypatch):
    from app.core import retrieval

    seen = []

    async def fake_search(conn, emb, *, limit, filters=None, **kw):
        seen.append(dict(filters or {}))
        return []

    monkeypatch.setattr(retrieval, "similarity_search", fake_search)
    results, level = asyncio.run(
        retrieval.similarity_search_with_fallback(
            None,
            [0.1],
            limit=3,
            filters={"season_year": 2025, "team_id": "KIA", "source_table": "t"},
            intent="stats_lookup",
        )
    )
    assert results == []
    assert all(f.get("season_year") == 2025 and f.get("team_id") == "KIA" for f in seen)
    assert any("source_table" not in f for f in seen)
