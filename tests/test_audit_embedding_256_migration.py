from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from scripts import audit_embedding_256_migration as audit


class _FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append((query, params))


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self, *args, **kwargs) -> _FakeCursor:
        del args, kwargs
        return self._cursor


def _settings(**overrides: Any) -> SimpleNamespace:
    values = {
        "embed_provider": "openrouter",
        "embed_model": "text-embedding-3-small",
        "openrouter_embed_model": "openai/text-embedding-3-small",
        "openai_embed_model": "text-embedding-3-small",
        "embed_dim": 256,
        "rag_embedding_version": 2,
        "ai_vector_index": "hnsw",
        "ai_vector_quantization": "halfvec",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _db_audit(
    *,
    summary_overrides: dict[str, int] | None = None,
    indexes: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    summary = {
        "total": 10,
        "active": 10,
        "null_embeddings": 0,
        "active_null_embeddings": 0,
        "embedded": 10,
        "metadata_dim_not_256": 0,
        "active_metadata_dim_not_256": 0,
        "version_not_2": 0,
        "embedded_missing_model": 0,
        "metadata_fixable": 0,
        "active_metadata_fixable": 0,
        "metadata_conflicts": 0,
        "active_metadata_conflicts": 0,
    }
    summary.update(summary_overrides or {})
    return {
        "summary": summary,
        "column_type": {"embedding_type": "extensions.vector(256)"},
        "pgvector": {"extversion": "0.8.2"},
        "indexes": indexes
        if indexes is not None
        else [
            {
                "indexname": audit.HALFVEC_INDEX_NAME,
                "indexdef": "CREATE INDEX USING hnsw (((embedding)::extensions.halfvec(256)))",
            }
        ],
    }


def test_audit_clean_post_cleanup_state_passes_without_warning_or_fail() -> None:
    findings = audit._build_findings(
        settings=_settings(),
        defaults={},
        db_audit=_db_audit(),
        distance_sql="embedding::halfvec(256) <=> %s::halfvec(256)",
        index_dry_run={"exit_code": 0},
    )

    assert audit._overall_status(findings) == "pass"
    assert not [
        item
        for item in findings
        if item["severity"] in {"warning", "fail"}
    ]


def test_audit_findings_surface_safe_cleanup_classes() -> None:
    findings = audit._build_findings(
        settings=_settings(),
        defaults={},
        db_audit=_db_audit(
            summary_overrides={
                "metadata_fixable": 413,
                "active_null_embeddings": 2,
            }
        ),
        distance_sql="embedding::halfvec(256) <=> %s::halfvec(256)",
        index_dry_run={"exit_code": 0},
    )

    codes = {item["code"]: item["severity"] for item in findings}
    assert codes["metadata_fixable"] == "warning"
    assert codes["null_embeddings"] == "warning"


def test_audit_findings_fail_when_runtime_is_not_hnsw_halfvec() -> None:
    findings = audit._build_findings(
        settings=_settings(ai_vector_index="auto", ai_vector_quantization="none"),
        defaults={},
        db_audit=_db_audit(),
        distance_sql="embedding <=> %s::vector",
        index_dry_run={"exit_code": 0},
    )

    codes = {item["code"]: item["severity"] for item in findings}
    assert codes["runtime_vector_index"] == "fail"
    assert codes["runtime_vector_quantization"] == "fail"


def test_audit_findings_fail_on_unsafe_metadata_conflicts() -> None:
    findings = audit._build_findings(
        settings=_settings(),
        defaults={},
        db_audit=_db_audit(summary_overrides={"metadata_conflicts": 1}),
        distance_sql="embedding::halfvec(256) <=> %s::halfvec(256)",
        index_dry_run={"exit_code": 0},
    )

    codes = {item["code"]: item["severity"] for item in findings}
    assert codes["metadata_conflicts"] == "fail"


def test_audit_findings_warn_on_duplicate_vector_hnsw_index() -> None:
    findings = audit._build_findings(
        settings=_settings(),
        defaults={},
        db_audit=_db_audit(
            indexes=[
                {
                    "indexname": audit.HALFVEC_INDEX_NAME,
                    "indexdef": "CREATE INDEX USING hnsw (((embedding)::halfvec(256)))",
                },
                {
                    "indexname": audit.VECTOR_HNSW_INDEX_NAME,
                    "indexdef": "CREATE INDEX USING hnsw (embedding vector_cosine_ops)",
                },
            ]
        ),
        distance_sql="embedding::halfvec(256) <=> %s::halfvec(256)",
        index_dry_run={"exit_code": 0},
        index_cleanup_dry_run={"exit_code": 0},
    )

    codes = {item["code"]: item["severity"] for item in findings}
    assert codes["duplicate_hnsw_indexes"] == "warning"


def test_configure_audit_session_starts_read_only_transaction() -> None:
    cursor = _FakeCursor()

    audit._configure_audit_session(
        _FakeConnection(cursor),
        statement_timeout_ms=123,
    )

    assert cursor.executed[0] == ("SET TRANSACTION READ ONLY", ())
    assert cursor.executed[1] == (
        "SELECT set_config('statement_timeout', %s, false)",
        ("123",),
    )
