"""HNSW 인덱스 마이그레이션 관련 회귀 테스트.

테스트 범위:
- AI_VECTOR_INDEX 설정 파싱
- _detect_active_index()의 HNSW/IVFFlat 감지 로직
- _ensure_pgvector_session()의 모드별 GUC 분기
- schema.sql 에 HNSW 인덱스 정의 존재 확인
- create_vector_index.py 의 pgvector 버전 파싱 헬퍼
"""

from __future__ import annotations

import importlib
import unittest.mock as mock
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# config.py — AI_VECTOR_INDEX 설정
# ---------------------------------------------------------------------------


def test_ai_vector_index_default_is_auto(monkeypatch) -> None:
    """AI_VECTOR_INDEX 미설정 시 기본값은 'auto'."""
    monkeypatch.delenv("AI_VECTOR_INDEX", raising=False)
    # get_settings()는 lru_cache이므로 직접 Settings 인스턴스를 생성한다.
    from app.config import Settings

    s = Settings()
    assert s.ai_vector_index == "auto"


def test_ai_vector_index_env_override(monkeypatch) -> None:
    """AI_VECTOR_INDEX=hnsw 설정 시 해당 값이 반영된다."""
    monkeypatch.setenv("AI_VECTOR_INDEX", "hnsw")
    from app.config import Settings

    s = Settings()
    assert s.ai_vector_index == "hnsw"


def test_ai_vector_index_ivfflat(monkeypatch) -> None:
    """AI_VECTOR_INDEX=ivfflat 설정 시 해당 값이 반영된다."""
    monkeypatch.setenv("AI_VECTOR_INDEX", "ivfflat")
    from app.config import Settings

    s = Settings()
    assert s.ai_vector_index == "ivfflat"


def test_ai_vector_quantization_default_is_none(monkeypatch) -> None:
    """AI_VECTOR_QUANTIZATION 미설정 시 기본값은 'none'."""
    monkeypatch.delenv("AI_VECTOR_QUANTIZATION", raising=False)
    from app.config import Settings

    s = Settings()
    assert s.ai_vector_quantization == "none"


def test_ai_vector_quantization_halfvec(monkeypatch) -> None:
    """AI_VECTOR_QUANTIZATION=halfvec 설정 시 halfvec 검색 경로가 활성화된다."""
    monkeypatch.setenv("AI_VECTOR_QUANTIZATION", "halfvec")
    from app.config import Settings

    s = Settings()
    assert s.ai_vector_quantization == "halfvec"


# ---------------------------------------------------------------------------
# retrieval.py — _detect_active_index()
# ---------------------------------------------------------------------------


def _make_mock_conn(indexdefs: List[str]) -> MagicMock:
    """pg_indexes 조회 결과를 반환하는 psycopg 연결 mock."""
    rows = [(idef,) for idef in indexdefs]
    cur = MagicMock()
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = rows
    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn


def test_detect_active_index_returns_hnsw_when_hnsw_index_found() -> None:
    """pg_indexes에 hnsw 키워드가 포함된 indexdef가 있으면 'hnsw' 반환."""
    import app.core.retrieval as retrieval

    # 캐시 초기화
    retrieval._detected_vector_index = None
    conn = _make_mock_conn(
        [
            "CREATE INDEX idx_rag_chunks_embedding_hnsw ON rag_chunks USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='64')"
        ]
    )
    result = retrieval._detect_active_index(conn)
    assert result == "hnsw"
    # 캐시에 저장됐는지 확인
    assert retrieval._detected_vector_index == "hnsw"


def test_detect_active_index_falls_back_to_ivfflat_when_no_hnsw() -> None:
    """pg_indexes에 hnsw가 없으면 'ivfflat' 반환."""
    import app.core.retrieval as retrieval

    retrieval._detected_vector_index = None
    conn = _make_mock_conn(
        [
            "CREATE INDEX idx_rag_chunks_embedding ON rag_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists='644')"
        ]
    )
    result = retrieval._detect_active_index(conn)
    assert result == "ivfflat"
    assert retrieval._detected_vector_index == "ivfflat"


def test_detect_active_index_falls_back_to_ivfflat_on_empty_result() -> None:
    """pg_indexes 결과가 없으면 'ivfflat' 반환."""
    import app.core.retrieval as retrieval

    retrieval._detected_vector_index = None
    conn = _make_mock_conn([])
    result = retrieval._detect_active_index(conn)
    assert result == "ivfflat"


def test_detect_active_index_uses_cache_on_second_call() -> None:
    """캐시 히트 시 DB 조회를 생략한다."""
    import app.core.retrieval as retrieval

    retrieval._detected_vector_index = "hnsw"
    conn = MagicMock()  # cursor() 호출이 있으면 테스트 실패
    conn.cursor.side_effect = AssertionError("DB 조회가 발생하면 안 됩니다")
    result = retrieval._detect_active_index(conn)
    assert result == "hnsw"


# ---------------------------------------------------------------------------
# retrieval.py — _ensure_pgvector_session()
# ---------------------------------------------------------------------------


def _make_session_conn() -> tuple[MagicMock, list[str]]:
    """cursor.execute() 호출 내역을 기록하는 mock 연결을 반환."""
    executed: list[str] = []

    cur = MagicMock()
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    cur.execute.side_effect = lambda sql, *args, **kw: executed.append(sql)

    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn, executed


def test_ensure_pgvector_session_hnsw_mode_sets_ef_search(monkeypatch) -> None:
    """AI_VECTOR_INDEX=hnsw → hnsw.ef_search GUC 설정, ivfflat.probes 미설정."""
    monkeypatch.setenv("AI_VECTOR_INDEX", "hnsw")

    from app.config import Settings
    import app.core.retrieval as retrieval

    retrieval._detected_vector_index = None  # 캐시 초기화
    settings = Settings()
    conn, executed = _make_session_conn()
    retrieval._ensure_pgvector_session(conn, settings)

    joined = " | ".join(executed)
    assert "hnsw.ef_search" in joined
    assert "ivfflat.probes" not in joined


def test_ensure_pgvector_session_ivfflat_mode_sets_probes(monkeypatch) -> None:
    """AI_VECTOR_INDEX=ivfflat → ivfflat.probes GUC 설정, hnsw.ef_search 미설정."""
    monkeypatch.setenv("AI_VECTOR_INDEX", "ivfflat")

    from app.config import Settings
    import app.core.retrieval as retrieval

    retrieval._detected_vector_index = None
    settings = Settings()
    conn, executed = _make_session_conn()
    retrieval._ensure_pgvector_session(conn, settings)

    joined = " | ".join(executed)
    assert "ivfflat.probes" in joined
    assert "hnsw.ef_search" not in joined


def test_ensure_pgvector_session_auto_mode_delegates_to_detect(monkeypatch) -> None:
    """AI_VECTOR_INDEX=auto → _detect_active_index() 결과에 따라 분기."""
    monkeypatch.setenv("AI_VECTOR_INDEX", "auto")

    from app.config import Settings
    import app.core.retrieval as retrieval

    # 캐시를 강제로 "hnsw"로 설정 → auto 모드에서도 hnsw GUC 선택
    retrieval._detected_vector_index = "hnsw"
    settings = Settings()
    conn, executed = _make_session_conn()
    retrieval._ensure_pgvector_session(conn, settings)

    joined = " | ".join(executed)
    assert "hnsw.ef_search" in joined
    assert "ivfflat.probes" not in joined


def test_embedding_distance_sql_uses_halfvec_when_enabled(monkeypatch) -> None:
    """AI_VECTOR_QUANTIZATION=halfvec → halfvec distance expression."""
    monkeypatch.setenv("AI_VECTOR_QUANTIZATION", "halfvec")
    monkeypatch.setenv("EMBED_DIM", "256")

    from app.config import Settings
    import app.core.retrieval as retrieval

    settings = Settings()
    assert (
        retrieval._embedding_distance_sql(settings)
        == "embedding::halfvec(256) <=> %s::halfvec(256)"
    )
    assert (
        retrieval._embedding_distance_sql(settings, table_alias="r")
        == "r.embedding::halfvec(256) <=> %s::halfvec(256)"
    )


# ---------------------------------------------------------------------------
# schema.sql — HNSW 인덱스 정의 존재 확인
# ---------------------------------------------------------------------------


def test_schema_sql_has_hnsw_index_definition() -> None:
    """schema.sql에 halfvec HNSW 인덱스 정의가 포함되어 있어야 한다."""
    schema_path = Path(__file__).parent.parent / "app" / "db" / "schema.sql"
    content = schema_path.read_text(encoding="utf-8").lower()
    assert "hnsw" in content, "schema.sql에 HNSW 인덱스 정의가 없습니다."
    assert "idx_rag_chunks_embedding_halfvec_hnsw" in content
    assert "embedding::halfvec(256)" in content
    assert "where embedding is not null" in content


def test_schema_sql_does_not_auto_create_vector_hnsw() -> None:
    """halfvec 전환 후 schema.sql은 vector HNSW를 자동 생성하지 않는다."""
    schema_path = Path(__file__).parent.parent / "app" / "db" / "schema.sql"
    lines = schema_path.read_text(encoding="utf-8").splitlines()
    active_lines = [
        line.strip().lower()
        for line in lines
        if line.strip() and not line.strip().startswith("--")
    ]

    assert not any(
        "create index" in line
        and "idx_rag_chunks_embedding_hnsw" in line
        and "halfvec" not in line
        for line in active_lines
    )


def test_schema_sql_ivfflat_is_commented_out() -> None:
    """schema.sql의 IVFFlat 인덱스 생성 구문은 주석 처리되어 있어야 한다."""
    schema_path = Path(__file__).parent.parent / "app" / "db" / "schema.sql"
    lines = schema_path.read_text(encoding="utf-8").splitlines()
    # 주석 아닌 라인에 ivfflat + create index 가 동시에 있으면 안 됨
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("--"):
            continue  # 주석은 무시
        if "ivfflat" in stripped.lower() and "create index" in stripped.lower():
            raise AssertionError(
                f"IVFFlat CREATE INDEX 구문이 주석 처리되지 않았습니다:\n  {line}"
            )


# ---------------------------------------------------------------------------
# create_vector_index.py — _parse_version() 헬퍼
# ---------------------------------------------------------------------------


def test_parse_version_parses_standard_semver() -> None:
    """'0.7.4' → (0, 7, 4)."""
    from scripts.create_vector_index import _parse_version

    assert _parse_version("0.7.4") == (0, 7, 4)
    assert _parse_version("0.5.0") == (0, 5, 0)
    assert _parse_version("1.0.0") == (1, 0, 0)


def test_parse_version_comparison_against_min_version() -> None:
    """HNSW 최소 버전(0.5.0) 비교가 올바르게 동작한다."""
    from scripts.create_vector_index import _parse_version, _PGVECTOR_MIN_VERSION_HNSW

    assert _parse_version("0.5.0") >= _PGVECTOR_MIN_VERSION_HNSW
    assert _parse_version("0.7.4") >= _PGVECTOR_MIN_VERSION_HNSW
    assert _parse_version("0.4.4") < _PGVECTOR_MIN_VERSION_HNSW


def test_create_vector_index_builds_halfvec_hnsw_sql() -> None:
    from scripts.create_vector_index import _build_hnsw_create_sql

    sql = _build_hnsw_create_sql(
        quantization="halfvec",
        embed_dim=256,
        m=16,
        ef_construction=64,
    ).lower()

    assert "idx_rag_chunks_embedding_halfvec_hnsw" in sql
    assert "using hnsw ((embedding::halfvec(256)) halfvec_cosine_ops)" in sql
    assert "create index concurrently" in sql
    assert "where embedding is not null" in sql


def test_create_vector_index_builds_partial_vector_hnsw_sql() -> None:
    from scripts.create_vector_index import _build_hnsw_create_sql

    sql = _build_hnsw_create_sql(
        quantization="none",
        embed_dim=256,
        m=16,
        ef_construction=64,
    ).lower()

    assert "idx_rag_chunks_embedding_hnsw" in sql
    assert "using hnsw (embedding vector_cosine_ops)" in sql
    assert "where embedding is not null" in sql


def test_create_vector_index_builds_vector_hnsw_drop_sql() -> None:
    from scripts.create_vector_index import _build_vector_hnsw_drop_sql

    assert (
        _build_vector_hnsw_drop_sql()
        == "DROP INDEX CONCURRENTLY IF EXISTS idx_rag_chunks_embedding_hnsw"
    )


def test_drop_vector_hnsw_requires_halfvec_index_and_search_path() -> None:
    from scripts.create_vector_index import _vector_hnsw_drop_error

    assert _vector_hnsw_drop_error(
        quantization="none",
        embed_dim=256,
        distance_sql="embedding::halfvec(256) <=> %s::halfvec(256)",
        halfvec_index_exists=True,
    )
    assert _vector_hnsw_drop_error(
        quantization="halfvec",
        embed_dim=256,
        distance_sql="embedding::halfvec(256) <=> %s::halfvec(256)",
        halfvec_index_exists=False,
    )
    assert _vector_hnsw_drop_error(
        quantization="halfvec",
        embed_dim=256,
        distance_sql="embedding <=> %s::vector",
        halfvec_index_exists=True,
    )
    assert (
        _vector_hnsw_drop_error(
            quantization="halfvec",
            embed_dim=256,
            distance_sql="embedding::halfvec(256) <=> %s::halfvec(256)",
            halfvec_index_exists=True,
        )
        is None
    )


class _VerifyCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[str, ...] | None]] = []

    def execute(self, sql: str, params=None) -> None:
        self.executed.append((sql, params))

    def fetchall(self) -> list[tuple[str]]:
        return [
            ("Index Scan using idx_rag_chunks_embedding_halfvec_hnsw on rag_chunks",)
        ]


def test_verify_hnsw_usage_uses_halfvec_query_when_quantized() -> None:
    from scripts.create_vector_index import _verify_hnsw_usage

    cur = _VerifyCursor()

    assert _verify_hnsw_usage(cur, 256, 100, quantization="halfvec") is True

    explain_sql, params = cur.executed[1]
    assert "WHERE embedding IS NOT NULL" in explain_sql
    assert "embedding::halfvec(256) <=> %s::halfvec(256)" in explain_sql
    assert params == ("[" + ",".join("0.0" for _ in range(256)) + "]",)


def test_halfvec_requires_pgvector_0_7() -> None:
    from scripts.create_vector_index import (
        _PGVECTOR_MIN_VERSION_HALFVEC,
        _required_pgvector_version,
        _version_at_least,
    )

    assert _required_pgvector_version("halfvec") == _PGVECTOR_MIN_VERSION_HALFVEC
    assert _version_at_least((0, 7), _PGVECTOR_MIN_VERSION_HALFVEC)
    assert not _version_at_least((0, 6, 2), _PGVECTOR_MIN_VERSION_HALFVEC)
