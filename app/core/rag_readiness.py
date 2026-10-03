"""Readiness aggregation for the one explicitly selected RAG backend."""

from __future__ import annotations

import inspect
from typing import Any

from .oracle_rag import (
    acquire_cursor,
    oracle_rag_readiness,
    resolve_oracle_generation,
)
from .retrieval_contract import (
    RetrievalContractViolation,
    missing_required_capabilities,
)
from .rag_runtime import (
    RagBackend,
    RagConfigurationError,
    RagSelection,
    classify_rag_dependency_error,
    resolve_rag_selection,
)


def _up(code: str, **details: object) -> dict[str, object]:
    return {"status": "UP", "code": code, **details}


def _down(code: str, **details: object) -> dict[str, object]:
    return {"status": "DOWN", "code": code, **details}


def _unavailable_storage_components() -> dict[str, dict[str, object]]:
    return {
        "rag_storage": _down("RAG_STORAGE_UNAVAILABLE"),
        "rag_schema": _down("RAG_STORAGE_UNAVAILABLE"),
        "rag_vector": _down("RAG_STORAGE_UNAVAILABLE"),
        "rag_index": _down("RAG_STORAGE_UNAVAILABLE"),
    }


def _classified_probe_failure(exception: Exception) -> dict[str, dict[str, object]]:
    code = classify_rag_dependency_error(exception)
    if code == "RAG_SCHEMA_NOT_READY":
        return {
            "rag_storage": _up("RAG_STORAGE_READY"),
            "rag_schema": _down(code),
            "rag_vector": _down(code),
            "rag_index": _down(code),
        }
    if code in {
        "RAG_VECTOR_CAPABILITY_NOT_READY",
        "RAG_EMBEDDING_DIMENSION_MISMATCH",
    }:
        return {
            "rag_storage": _up("RAG_STORAGE_READY"),
            "rag_schema": _up("RAG_SCHEMA_READY"),
            "rag_vector": _down(code),
            "rag_index": _down(code),
        }
    return _unavailable_storage_components()


def _embedding_component(settings: Any) -> dict[str, object]:
    provider = str(getattr(settings, "embed_provider", "") or "").strip().lower()
    dimension = max(1, int(getattr(settings, "embed_dim", 0) or 0))
    configured = False
    if provider == "local":
        configured = True
    elif provider == "hf":
        configured = bool(
            getattr(settings, "embed_model", None)
            or getattr(settings, "hf_embed_model", None)
        )
    elif provider == "openai":
        configured = bool(
            getattr(settings, "openai_api_key", None)
            and (
                getattr(settings, "openai_embed_model", None)
                or getattr(settings, "embed_model", None)
            )
        )
    elif provider == "openrouter":
        configured = bool(
            getattr(settings, "openrouter_api_key", None)
            and (
                getattr(settings, "openrouter_embed_model", None)
                or getattr(settings, "embed_model", None)
            )
        )
    elif provider == "gemini":
        configured = bool(
            getattr(settings, "gemini_api_key", None)
            and (
                getattr(settings, "gemini_embed_model", None)
                or getattr(settings, "embed_model", None)
            )
        )
    details = {"provider": provider or "unknown", "dimension": dimension}
    return (
        _up("RAG_EMBEDDING_READY", **details)
        if configured
        else _down("RAG_EMBEDDING_NOT_READY", **details)
    )


async def _close_cursor(cursor: Any) -> None:
    close = getattr(cursor, "close", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result


async def _probe_physical_generation(conn: Any) -> dict[str, object]:
    """Live state of the ACTIVE generation for the physical embedding store.

    Reads the index from pg_index (not the registry's ``index_ready`` flag) so a
    dropped or invalidated index is caught even if the registry is stale.
    """
    try:
        cursor = await conn.execute("""
            SELECT g.generation_id, g.embedding_dim,
                   EXISTS (
                       SELECT 1 FROM pg_index i
                       JOIN pg_class c ON c.oid = i.indexrelid
                       WHERE c.relname = 'idx_rag_emb_'
                             || replace(g.generation_id, '-', '_')
                         AND i.indisvalid AND i.indisready
                   ) AS index_live,
                   EXISTS (
                       SELECT 1 FROM rag_chunk_embeddings e
                       WHERE e.generation_id = g.generation_id LIMIT 1
                   ) AS has_rows
            FROM rag_embedding_generations g
            WHERE g.status = 'ACTIVE'
            """)
    except Exception:  # noqa: BLE001 - migrations 007/009 not applied
        return {"error": "registry_unavailable"}
    row = await cursor.fetchone()
    if row is None:
        return {"error": "no_active_generation"}
    return {
        "generation_id": str(row[0]),
        "dimension": int(row[1]),
        "index_live": bool(row[2]),
        "has_rows": bool(row[3]),
    }


def _physical_components(
    table_ready: bool, extension_ready: bool, physical: dict[str, object]
) -> dict[str, dict[str, object]]:
    """Readiness for ``RAG_EMBEDDING_STORE=generations`` (fail closed)."""
    base_ok = table_ready and extension_ready
    error = physical.get("error")
    if error:
        code = (
            "RAG_GENERATION_NOT_ACTIVE"
            if error == "no_active_generation"
            else "RAG_GENERATION_REGISTRY_UNAVAILABLE"
        )
        return {
            "rag_storage": _up("RAG_STORAGE_READY"),
            "rag_schema": (
                _up("RAG_SCHEMA_READY")
                if table_ready
                else _down("RAG_SCHEMA_NOT_READY")
            ),
            "rag_vector": _down(code),
            "rag_index": _down(code),
        }
    dim = int(physical["dimension"])
    return {
        "rag_storage": _up("RAG_STORAGE_READY"),
        "rag_schema": (
            _up("RAG_SCHEMA_READY") if table_ready else _down("RAG_SCHEMA_NOT_READY")
        ),
        "rag_vector": (
            _up(
                "RAG_VECTOR_READY",
                dimension=dim,
                generation=physical["generation_id"],
            )
            if base_ok and physical["has_rows"]
            else _down("RAG_VECTOR_CAPABILITY_NOT_READY", dimension=dim)
        ),
        "rag_index": (
            _up("RAG_VECTOR_INDEX_READY", generation=physical["generation_id"])
            if physical["index_live"]
            else _down("RAG_VECTOR_INDEX_NOT_READY")
        ),
    }


async def _probe_postgres(pool: Any, settings: Any) -> dict[str, dict[str, object]]:
    expected_index = (
        "idx_rag_chunks_embedding_halfvec_hnsw"
        if str(getattr(settings, "ai_vector_quantization", "")).lower() == "halfvec"
        else "idx_rag_chunks_embedding_hnsw"
    )
    async with pool.connection(timeout=1.5) as conn:
        await conn.execute("SELECT 1")
        cursor = await conn.execute(
            """
            SELECT
                EXISTS (
                    SELECT 1
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relname = 'rag_chunks'
                      AND c.relkind IN ('r', 'p')
                      AND n.nspname = ANY(current_schemas(false))
                ) AS table_ready,
                EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                ) AS vector_ready,
                COALESCE((
                    SELECT format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relname = 'rag_chunks'
                      AND n.nspname = ANY(current_schemas(false))
                      AND a.attname = 'embedding'
                      AND NOT a.attisdropped
                    LIMIT 1
                ), '') AS embedding_type,
                EXISTS (
                    SELECT 1
                    FROM pg_index i
                    JOIN pg_class index_class ON index_class.oid = i.indexrelid
                    JOIN pg_am am ON am.oid = index_class.relam
                    WHERE index_class.relname = %s
                      AND i.indisvalid
                      AND i.indisready
                      AND am.amname = 'hnsw'
                ) AS index_ready
            """,
            (expected_index,),
        )
        row = await cursor.fetchone()
        physical = None
        if str(getattr(settings, "rag_embedding_store", "inline")) == "generations":
            physical = await _probe_physical_generation(conn)

    table_ready = bool(row and row[0])
    extension_ready = bool(row and row[1])
    embedding_type = str(row[2] if row and len(row) > 2 else "").lower()
    index_ready = bool(row and len(row) > 3 and row[3])
    expected_type = f"vector({max(1, int(settings.embed_dim))})"
    dimension_ready = embedding_type == expected_type
    if physical is not None:
        return _physical_components(table_ready, extension_ready, physical)
    return {
        "rag_storage": _up("RAG_STORAGE_READY"),
        "rag_schema": (
            _up("RAG_SCHEMA_READY") if table_ready else _down("RAG_SCHEMA_NOT_READY")
        ),
        "rag_vector": (
            _up("RAG_VECTOR_READY", dimension=int(settings.embed_dim))
            if table_ready and extension_ready and dimension_ready
            else _down(
                (
                    "RAG_EMBEDDING_DIMENSION_MISMATCH"
                    if table_ready and extension_ready and embedding_type
                    else "RAG_VECTOR_CAPABILITY_NOT_READY"
                ),
                dimension=int(settings.embed_dim),
            )
        ),
        "rag_index": (
            _up("RAG_VECTOR_INDEX_READY")
            if index_ready
            else _down("RAG_VECTOR_INDEX_NOT_READY")
        ),
    }


async def _probe_oracle(pool: Any, settings: Any) -> dict[str, dict[str, object]]:
    try:
        active_index_version = resolve_oracle_generation(settings)
    except RetrievalContractViolation:
        # Gate enabled but no active generation configured: serving would mix
        # generations, so the backend is not ready (fail closed).
        return {
            "rag_storage": _up("RAG_STORAGE_READY"),
            "rag_schema": _down(
                "RETRIEVAL_CONTRACT_NOT_MET", reason="no_active_generation"
            ),
            "rag_vector": _down("RETRIEVAL_CONTRACT_NOT_MET"),
            "rag_index": _down("RETRIEVAL_CONTRACT_NOT_MET"),
        }
    async with pool.connection(timeout=1.5) as conn:
        cursor = await acquire_cursor(conn)
        try:
            await cursor.execute("SELECT 1 FROM dual")
            await cursor.fetchone()
        finally:
            await _close_cursor(cursor)
        oracle = await oracle_rag_readiness(
            conn,
            expected_dim=max(1, int(settings.embed_dim)),
            active_index_version=active_index_version,
        )
    contract_missing = missing_required_capabilities(oracle.get("contract") or {})
    if contract_missing or not oracle.get("generation_ok", True):
        return {
            "rag_storage": _up("RAG_STORAGE_READY"),
            "rag_schema": _down(
                "RETRIEVAL_CONTRACT_NOT_MET",
                missing_capabilities=contract_missing,
                generation_ok=bool(oracle.get("generation_ok", True)),
            ),
            "rag_vector": _down("RETRIEVAL_CONTRACT_NOT_MET"),
            "rag_index": _down("RETRIEVAL_CONTRACT_NOT_MET"),
        }
    vectors = int(oracle.get("vector_rows") or 0)
    matching = int(oracle.get("matching_dim_rows") or 0)
    missing = int(oracle.get("missing_rows") or 0)
    vector_ready = matching == vectors and missing == 0
    return {
        "rag_storage": _up("RAG_STORAGE_READY"),
        "rag_schema": _up("RAG_SCHEMA_READY"),
        "rag_vector": (
            _up("RAG_VECTOR_READY", dimension=int(settings.embed_dim))
            if vector_ready
            else _down(
                "RAG_EMBEDDING_DIMENSION_MISMATCH",
                dimension=int(settings.embed_dim),
            )
        ),
        "rag_index": (
            _up("RAG_VECTOR_INDEX_READY")
            if bool(oracle.get("index_valid"))
            else _down("RAG_VECTOR_INDEX_NOT_READY")
        ),
    }


async def _probe_selected_backend(
    pool: Any,
    settings: Any,
    selection: RagSelection,
) -> dict[str, dict[str, object]]:
    if selection.backend is RagBackend.FAKE:
        return {
            "rag_storage": _up("RAG_STORAGE_READY"),
            "rag_schema": _up("RAG_SCHEMA_READY"),
            "rag_vector": _up("RAG_VECTOR_READY", dimension=int(settings.embed_dim)),
            "rag_index": _up("RAG_VECTOR_INDEX_READY"),
        }
    if selection.backend is RagBackend.ORACLE:
        return await _probe_oracle(pool, settings)
    return await _probe_postgres(pool, settings)


async def build_rag_readiness_report(
    settings: Any,
    pool: Any,
) -> dict[str, Any]:
    """Return a secret-safe report for only the selected RAG backend."""

    selection = resolve_rag_selection(settings)
    actual_backend = str(getattr(pool, "backend", "") or "").lower()
    components: dict[str, dict[str, object]] = {
        "rag_backend": {
            "status": "UP",
            "backend": selection.backend.value,
            "profile": selection.profile.value,
        },
        "embedding": _embedding_component(settings),
    }
    if actual_backend != selection.backend.value:
        components["rag_backend"] = _down(
            "RAG_BACKEND_IDENTITY_MISMATCH",
            backend=selection.backend.value,
            profile=selection.profile.value,
        )
        components.update(_unavailable_storage_components())
    else:
        try:
            components.update(await _probe_selected_backend(pool, settings, selection))
        except Exception as exc:  # noqa: BLE001 - return a safe 503 report.
            components.update(_classified_probe_failure(exc))

    ready = all(component.get("status") == "UP" for component in components.values())
    return {"status": "UP" if ready else "NOT_READY", "components": components}


def build_rag_configuration_error_report(
    settings: Any,
    error: RagConfigurationError,
) -> dict[str, Any]:
    """Return the same wire shape when selection fails before pool creation."""

    components = {
        "rag_backend": _down(error.code),
        "embedding": _embedding_component(settings),
        **_unavailable_storage_components(),
    }
    return {"status": "NOT_READY", "components": components}
