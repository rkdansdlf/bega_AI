"""Oracle-backed RAG storage and dense retrieval boundary."""

from __future__ import annotations

import array
import asyncio
import inspect
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Mapping, Sequence
from urllib.parse import parse_qs, unquote, urlsplit

from ..config import DEFAULT_EMBED_DIM
from .exceptions import DBRetrievalError

try:
    import oracledb
except ModuleNotFoundError:  # pragma: no cover
    oracledb = None  # type: ignore[assignment]
else:
    # Return CLOB/JSON columns as inline str instead of LOB locators. Lazy LOB
    # reads cost one network round trip per row (~16s for a 24-row dense hit
    # against ADB); inline strings measured ~110ms for the same query.
    # lob_to_text() stays as a passthrough safety net.
    oracledb.defaults.fetch_lobs = False

from .retrieval_contract import (
    RetrievalContractViolation,
    backend_capabilities,
    enforce_result_contract,
    entity_constraints,
    normalize_result,
    resolve_backend_filters,
)

logger = logging.getLogger(__name__)
_ORACLE_SCHEMES = ("oracle+oracledb://", "oracle://")
_RETRIEVABLE_STATUSES = ("ACTIVE", "INDEXED")
_ORACLE_RRF_K_BY_INTENT = {
    "player_profile": 30,
    "stats_lookup": 40,
    "comparison": 50,
}
_SEARCH_TOKEN_SUFFIXES = (
    "으로",
    "에서",
    "에게",
    "께서",
    "의",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "와",
    "과",
    "도",
    "에",
)
_RAG_SOURCE_TABLES = {
    "source_table",
    "team_id",
    "season_year",
    "league_type_code",
    "player_id",
    "index_version",
}
# Columns present on Oracle rag_chunks that the shared filter policy can target.
_ORACLE_FILTER_COLUMNS = frozenset(
    {"source_table", "team_id", "season_year", "league_type_code", "player_id"}
)
# Oracle-only keys: sparse-postings date narrowing and the index version.
_ORACLE_EXTRA_FILTER_KEYS = frozenset({"game_date", "index_version"})
_ENTITY_SELECT = "season_year, team_id, player_id"


_AUTO: Any = object()


def _generation_or_default(value: Any) -> str | None:
    """Explicit value wins; the ``_AUTO`` default reads the process settings so
    every caller (RAG, tools) gets the gate without extra plumbing."""
    if value is _AUTO:
        from ..config import get_settings

        return resolve_oracle_generation(get_settings())
    return value


def resolve_oracle_generation(settings: Any) -> str | None:
    """Active ``index_version`` when the generation gate is on, else ``None``.

    Fails closed: the gate enabled without a configured active version would
    silently serve every generation, which is exactly what the gate prevents.
    """
    if not bool(getattr(settings, "rag_generation_gate_enabled", False)):
        return None
    version = str(
        getattr(settings, "rag_oracle_active_index_version", "") or ""
    ).strip()
    if not version:
        raise RetrievalContractViolation(
            "RAG_GENERATION_GATE_ENABLED requires RAG_ORACLE_ACTIVE_INDEX_VERSION "
            "for the Oracle backend"
        )
    return version


def oracle_capabilities(active_index_version: str | None) -> dict[str, Any]:
    """What the Oracle retriever can enforce (see ``retrieval_contract``)."""
    return backend_capabilities(
        backend="oracle",
        supported_columns=_ORACLE_FILTER_COLUMNS,
        # The gate is enforceable (index_version); "on" is a config choice.
        generation_gate=True,
        # No valid_from/valid_to/expires_at columns: lifecycle is index_status.
        temporal_filters=False,
    )


def is_oracle_rag_url(value: str | None) -> bool:
    """Return whether a RAG URL uses the python-oracledb driver."""
    return bool(value and value.lower().startswith(_ORACLE_SCHEMES))


class OracleRagConnection:
    """Mark and delegate an async Oracle connection."""

    backend = "oracle"

    def __init__(self, raw_connection: Any) -> None:
        self.raw_connection = raw_connection

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw_connection, name)


def is_oracle_rag_connection(connection: Any) -> bool:
    """Return whether a connection came from the Oracle RAG pool."""
    return getattr(connection, "backend", None) == "oracle"


async def acquire_cursor(connection: Any) -> Any:
    """Get an Oracle cursor; python-oracledb 4.x returns it synchronously."""
    cursor = connection.cursor()
    if inspect.isawaitable(cursor):
        return await cursor
    return cursor


def _oracle_connect_args(conninfo: str) -> dict[str, Any]:
    """Translate an SQLAlchemy-style Oracle URL into oracledb arguments."""
    normalized = conninfo
    for scheme in _ORACLE_SCHEMES:
        if normalized.lower().startswith(scheme):
            normalized = "oracle://" + normalized[len(scheme) :]
            break
    parsed = urlsplit(normalized)
    if not parsed.hostname:
        raise ValueError("AI_RAG_DB_URL must include an Oracle DSN")
    dsn = unquote(parsed.hostname)
    if parsed.port:
        dsn = f"{dsn}:{parsed.port}"
    path = unquote(parsed.path or "").strip("/")
    service_name = (parse_qs(parsed.query).get("service_name") or [None])[0]
    if path:
        dsn = f"{dsn}/{path}"
    elif service_name:
        dsn = f"{dsn}/{unquote(service_name)}"
    args: dict[str, Any] = {
        "dsn": dsn,
        # Fall back to the crawler-side env contract when the URL omits
        # credentials (same behavior as src/db/engine.py).
        "user": unquote(parsed.username or "") or os.getenv("ORACLE_APP_USER", ""),
        "password": unquote(parsed.password or "")
        or os.getenv("ORACLE_APP_PASSWORD", ""),
    }
    tns_admin = os.getenv("TNS_ADMIN")
    if tns_admin:
        # Thin mode needs both: config_dir for tnsnames.ora resolution and
        # wallet_location for the TLS identity (cwallet.sso).
        args["config_dir"] = tns_admin
        args["wallet_location"] = tns_admin
        # Thin mode needs the password chosen when the wallet was downloaded.
        # It is independent from the database user's password; never guess it
        # from the credential-bearing connection URL.
        wallet_password = os.getenv("OCI_WALLET_PASSWORD") or None
        if wallet_password:
            args["wallet_password"] = wallet_password
    return args


class OracleRagPool:
    """Lazy async connection pool for the Oracle RAG database."""

    backend = "oracle"

    def __init__(self, conninfo: str, *, max_size: int = 8) -> None:
        self.conninfo = conninfo
        self.max_size = max(1, int(max_size))
        self._pool: Any = None
        self._open_lock = asyncio.Lock()

    async def open(self, *, wait: bool = True, timeout: float | None = None) -> None:
        """Open the pool using the same lifecycle shape as psycopg pools."""
        del wait
        if self._pool is not None:
            return
        if oracledb is None:
            raise RuntimeError("oracledb is required when AI_RAG_DB_URL uses Oracle")
        async with self._open_lock:
            if self._pool is not None:
                return
            # oracledb 4.x returns the AsyncConnectionPool synchronously; older
            # versions returned a coroutine. Support both shapes.
            created = oracledb.create_pool_async(
                **_oracle_connect_args(self.conninfo),
                min=1,
                max=self.max_size,
                increment=1,
            )
            if inspect.isawaitable(created):
                self._pool = (
                    await asyncio.wait_for(created, timeout=timeout)
                    if timeout is not None
                    else await created
                )
            else:
                del timeout
                self._pool = created

    @asynccontextmanager
    async def connection(
        self, timeout: float | None = None
    ) -> AsyncIterator[OracleRagConnection]:
        """Borrow one Oracle connection and return it to the pool."""
        if self._pool is None:
            await self.open(timeout=timeout)
        raw_connection = await self._pool.acquire()
        try:
            yield OracleRagConnection(raw_connection)
        finally:
            result = self._pool.release(raw_connection)
            if inspect.isawaitable(result):
                await result

    async def close(self) -> None:
        """Close the Oracle pool if it was opened."""
        pool = self._pool
        self._pool = None
        if pool is not None:
            result = pool.close()
            if inspect.isawaitable(result):
                await result

    def get_stats(self) -> dict[str, Any]:
        """Expose compatible pool metrics without connection details."""
        pool = self._pool
        if pool is None:
            return {"pool_open": False}
        get_stats = getattr(pool, "get_stats", None)
        return get_stats() if callable(get_stats) else {"pool_open": True}


async def lob_to_text(value: Any) -> Any:
    """Read sync or async Oracle LOBs into plain Python text."""
    if value is None or not hasattr(value, "read"):
        return value
    data = value.read()
    if inspect.isawaitable(data):
        data = await data
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


async def _parse_meta_async(value: Any) -> dict[str, Any]:
    """Decode Oracle JSON/CLOB metadata into the shared result shape."""
    return _parse_meta(await lob_to_text(value))


def _row_mapping(cursor: Any, row: Any) -> dict[str, Any]:
    """Convert an Oracle tuple row into lowercase column names."""
    if isinstance(row, Mapping):
        return {str(key).lower(): value for key, value in row.items()}

    names = []
    for item in cursor.description or ():
        name = getattr(item, "name", None)
        names.append(str(name if name is not None else item[0]).lower())
    return dict(zip(names, row, strict=False))


def _parse_meta(value: Any) -> dict[str, Any]:
    """Decode Oracle JSON/CLOB metadata into the shared result shape."""
    if value is None:
        return {}
    if hasattr(value, "read"):
        value = value.read()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _bind_filter_clauses(
    filters: Mapping[str, Any] | None,
    *,
    table_alias: str | None = None,
    require_embedding: bool = True,
    active_index_version: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Build parameterized Oracle predicates from shared RAG filters.

    Unknown or unenforceable keys raise (shared allowlist) instead of being
    dropped: a silently ignored filter widens the search scope.
    """
    prefix = f"{table_alias}." if table_alias else ""
    clauses = [f"{prefix}index_status IN ('ACTIVE', 'INDEXED')"]
    if require_embedding:
        clauses.insert(0, f"{prefix}embedding_vector IS NOT NULL")
    params: dict[str, Any] = {}
    if active_index_version:
        clauses.append(f"{prefix}index_version = :active_index_version")
        params["active_index_version"] = active_index_version
    if not filters:
        return clauses, params
    resolved = resolve_backend_filters(
        filters,
        backend="oracle",
        supported_columns=_ORACLE_FILTER_COLUMNS,
        extra_keys=_ORACLE_EXTRA_FILTER_KEYS,
    )
    excluded = resolved.get("_exclude_source_tables", ())
    if isinstance(excluded, str):
        excluded = (excluded,)
    for index, value in enumerate(excluded or ()):
        key = f"excluded_source_{index}"
        clauses.append(f"{prefix}source_table <> :{key}")
        params[key] = str(value)
    included = resolved.get("source_table_in", ())
    if isinstance(included, str):
        included = (included,)
    include_keys = []
    for index, value in enumerate(included or ()):
        key = f"included_source_{index}"
        include_keys.append(f":{key}")
        params[key] = str(value)
    if include_keys:
        clauses.append(f"{prefix}source_table IN ({', '.join(include_keys)})")
    for key, value in resolved.items():
        if (
            value is None
            or key.startswith("_")
            or key
            in (
                "source_table_in",
                "game_date",
            )
        ):
            continue
        if key.startswith("meta."):
            # Key already validated against ^[A-Za-z0-9_]{1,64}$ (no quoting
            # tricks possible), so it is safe as a JSON path literal.
            bind_key = f"meta_{len(params)}"
            clauses.append(f"JSON_VALUE({prefix}meta, '$.{key[5:]}') = :{bind_key}")
            params[bind_key] = str(value)
            continue
        if key == "index_version" and active_index_version:
            # An explicit filter may not widen past the active generation.
            if str(value) != active_index_version:
                raise RetrievalContractViolation(
                    f"index_version filter {value!r} conflicts with the active "
                    f"generation {active_index_version!r}"
                )
            continue
        bind_key = f"filter_{key}"
        clauses.append(f"{prefix}{key} = :{bind_key}")
        params[bind_key] = value
    return clauses, params


def _search_tokens(keyword: str | None) -> list[str]:
    """Normalize query terms using the Oracle sparse-index contract."""
    if not keyword:
        return []
    tokens: list[str] = []
    for raw_token in keyword.split():
        token = _normalize_search_token(raw_token)
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def _normalize_search_token(raw_token: str) -> str | None:
    """Normalize one sparse token and remove one Korean particle."""
    token = raw_token.strip().casefold()
    for suffix in _SEARCH_TOKEN_SUFFIXES:
        if len(token) > len(suffix) + 1 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    if 1 < len(token) <= 128:
        return token
    return None


def _sparse_term_rows(
    *,
    chunk_id: int,
    source_table: str,
    title: str,
    content: str,
    meta: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build Oracle posting rows with the canonical title boost counts."""
    title_counts: dict[str, int] = {}
    content_counts: dict[str, int] = {}
    for raw_token in re.findall(r"[^\W_]+", title or "", re.UNICODE):
        token = _normalize_search_token(raw_token)
        if token:
            title_counts[token] = title_counts.get(token, 0) + 1
    for raw_token in re.findall(r"[^\W_]+", content or "", re.UNICODE):
        token = _normalize_search_token(raw_token)
        if token:
            content_counts[token] = content_counts.get(token, 0) + 1
    game_date = meta.get("game_date")
    return [
        {
            "rag_chunk_id": chunk_id,
            "source_table": source_table,
            "token": token,
            "term_count": title_counts.get(token, 0) + content_counts.get(token, 0),
            "title_count": title_counts.get(token, 0),
            "game_date": str(game_date) if game_date is not None else None,
        }
        for token in sorted(title_counts.keys() | content_counts.keys())
    ]


async def _oracle_dense_similarity_search(
    connection: OracleRagConnection,
    embedding: Sequence[float],
    *,
    result_limit: int,
    filters: Mapping[str, Any] | None = None,
    document_type: str | None = None,
    game_date: str | None = None,
    active_index_version: str | None = None,
) -> list[dict[str, Any]]:
    """Search Oracle native VECTOR rows by cosine distance."""
    clauses, params = _bind_filter_clauses(
        filters, active_index_version=active_index_version
    )
    candidate_limit = max(int(result_limit), 1)
    params["query_vector"] = array.array("f", [float(value) for value in embedding])
    # APPROX FIRST lets Oracle answer via the HNSW index; an exact
    # `FETCH FIRST ... ORDER BY distance` would full-scan every VECTOR row.
    sql = f"""
        SELECT id, title, content, source_table, source_row_id, meta,
               content_hash, index_version, index_status, indexed_at, updated_at,
               season_year, team_id, player_id,
               VECTOR_DISTANCE(embedding_vector, :query_vector, COSINE) AS distance
        FROM rag_chunks
        WHERE {" AND ".join(clauses)}
        ORDER BY distance ASC
        FETCH APPROX FIRST {candidate_limit} ROWS ONLY
    """
    try:
        cursor = await acquire_cursor(connection)
        try:
            await cursor.execute(sql, params)
            rendered: list[dict[str, Any]] = []
            for raw_row in await cursor.fetchall():
                row = _row_mapping(cursor, raw_row)
                meta = await _parse_meta_async(row.get("meta"))
                actual_document_type = meta.get("document_type") or meta.get("category")
                actual_game_date = meta.get("game_date")
                if document_type and actual_document_type != document_type:
                    continue
                if game_date and str(actual_game_date) != str(game_date):
                    continue
                similarity = 1.0 - float(row.get("distance") or 0.0)
                rendered.append(
                    {
                        "id": row.get("id"),
                        "title": await lob_to_text(row.get("title")),
                        "content": await lob_to_text(row.get("content")),
                        "source_table": row.get("source_table"),
                        "source_row_id": row.get("source_row_id"),
                        "meta": meta,
                        "metadata": meta,
                        "source_type": meta.get("source_type"),
                        "source_uri": meta.get("source_url"),
                        "topic_key": meta.get("topic_key"),
                        "content_hash": row.get("content_hash"),
                        "updated_at": row.get("updated_at"),
                        "season_year": row.get("season_year"),
                        "team_id": row.get("team_id"),
                        "player_id": row.get("player_id"),
                        "index_version": row.get("index_version"),
                        "similarity": similarity,
                        "keyword_rank_val": 0.0,
                        "combined_score": similarity,
                    }
                )
                if len(rendered) >= result_limit:
                    break
            return rendered
        finally:
            close = getattr(cursor, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
    except Exception as exc:
        logger.exception("Oracle VECTOR retrieval failed")
        raise DBRetrievalError("Oracle VECTOR query failed", cause=exc) from exc


async def _oracle_sparse_search(
    connection: OracleRagConnection,
    keyword: str,
    *,
    candidate_limit: int,
    filters: Mapping[str, Any] | None = None,
    active_index_version: str | None = None,
) -> list[dict[str, Any]]:
    """Return bounded sparse candidates from Oracle term postings.

    Runs one index-driven top-N query per token (the proven terms-canary
    shape) and merges scores in Python. A single grouped JOIN over all
    tokens measured ~1.8s against ADB versus ~0.5s for the per-token path.
    """
    tokens = _search_tokens(keyword)
    if not tokens:
        return []
    per_token_limit = max(int(candidate_limit) // len(tokens), 100)
    posting_clauses = ["t.token = :sparse_token"]
    posting_params: dict[str, Any] = {}
    if filters and filters.get("source_table"):
        posting_params["sparse_source"] = str(filters["source_table"])
        posting_clauses.append("t.source_table = :sparse_source")
    if filters and filters.get("game_date") is not None:
        posting_params["sparse_game_date"] = str(filters["game_date"])
        posting_clauses.append("t.game_date = :sparse_game_date")
    posting_sql = f"""
                SELECT t.rag_chunk_id,
                       t.term_count + 2 * t.title_count AS sparse_score
                FROM rag_chunk_terms t
                WHERE {" AND ".join(posting_clauses)}
                ORDER BY sparse_score DESC, t.rag_chunk_id
                FETCH FIRST {per_token_limit} ROWS ONLY
            """
    scores: dict[int, float] = {}
    cursor = await acquire_cursor(connection)
    try:
        for token in tokens:
            posting_params["sparse_token"] = token
            await cursor.execute(posting_sql, dict(posting_params))
            for raw_row in await cursor.fetchall():
                chunk_id = int(raw_row[0])
                scores[chunk_id] = scores.get(chunk_id, 0.0) + float(raw_row[1] or 0.0)
        if not scores:
            return []
        merged_ids = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[
            : max(int(candidate_limit), 1)
        ]
        identity_params = {
            f"sparse_id_{index}": chunk_id
            for index, (chunk_id, _score) in enumerate(merged_ids)
        }
        generation_sql = ""
        if active_index_version:
            generation_sql = " AND index_version = :active_index_version"
        # Narrow identity fetch: no CLOBs here; full hydration happens later
        # for just the fused survivors.
        await cursor.execute(
            f"""
                SELECT id, source_table, source_row_id
                FROM rag_chunks
                WHERE id IN ({", ".join(f":{key}" for key in identity_params)})
                  AND index_status IN ('ACTIVE', 'INDEXED'){generation_sql}
            """,
            (
                {**identity_params, "active_index_version": active_index_version}
                if active_index_version
                else identity_params
            ),
        )
        identity_rows = {
            int(row[0]): (row[1], row[2]) for row in await cursor.fetchall()
        }
        candidates: list[dict[str, Any]] = []
        for chunk_id, score in merged_ids:
            identity = identity_rows.get(chunk_id)
            if identity is None:
                continue
            candidates.append(
                {
                    "id": chunk_id,
                    "source_table": identity[0],
                    "source_row_id": identity[1],
                    "title": None,
                    "content": None,
                    "meta": {},
                    "metadata": {},
                    "similarity": 0.0,
                    "keyword_rank_val": score,
                    "combined_score": 0.0,
                }
            )
        return candidates
    finally:
        close = getattr(cursor, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


async def _hydrate_oracle_rows(
    connection: OracleRagConnection,
    chunk_ids: Sequence[int],
    filters: Mapping[str, Any] | None = None,
    active_index_version: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Fetch full rows for the fused survivors only."""
    if not chunk_ids:
        return {}
    params = {
        f"hydrate_id_{index}": int(chunk_id) for index, chunk_id in enumerate(chunk_ids)
    }
    clauses, extra_params = _bind_filter_clauses(
        filters,
        table_alias=None,
        require_embedding=False,
        active_index_version=active_index_version,
    )
    params.update(extra_params)
    sql = f"""
        SELECT id, title, content, source_table, source_row_id, meta,
               content_hash, index_version, index_status, indexed_at, updated_at,
               season_year, team_id, player_id
        FROM rag_chunks
        WHERE id IN ({", ".join(f":{key}" for key in params)})
          AND {" AND ".join(clauses)}
    """
    cursor = await acquire_cursor(connection)
    try:
        await cursor.execute(sql, params)
        rows_by_id: dict[int, dict[str, Any]] = {}
        for raw_row in await cursor.fetchall():
            row = _row_mapping(cursor, raw_row)
            row["title"] = await lob_to_text(row.get("title"))
            row["content"] = await lob_to_text(row.get("content"))
            meta = await _parse_meta_async(row.get("meta"))
            row["meta"] = meta
            row["metadata"] = meta
            rows_by_id[int(row["id"])] = row
        return rows_by_id
    finally:
        close = getattr(cursor, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


def _fuse_oracle_results(
    dense_rows: Sequence[dict[str, Any]],
    sparse_rows: Sequence[dict[str, Any]],
    *,
    limit: int,
    intent: str,
) -> list[dict[str, Any]]:
    """Fuse dense and sparse Oracle ranks with the PostgreSQL RRF contract."""
    rrf_k = _ORACLE_RRF_K_BY_INTENT.get(intent, 60)
    scores: dict[tuple[str, str], float] = {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    dense_ranks: dict[tuple[str, str], int] = {}
    sparse_ranks: dict[tuple[str, str], int] = {}

    def row_key(row: Mapping[str, Any]) -> tuple[str, str]:
        return (
            str(row.get("source_table") or ""),
            str(row.get("source_row_id") or row.get("id")),
        )

    for rank, row in enumerate(dense_rows, start=1):
        key = row_key(row)
        rows[key] = dict(row)
        dense_ranks[key] = rank
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    for rank, row in enumerate(sparse_rows, start=1):
        key = row_key(row)
        rows.setdefault(key, dict(row))
        sparse_ranks[key] = rank
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    ordered_keys = sorted(
        scores,
        key=lambda key: (-scores[key], -float(rows[key].get("similarity") or 0.0)),
    )
    results: list[dict[str, Any]] = []
    for key in ordered_keys[: max(int(limit), 1)]:
        row = rows[key]
        row["vector_rank"] = dense_ranks.get(key)
        row["keyword_rank"] = sparse_ranks.get(key)
        row["combined_score"] = scores[key]
        results.append(row)
    return results


async def oracle_similarity_search(
    connection: OracleRagConnection,
    embedding: Sequence[float],
    *,
    limit: int,
    filters: Mapping[str, Any] | None = None,
    keyword: str | None = None,
    document_type: str | None = None,
    game_date: str | None = None,
    intent: str = "",
    active_index_version: Any = _AUTO,
) -> list[dict[str, Any]]:
    """Run Oracle dense retrieval and optional sparse RRF fusion.

    Every returned row satisfies ``retrieval_contract`` (entity scope,
    provenance keys, generation); a violation raises rather than degrading.
    """
    active_index_version = _generation_or_default(active_index_version)
    # Validate up front so a bad filter fails before any database round trip.
    _bind_filter_clauses(filters, active_index_version=active_index_version)
    rows = await _oracle_similarity_search_rows(
        connection,
        embedding,
        limit=limit,
        filters=filters,
        keyword=keyword,
        document_type=document_type,
        game_date=game_date,
        intent=intent,
        active_index_version=active_index_version,
    )
    return _finalize_oracle_rows(
        rows, filters=filters, active_index_version=active_index_version
    )


def _finalize_oracle_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    filters: Mapping[str, Any] | None,
    active_index_version: str | None,
) -> list[dict[str, Any]]:
    normalized = [
        normalize_result(
            row,
            backend="oracle",
            index_generation=(
                f"oracle:{row.get('index_version')}"
                if row.get("index_version")
                else (
                    f"oracle:{active_index_version}" if active_index_version else None
                )
            ),
        )
        for row in rows
    ]
    enforce_result_contract(
        normalized,
        backend="oracle",
        require_generation=bool(active_index_version),
        constraints=entity_constraints(filters),
    )
    return normalized


async def _oracle_similarity_search_rows(
    connection: OracleRagConnection,
    embedding: Sequence[float],
    *,
    limit: int,
    filters: Mapping[str, Any] | None,
    keyword: str | None,
    document_type: str | None,
    game_date: str | None,
    intent: str,
    active_index_version: str | None,
) -> list[dict[str, Any]]:
    requested_limit = max(int(limit), 1)
    dense_rows = await _oracle_dense_similarity_search(
        connection,
        embedding,
        # Fusion needs rank depth, not every candidate's payload: transferring
        # fewer fully-hydrated rows cuts the dominant network cost.
        result_limit=max(requested_limit * 3, requested_limit),
        filters=filters,
        document_type=document_type,
        game_date=game_date,
        active_index_version=active_index_version,
    )
    if not _search_tokens(keyword):
        return dense_rows[:requested_limit]
    try:
        sparse_rows = await _oracle_sparse_search(
            connection,
            keyword,
            candidate_limit=max(requested_limit * 4, 80),
            filters=filters,
            active_index_version=active_index_version,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Oracle sparse retrieval failed; returning dense results: %s", exc
        )
        return dense_rows[:requested_limit]
    fused = _fuse_oracle_results(
        dense_rows,
        sparse_rows,
        limit=requested_limit,
        intent=intent,
    )

    # Hydrate full content only for fused survivors missing it (sparse-only).
    # Rows dropped by chunk-level filters during hydration are removed too.
    missing_ids = {
        int(row["id"])
        for row in fused
        if row.get("content") is None and isinstance(row.get("id"), int)
    }
    if missing_ids:
        hydrated = await _hydrate_oracle_rows(
            connection,
            sorted(missing_ids),
            filters,
            active_index_version=active_index_version,
        )
        fused = [
            row
            for row in fused
            if row["id"] not in missing_ids or row["id"] in hydrated
        ]
        for row in fused:
            full = hydrated.get(row.get("id"))
            if full is not None:
                row.update(full)
                if not isinstance(row.get("meta"), dict) or not row["meta"]:
                    row["meta"] = full.get("meta") or {}
                    row["metadata"] = row["meta"]
    return fused


async def oracle_exact_document_search(
    connection: OracleRagConnection,
    terms: Sequence[str],
    *,
    limit: int,
    source_tables: Sequence[str],
    active_index_version: Any = _AUTO,
) -> list[dict[str, Any]]:
    """Return exact-term document matches using Oracle CLOB substring search."""
    active_index_version = _generation_or_default(active_index_version)
    normalized_terms = [
        str(term).strip().casefold() for term in terms if str(term).strip()
    ]
    if not normalized_terms:
        return []
    params: dict[str, Any] = {}
    source_binds = []
    for index, source_table in enumerate(source_tables):
        key = f"source_{index}"
        source_binds.append(f":{key}")
        params[key] = source_table
    generation_sql = ""
    if active_index_version:
        generation_sql = " AND index_version = :active_index_version"
        params["active_index_version"] = active_index_version
    term_clauses = []
    for index, term in enumerate(normalized_terms[:4]):
        key = f"term_{index}"
        params[key] = term
        term_clauses.append(
            f"(DBMS_LOB.INSTR(LOWER(content), :{key}) > 0 OR "
            f"DBMS_LOB.INSTR(LOWER(title), :{key}) > 0)"
        )
    sql = f"""
        SELECT id, title, content, source_table, source_row_id, meta,
               index_version, season_year, team_id, player_id,
               1.0 AS similarity, 1.0 AS combined_score
        FROM rag_chunks
        WHERE source_table IN ({", ".join(source_binds)})
          AND index_status IN ('ACTIVE', 'INDEXED'){generation_sql}
          AND ({" OR ".join(term_clauses)})
        ORDER BY id
        FETCH FIRST {max(1, int(limit))} ROWS ONLY
    """
    cursor = await acquire_cursor(connection)
    try:
        await cursor.execute(sql, params)
        rendered_rows: list[dict[str, Any]] = []
        for raw_row in await cursor.fetchall():
            row = _row_mapping(cursor, raw_row)
            row["title"] = await lob_to_text(row.get("title"))
            row["content"] = await lob_to_text(row.get("content"))
            meta = await _parse_meta_async(row.get("meta"))
            row["meta"] = meta
            row["metadata"] = meta
            rendered_rows.append(row)
        return _finalize_oracle_rows(
            rendered_rows, filters=None, active_index_version=active_index_version
        )
    finally:
        close = getattr(cursor, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


_ORACLE_RAG_MERGE_SQL = """
    MERGE INTO rag_chunks target
    USING (
        SELECT :source_table AS source_table, :source_row_id AS source_row_id
        FROM dual
    ) incoming
    ON (
        target.source_table = incoming.source_table
        AND target.source_row_id = incoming.source_row_id
    )
    WHEN MATCHED THEN UPDATE SET
        target.title = :title,
        target.content = :content,
        target.meta = :meta,
        target.content_hash = :content_hash,
        target.index_version = :index_version,
        target.index_status = 'ACTIVE',
        target.indexed_at = :indexed_at,
        target.embedding_vector = :embedding_vector,
        target.season_year = :season_year,
        target.league_type_code = :league_type_code,
        target.team_id = :team_id,
        target.player_id = :player_id,
        target.updated_at = :indexed_at
    WHEN NOT MATCHED THEN INSERT (
        source_table, source_row_id, title, content, meta, content_hash,
        index_version, index_status, indexed_at, embedding_vector,
        season_year, league_type_code, team_id, player_id, created_at, updated_at
    ) VALUES (
        :source_table, :source_row_id, :title, :content, :meta, :content_hash,
        :index_version, 'ACTIVE', :indexed_at, :embedding_vector,
        :season_year, :league_type_code, :team_id, :player_id, :indexed_at, :indexed_at
    )
"""

_ORACLE_RAG_TERM_ID_SQL = """
    SELECT id
    FROM rag_chunks
    WHERE source_table = :source_table AND source_row_id = :source_row_id
"""
_ORACLE_RAG_TERM_DELETE_SQL = """
    DELETE FROM rag_chunk_terms WHERE rag_chunk_id = :rag_chunk_id
"""
_ORACLE_RAG_TERM_INSERT_SQL = """
    INSERT INTO rag_chunk_terms (
        rag_chunk_id, source_table, token, term_count, title_count, game_date
    ) VALUES (
        :rag_chunk_id, :source_table, :token, :term_count, :title_count, :game_date
    )
"""


async def _refresh_oracle_sparse_terms(
    cursor: Any,
    *,
    source_table: str,
    source_row_id: str,
    title: str,
    content: str,
    meta: Mapping[str, Any],
) -> None:
    """Replace term postings after one canonical chunk upsert."""
    await cursor.execute(
        _ORACLE_RAG_TERM_ID_SQL,
        {"source_table": source_table, "source_row_id": source_row_id},
    )
    row = await cursor.fetchone()
    if not row:
        raise RuntimeError("Oracle RAG chunk upsert did not return an identity")
    chunk_id = int(row[0])
    await cursor.execute(_ORACLE_RAG_TERM_DELETE_SQL, {"rag_chunk_id": chunk_id})
    term_rows = _sparse_term_rows(
        chunk_id=chunk_id,
        source_table=source_table,
        title=title,
        content=content,
        meta=meta,
    )
    if not term_rows:
        return
    executemany = getattr(cursor, "executemany", None)
    if callable(executemany):
        result = executemany(_ORACLE_RAG_TERM_INSERT_SQL, term_rows)
        if inspect.isawaitable(result):
            await result
        return
    for term_row in term_rows:
        await cursor.execute(_ORACLE_RAG_TERM_INSERT_SQL, term_row)


async def _soft_deactivate_oracle_missing_parts(
    cursor: Any,
    *,
    source_table: str,
    source_prefix: str | None,
    active_source_row_ids: Sequence[str],
    updated_at: datetime,
) -> None:
    """Mark removed multipart chunks as non-retrievable in Oracle."""
    if not source_prefix or source_table == "game_summary":
        return
    active_ids = [value for value in active_source_row_ids if value]
    if len(active_ids) == 1 and active_ids[0] == source_prefix:
        return
    binds = []
    params: dict[str, Any] = {
        "source_table": source_table,
        "source_prefix": source_prefix,
        "source_prefix_like": f"{source_prefix}#part%",
        "updated_at": updated_at,
    }
    for index, source_row_id in enumerate(active_ids):
        key = f"active_source_{index}"
        binds.append(f":{key}")
        params[key] = source_row_id
    active_clause = f"source_row_id NOT IN ({', '.join(binds)})" if binds else "1 = 1"
    await cursor.execute(
        f"""
            UPDATE rag_chunks
            SET index_status = 'DELETED', updated_at = :updated_at
            WHERE source_table = :source_table
              AND (source_row_id = :source_prefix
                   OR source_row_id LIKE :source_prefix_like)
              AND {active_clause}
        """,
        params,
    )


async def upsert_oracle_rag_chunks(
    connection: OracleRagConnection,
    *,
    source_table: str,
    records: Sequence[tuple[Any, str, str, str, Mapping[str, Any]]],
    embeddings: Sequence[Sequence[float] | None],
    season_year: int | None = None,
    league_type_code: str | None = None,
    team_id: str | None = None,
    player_id: str | None = None,
    source_prefix: str | None = None,
    active_source_row_ids: Sequence[str] | None = None,
) -> int:
    """Upsert embedded chunks into Oracle's canonical RAG table."""
    cursor = await acquire_cursor(connection)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    count = 0
    try:
        for record, embedding in zip(records, embeddings, strict=False):
            if embedding is None:
                continue
            _index, source_row_id, title, content, storage_fields = record
            metadata = storage_fields.get("metadata") or {}
            await cursor.execute(
                _ORACLE_RAG_MERGE_SQL,
                {
                    "source_table": source_table,
                    "source_row_id": source_row_id,
                    "title": title,
                    "content": content,
                    "meta": json.dumps(metadata, ensure_ascii=False, default=str),
                    "content_hash": storage_fields.get("content_hash"),
                    "index_version": os.getenv("RAG_INDEX_VERSION", "rag-v1"),
                    "indexed_at": now,
                    "embedding_vector": array.array(
                        "f", [float(value) for value in embedding]
                    ),
                    "season_year": season_year,
                    "league_type_code": league_type_code,
                    "team_id": team_id,
                    "player_id": player_id,
                },
            )
            await _refresh_oracle_sparse_terms(
                cursor,
                source_table=source_table,
                source_row_id=source_row_id,
                title=title,
                content=content,
                meta=metadata,
            )
            count += 1
        await _soft_deactivate_oracle_missing_parts(
            cursor,
            source_table=source_table,
            source_prefix=source_prefix,
            active_source_row_ids=active_source_row_ids or (),
            updated_at=now,
        )
        await connection.commit()
        return count
    except Exception:
        rollback = getattr(connection, "rollback", None)
        if callable(rollback):
            result = rollback()
            if inspect.isawaitable(result):
                await result
        raise
    finally:
        close = getattr(cursor, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


async def oracle_rag_readiness(
    connection: OracleRagConnection,
    *,
    expected_index: str = "IDX_RAG_CHUNKS_EMBEDDING_HNSW",
    expected_dim: int = DEFAULT_EMBED_DIM,
    active_index_version: str | None = None,
) -> dict[str, Any]:
    """Check Oracle row coverage, native vector dimension, and HNSW index.

    With ``active_index_version`` set, also reports how many retrievable rows
    belong to that generation; rows of other versions are never served.
    """
    cursor = await acquire_cursor(connection)
    try:
        await cursor.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM rag_chunks) AS total_rows,
                (SELECT COUNT(*) FROM rag_chunks WHERE embedding_vector IS NOT NULL) AS vector_rows,
                (SELECT COUNT(*) FROM rag_chunks WHERE embedding_vector IS NULL AND index_status IN ('ACTIVE', 'INDEXED')) AS missing_rows,
                (SELECT COUNT(*) FROM rag_chunks WHERE embedding_vector IS NOT NULL AND VECTOR_DIMENSION_COUNT(embedding_vector) = :expected_dim) AS matching_dim_rows,
                (SELECT COUNT(*) FROM user_indexes WHERE index_name = :expected_index AND status = 'VALID' AND visibility = 'VISIBLE') AS valid_index_rows,
                (SELECT COUNT(*) FROM rag_chunks WHERE index_status IN ('ACTIVE', 'INDEXED')) AS retrievable_rows,
                (SELECT COUNT(*) FROM rag_chunks WHERE index_status IN ('ACTIVE', 'INDEXED') AND index_version = :active_index_version) AS active_generation_rows
            FROM dual
            """,
            {
                "expected_dim": expected_dim,
                "expected_index": expected_index.upper(),
                "active_index_version": active_index_version or "",
            },
        )
        row = _row_mapping(cursor, await cursor.fetchone())
        total = int(row.get("total_rows") or 0)
        vectors = int(row.get("vector_rows") or 0)
        missing = int(row.get("missing_rows") or 0)
        matching = int(row.get("matching_dim_rows") or 0)
        valid_index = int(row.get("valid_index_rows") or 0)
        retrievable = int(row.get("retrievable_rows") or 0)
        in_generation = int(row.get("active_generation_rows") or 0)
        generation_ok = active_index_version is None or (
            retrievable > 0 and in_generation == retrievable
        )
        return {
            "ready": bool(
                total
                and total == vectors
                and not missing
                and matching == vectors
                and valid_index == 1
                and generation_ok
            ),
            "active_index_version": active_index_version,
            "retrievable_rows": retrievable,
            "active_generation_rows": in_generation,
            "generation_ok": generation_ok,
            "contract": oracle_capabilities(active_index_version),
            "total_rows": total,
            "vector_rows": vectors,
            "missing_rows": missing,
            "dimension": expected_dim,
            "matching_dim_rows": matching,
            "index": expected_index,
            "index_valid": valid_index == 1,
        }
    finally:
        close = getattr(cursor, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
