"""Shared helpers for storing searchable RAG chunks.

The storage policy intentionally keeps raw logs out of the vector index and
stores only chunks with enough standalone retrieval value.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

_WHITESPACE_RE = re.compile(r"\s+")
_SOURCE_LINE_RE = re.compile(r"(?m)^\s*(?:출처|source)\s*:.*$")
_URL_RE = re.compile(r"https?://\S+")
_MEANINGFUL_RE = re.compile(r"[0-9A-Za-z가-힣]")
_PART_SUFFIX_RE = re.compile(r"#part(\d+)$")
_API_PATH_RE = re.compile(
    r"(?:^|[\s`'\"(])/(?:api/)?[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+"
)
_OAUTH_URI_RE = re.compile(r"https?://\S*(?:oauth2|redirect_uri|/login/)\S*")
_ERROR_CODE_RE = re.compile(
    r"\b(?:[45]\d{2}|ERR_[A-Z0-9_]+|[A-Z][A-Z0-9_]{2,}|[A-Za-z]+Error)\b"
)
_CONFIG_KEY_RE = re.compile(
    r"\b(?:[a-z][a-z0-9_]*(?:\.[a-z0-9_-]+)+|[A-Z][A-Z0-9_]{2,})\b"
)
_KBO_TEAM_RE = re.compile(
    r"\b(?:KIA|KT|LG|NC|SSG|롯데|삼성|한화|두산|키움|기아|케이티|엔씨)\b",
    re.IGNORECASE,
)
_SEAT_NAME_RE = re.compile(
    r"(?:블루석|레드석|테이블석|외야석|내야석|응원석|중앙석|지정석|프리미엄석|익사이팅존|스카이박스|seat|section)",
    re.IGNORECASE,
)
_RAW_CHAT_SOURCE_TYPES = {"raw_chat", "raw_chat_log"}
_RAW_CHAT_TRANSCRIPT_RE = re.compile(
    r"(?mi)^\s*(?:사용자|user|ai|assistant|챗봇)\s*[:：]"
)
_CHAT_MEMORY_SUMMARY_MARKERS = (
    "[문제]",
    "[결론]",
    "[요약]",
    "[기억]",
    "요약:",
    "summary:",
    "tl;dr",
)
_SECRET_KEY_PREFIX_RE = r"(?:[A-Za-z0-9]+[_-])*"
_SECRET_ASSIGNMENT_VALUE_RE = r"\s*[:=]\s*[\"']?[^\s,;\"'}]+"


def _secret_assignment_pattern(keyword_pattern: str) -> re.Pattern[str]:
    return re.compile(
        r"(?i)[\"']?"
        r"(?<![A-Za-z0-9])"
        + _SECRET_KEY_PREFIX_RE
        + r"(?:"
        + keyword_pattern
        + r")(?![A-Za-z0-9])"
        + r"[\"']?"
        + _SECRET_ASSIGNMENT_VALUE_RE
    )


_SECRET_KEY_RE = _secret_assignment_pattern(
    r"api[_-]?key|apikey|client[_-]?secret|jwt[_-]?secret|password|database[_-]?url"
)
_RRN_RE = re.compile(
    r"(?i)(?:\b\d{6}-[1-4]\d{6}\b|(?:주민등록번호|주민번호|rrn|resident registration)"
    r"\s*[:=]?\s*\d{6}[-\s]?[1-4]\d{6})"
)
_PHONE_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:01[016789]|02|0[3-6][1-5])[-.\s]\d{3,4}[-.\s]\d{4}\b"
    r"|(?:전화번호|휴대폰|연락처|phone|tel|mobile)\s*[:=]?\s*"
    r"(?:01[016789]|02|0[3-6][1-5])[-.\s]?\d{3,4}[-.\s]?\d{4}"
    r")"
)
SECRET_PATTERNS = [
    _SECRET_KEY_RE,
    re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mariadb|oracle)://[^/\s:@]+:[^@\s]+@"),
]
_SENSITIVE_PATTERNS = (
    (
        "api_key",
        _secret_assignment_pattern(r"api[_-]?key|apikey"),
    ),
    (
        "client_secret",
        _secret_assignment_pattern(r"client[_-]?secret"),
    ),
    (
        "jwt_secret",
        _secret_assignment_pattern(r"jwt[_-]?secret"),
    ),
    (
        "password",
        _secret_assignment_pattern(r"password"),
    ),
    (
        "bearer_token",
        re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]+"),
    ),
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "db_url_credential",
        re.compile(
            r"(?i)\b(?:postgres(?:ql)?|mysql|mariadb|oracle)://[^/\s:@]+:[^@\s]+@"
        ),
    ),
    (
        "database_url",
        _secret_assignment_pattern(r"database[_-]?url"),
    ),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "phone",
        _PHONE_RE,
    ),
    ("rrn", _RRN_RE),
)
_LOW_VALUE_PHRASES = {
    "네",
    "넵",
    "예",
    "아니요",
    "알겠습니다",
    "확인했습니다",
    "감사합니다",
    "고맙습니다",
    "좋아요",
    "확인",
    "ok",
    "okay",
}
RAW_CHAT_SOURCE_TABLES = {"ai_chat_message", "ai_chat_messages", "chat_messages"}
OFFICIAL_KNOWLEDGE_TABLES = {"kbo_regulations", "kbo_definitions"}
CANONICAL_KBO_SOURCE_TABLES = {
    "teams",
    "team_franchises",
    "team_history",
    "stadiums",
    "kbo_seasons",
    "player_basic",
    "awards",
    "player_movements",
    "player_season_batting",
    "player_season_pitching",
    "team_season_batting",
    "team_season_pitching",
    "stat_rankings",
    "game",
    "game_metadata",
    "game_flow_summary",
    "game_lineups",
    "game_batting_stats",
    "game_pitching_stats",
    "game_summary",
}
DEFAULT_RAG_EMBEDDING_DIM = 256
DEFAULT_RAG_EMBEDDING_VERSION = 2


def normalize_content_for_hash(text: str) -> str:
    """Normalize text before hashing and duplicate checks."""
    if not text:
        return ""
    without_source = _SOURCE_LINE_RE.sub("", text)
    normalized = _WHITESPACE_RE.sub(" ", without_source).strip().casefold()
    return normalized


def strip_source_footer(text: str) -> str:
    if not text:
        return ""
    return _SOURCE_LINE_RE.sub("", text).strip()


def is_search_worthy_content(text: str, *, min_chars: int = 50) -> bool:
    """Return whether a chunk has enough value to store in vector search."""
    body = strip_source_footer(text)
    normalized = normalize_content_for_hash(body)
    if contains_sensitive_content(body):
        return False
    if normalized in _LOW_VALUE_PHRASES:
        return False
    if not _MEANINGFUL_RE.search(normalized):
        return False
    if _is_short_important_content(body):
        return True
    if len(normalized) < min_chars:
        return False
    without_urls = _URL_RE.sub("", normalized).strip()
    if not without_urls:
        return False
    return True


def scan_sensitive_content(value: Any) -> List[str]:
    """Return sensitive finding types without exposing matched values."""
    if value is None:
        return []
    text = value if isinstance(value, str) else json_dumps(value)
    if not text:
        return []
    findings: List[str] = []
    for finding_type, pattern in _SENSITIVE_PATTERNS:
        if pattern.search(text):
            findings.append(finding_type)
    return findings


def contains_sensitive_content(value: Any) -> bool:
    return bool(scan_sensitive_content(value))


def contains_secret(text: str) -> bool:
    return bool(text and any(pattern.search(text) for pattern in SECRET_PATTERNS))


def _raise_if_sensitive_storage_payload(**payload: Any) -> None:
    findings: List[str] = []
    for value in payload.values():
        findings.extend(scan_sensitive_content(value))
    if findings:
        finding_types = ",".join(sorted(set(findings)))
        raise ValueError(f"rag chunk contains sensitive content: {finding_types}")


def _is_short_important_content(text: str) -> bool:
    """Allow compact identifiers that are valuable retrieval targets."""
    normalized = normalize_content_for_hash(text)
    if not normalized:
        return False
    if _OAUTH_URI_RE.search(text) or _OAUTH_URI_RE.search(normalized):
        return True
    if not _URL_RE.sub("", normalized).strip():
        return False
    return any(
        pattern.search(text) or pattern.search(normalized)
        for pattern in (
            _API_PATH_RE,
            _ERROR_CODE_RE,
            _CONFIG_KEY_RE,
            _KBO_TEAM_RE,
            _SEAT_NAME_RE,
        )
    )


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_content_for_hash(text).encode("utf-8")).hexdigest()


def infer_chunk_index(source_row_id: str) -> int:
    match = _PART_SUFFIX_RE.search(source_row_id or "")
    if not match:
        return 1
    try:
        return int(match.group(1))
    except ValueError:
        return 1


def base_source_row_id(source_row_id: str) -> str:
    if not source_row_id:
        return ""
    return _PART_SUFFIX_RE.sub("", source_row_id)


def infer_topic_key(
    source_table: str,
    source_row_id: str,
    meta: Optional[Dict[str, Any]] = None,
    explicit: Optional[str] = None,
) -> str:
    if explicit:
        return explicit
    metadata = meta or {}
    for key in ("topic_key", "canonical_topic_key"):
        value = metadata.get(key)
        if value:
            return str(value)
    return f"{source_table}:{base_source_row_id(source_row_id)}"


def chunk_hash(
    *,
    source_table: str,
    source_row_id: str,
    chunk_index: int,
    content_hash_value: str,
) -> str:
    raw = f"{source_table}|{source_row_id}|{chunk_index}|{content_hash_value}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resolve_embedding_model(settings: Any) -> str:
    provider = str(getattr(settings, "embed_provider", "") or "unknown").strip()
    if provider == "openai":
        model = (
            getattr(settings, "openai_embed_model", None)
            or getattr(settings, "embed_model", None)
            or "text-embedding-3-small"
        )
    elif provider == "openrouter":
        model = (
            getattr(settings, "openrouter_embed_model", None)
            or getattr(settings, "embed_model", None)
            or "openai/text-embedding-3-small"
        )
    elif provider == "gemini":
        model = (
            getattr(settings, "gemini_embed_model", None)
            or getattr(settings, "embed_model", None)
            or "text-embedding-004"
        )
    elif provider == "hf":
        model = (
            getattr(settings, "embed_model", None)
            or getattr(settings, "hf_embed_model", None)
            or "intfloat/multilingual-e5-large"
        )
    elif provider == "local":
        model = "local"
    else:
        model = getattr(settings, "embed_model", None) or "unknown"
    return f"{provider}:{model}"


def resolve_embedding_version(settings: Any) -> int:
    return max(
        1,
        int(
            getattr(
                settings,
                "rag_embedding_version",
                DEFAULT_RAG_EMBEDDING_VERSION,
            )
            or DEFAULT_RAG_EMBEDDING_VERSION
        ),
    )


def resolve_embedding_dim(settings: Any) -> int:
    return max(
        1,
        int(
            getattr(settings, "embed_dim", DEFAULT_RAG_EMBEDDING_DIM)
            or DEFAULT_RAG_EMBEDDING_DIM
        ),
    )


def resolve_chunking_version(settings: Any) -> int:
    return max(1, int(getattr(settings, "rag_chunking_version", 1) or 1))


def infer_quality_score(
    source_type: str,
    source_table: str,
    meta: Optional[Dict[str, Any]] = None,
    explicit: Optional[float] = None,
) -> float:
    """Infer retrieval quality score from source semantics."""
    if explicit is not None:
        return float(explicit)
    metadata = meta or {}
    meta_score = metadata.get("quality_score")
    if meta_score is not None:
        return float(meta_score)

    normalized_source_type = str(source_type or "").strip()
    normalized_source_table = str(source_table or "").strip()
    if (
        normalized_source_type == "official_rulebook"
        or normalized_source_table in OFFICIAL_KNOWLEDGE_TABLES
        or normalized_source_table.startswith("kbo_regulations")
    ):
        return 0.95
    if (
        normalized_source_type in {"canonical_knowledge", "kbo_db_table"}
        or normalized_source_table in CANONICAL_KBO_SOURCE_TABLES
    ):
        return 0.85
    if normalized_source_type in {"document", "markdown_doc"}:
        return 0.70
    if normalized_source_type == "chat_memory":
        return 0.55
    return 0.50


def infer_source_type(
    source_table: str,
    meta: Optional[Dict[str, Any]] = None,
    explicit: Optional[str] = None,
) -> str:
    if explicit:
        return explicit
    metadata = meta or {}
    value = metadata.get("source_type") or metadata.get("knowledge_type")
    if value == "chat_memory":
        return "chat_memory"
    if source_table in RAW_CHAT_SOURCE_TABLES:
        return "raw_chat_log"
    if source_table in {"kbo_regulations", "kbo_definitions", "markdown_docs"}:
        return "canonical_knowledge"
    if source_table.startswith("kbo_regulations"):
        return "canonical_knowledge"
    if source_table in CANONICAL_KBO_SOURCE_TABLES:
        return "canonical_knowledge"
    return "document"


def infer_source_uri(
    source_table: str,
    source_row_id: str,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    metadata = meta or {}
    for key in ("source_uri", "source_url", "source_path", "source_file"):
        value = metadata.get(key)
        if value:
            return str(value)
    return f"{source_table}#{source_row_id}"


def normalize_metadata(
    meta: Optional[Dict[str, Any]],
    *,
    source_type: str,
    source_uri: str,
    topic_key: str,
    chunk_index: int,
    valid_from: Any = None,
    valid_to: Any = None,
    expires_at: Any = None,
) -> Dict[str, Any]:
    metadata = dict(meta or {})
    metadata.setdefault("source_type", source_type)
    metadata.setdefault("source_uri", source_uri)
    metadata.setdefault("topic_key", topic_key)
    metadata.setdefault("source_version", "v1")
    metadata.setdefault("visibility", "public")
    metadata.setdefault("sensitivity_checked", True)
    metadata.setdefault("chunk_index", chunk_index)
    if valid_from is not None:
        metadata.setdefault("valid_from", valid_from)
    if valid_to is not None:
        metadata.setdefault("valid_to", valid_to)
    if expires_at is not None:
        metadata.setdefault("expires_at", expires_at)
    return metadata


def guard_raw_chat_source(
    source_table: str, source_type: str, content: str = ""
) -> None:
    if source_type in _RAW_CHAT_SOURCE_TYPES:
        raise ValueError(
            "raw chat rows must not be embedded directly; store summarized "
            "memory chunks with source_type='chat_memory' instead"
        )
    if source_table in RAW_CHAT_SOURCE_TABLES and source_type != "chat_memory":
        raise ValueError(
            "raw chat rows must not be embedded directly; store summarized "
            "memory chunks with source_type='chat_memory' instead"
        )
    if source_type == "chat_memory":
        _guard_chat_memory_summary(content)


def _guard_chat_memory_summary(content: str) -> None:
    normalized = normalize_content_for_hash(content)
    has_summary_marker = any(
        marker in normalized for marker in _CHAT_MEMORY_SUMMARY_MARKERS
    )
    if _RAW_CHAT_TRANSCRIPT_RE.search(content) and not has_summary_marker:
        raise ValueError(
            "chat_memory chunks must be summarized memory, not raw transcript"
        )
    if not has_summary_marker:
        raise ValueError("chat_memory chunks must include a summary marker")


def build_chunk_storage_fields(
    *,
    settings: Any,
    source_table: str,
    source_row_id: str,
    content: str,
    meta: Optional[Dict[str, Any]],
    source_type: Optional[str] = None,
    source_uri: Optional[str] = None,
    topic_key: Optional[str] = None,
    valid_from: Any = None,
    valid_to: Any = None,
    expires_at: Any = None,
    embedding_model: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    embedding_version: Optional[int] = None,
    chunking_version: Optional[int] = None,
    quality_score: Optional[float] = None,
) -> Dict[str, Any]:
    chunk_index = infer_chunk_index(source_row_id)
    resolved_source_type = infer_source_type(source_table, meta, source_type)
    _raise_if_sensitive_storage_payload(content=content, meta=meta)
    guard_raw_chat_source(source_table, resolved_source_type, content)
    resolved_source_uri = source_uri or infer_source_uri(
        source_table, source_row_id, meta
    )
    resolved_topic_key = infer_topic_key(source_table, source_row_id, meta, topic_key)
    resolved_metadata = normalize_metadata(
        meta,
        source_type=resolved_source_type,
        source_uri=resolved_source_uri,
        topic_key=resolved_topic_key,
        chunk_index=chunk_index,
        valid_from=valid_from,
        valid_to=valid_to,
        expires_at=expires_at,
    )
    hash_value = content_hash(content)
    return {
        "source_type": resolved_source_type,
        "source_uri": resolved_source_uri,
        "topic_key": resolved_topic_key,
        "content_hash": hash_value,
        "chunk_hash": chunk_hash(
            source_table=source_table,
            source_row_id=source_row_id,
            chunk_index=chunk_index,
            content_hash_value=hash_value,
        ),
        "embedding_model": embedding_model or resolve_embedding_model(settings),
        "embedding_dim": (
            max(1, int(embedding_dim))
            if embedding_dim is not None
            else resolve_embedding_dim(settings)
        ),
        "embedding_version": (
            max(1, int(embedding_version))
            if embedding_version is not None
            else resolve_embedding_version(settings)
        ),
        "chunking_version": chunking_version or resolve_chunking_version(settings),
        "quality_score": infer_quality_score(
            resolved_source_type,
            source_table,
            meta=meta,
            explicit=quality_score,
        ),
        "valid_from": valid_from,
        "valid_to": valid_to,
        "expires_at": expires_at,
        "metadata": resolved_metadata,
    }


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def fetch_existing_embedding_texts(
    cur: Any,
    *,
    content_hashes: Iterable[str],
    embedding_model: str,
    embedding_dim: int,
    embedding_version: int,
    chunking_version: int,
) -> Dict[str, str]:
    hashes = sorted({value for value in content_hashes if value})
    if not hashes:
        return {}
    cur.execute(
        """
        SELECT DISTINCT ON (content_hash)
            content_hash,
            embedding::text AS embedding_text
        FROM rag_chunks
        WHERE content_hash = ANY(%s)
          AND embedding_model = %s
          AND embedding_dim = %s
          AND embedding_version = %s
          AND chunking_version = %s
          AND embedding IS NOT NULL
        ORDER BY content_hash, updated_at DESC
        """,
        (hashes, embedding_model, embedding_dim, embedding_version, chunking_version),
    )
    rows = cur.fetchall()
    found: Dict[str, str] = {}
    for row in rows:
        if isinstance(row, dict):
            hash_value = row.get("content_hash")
            embedding_text = row.get("embedding_text")
        else:
            hash_value = row[0]
            embedding_text = row[1]
        if hash_value and embedding_text:
            found[str(hash_value)] = str(embedding_text)
    return found


RAG_CHUNKS_UPSERT_SQL = """
INSERT INTO rag_chunks (
    meta,
    metadata,
    source_type,
    source_uri,
    topic_key,
    content_hash,
    chunk_hash,
    embedding_model,
    embedding_dim,
    embedding_version,
    chunking_version,
    quality_score,
    is_active,
    valid_from,
    valid_to,
    expires_at,
    season_year,
    season_id,
    league_type_code,
    team_id,
    player_id,
    source_table,
    source_row_id,
    title,
    content,
    embedding
) VALUES (
    %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, true,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector
)
ON CONFLICT (source_table, source_row_id)
DO UPDATE SET
    meta = EXCLUDED.meta,
    metadata = EXCLUDED.metadata,
    source_type = EXCLUDED.source_type,
    source_uri = EXCLUDED.source_uri,
    topic_key = EXCLUDED.topic_key,
    content_hash = EXCLUDED.content_hash,
    chunk_hash = EXCLUDED.chunk_hash,
    embedding_model = EXCLUDED.embedding_model,
    embedding_dim = EXCLUDED.embedding_dim,
    embedding_version = EXCLUDED.embedding_version,
    chunking_version = EXCLUDED.chunking_version,
    quality_score = EXCLUDED.quality_score,
    is_active = true,
    valid_from = EXCLUDED.valid_from,
    valid_to = EXCLUDED.valid_to,
    expires_at = EXCLUDED.expires_at,
    content = EXCLUDED.content,
    embedding =
        CASE
            WHEN rag_chunks.content_hash = EXCLUDED.content_hash
                THEN COALESCE(EXCLUDED.embedding, rag_chunks.embedding)
            ELSE EXCLUDED.embedding
        END,
    season_year = COALESCE(EXCLUDED.season_year, rag_chunks.season_year),
    season_id = COALESCE(EXCLUDED.season_id, rag_chunks.season_id),
    league_type_code = COALESCE(EXCLUDED.league_type_code, rag_chunks.league_type_code),
    team_id = COALESCE(EXCLUDED.team_id, rag_chunks.team_id),
    player_id = COALESCE(EXCLUDED.player_id, rag_chunks.player_id),
    title = EXCLUDED.title,
    updated_at = now();
"""


def build_upsert_tuple(
    *,
    meta: Optional[Dict[str, Any]],
    storage_fields: Dict[str, Any],
    season_year: Optional[int],
    season_id: Optional[int],
    league_type_code: Optional[int],
    team_id: Optional[str],
    player_id: Optional[str],
    source_table: str,
    source_row_id: str,
    title: str,
    content: str,
    embedding_text: Optional[str],
) -> tuple[Any, ...]:
    metadata = storage_fields["metadata"]
    _raise_if_sensitive_storage_payload(content=content, meta=meta, metadata=metadata)
    return (
        json_dumps(meta or {}),
        json_dumps(metadata),
        storage_fields["source_type"],
        storage_fields["source_uri"],
        storage_fields["topic_key"],
        storage_fields["content_hash"],
        storage_fields["chunk_hash"],
        storage_fields["embedding_model"],
        storage_fields["embedding_dim"],
        storage_fields["embedding_version"],
        storage_fields["chunking_version"],
        storage_fields["quality_score"],
        storage_fields.get("valid_from"),
        storage_fields.get("valid_to"),
        storage_fields.get("expires_at"),
        season_year,
        season_id,
        league_type_code,
        team_id,
        player_id,
        source_table,
        source_row_id,
        title,
        content,
        embedding_text,
    )


async def fetch_existing_embedding_texts_async(
    cur: Any,
    *,
    content_hashes: Iterable[str],
    embedding_model: str,
    embedding_dim: int,
    embedding_version: int,
    chunking_version: int,
) -> Dict[str, str]:
    """Async (psycopg3 AsyncCursor) variant of ``fetch_existing_embedding_texts``.

    Used by the async ingest router. The standalone sync ingest scripts keep
    using the sync version above (they manage their own sync connections).
    """
    hashes = sorted({value for value in content_hashes if value})
    if not hashes:
        return {}
    await cur.execute(
        """
        SELECT DISTINCT ON (content_hash)
            content_hash,
            embedding::text AS embedding_text
        FROM rag_chunks
        WHERE content_hash = ANY(%s)
          AND embedding_model = %s
          AND embedding_dim = %s
          AND embedding_version = %s
          AND chunking_version = %s
          AND embedding IS NOT NULL
        ORDER BY content_hash, updated_at DESC
        """,
        (hashes, embedding_model, embedding_dim, embedding_version, chunking_version),
    )
    rows = await cur.fetchall()
    found: Dict[str, str] = {}
    for row in rows:
        if isinstance(row, dict):
            hash_value = row.get("content_hash")
            embedding_text = row.get("embedding_text")
        else:
            hash_value = row[0]
            embedding_text = row[1]
        if hash_value and embedding_text:
            found[str(hash_value)] = str(embedding_text)
    return found


def soft_deactivate_missing_parts(
    cur: Any,
    *,
    source_table: str,
    source_prefix: str,
    active_source_row_ids: Sequence[str],
) -> None:
    if not source_prefix:
        return
    if source_table == "game_summary":
        return
    active_ids = [value for value in active_source_row_ids if value]
    if len(active_ids) == 1 and active_ids[0] == source_prefix:
        return
    cur.execute(
        """
        UPDATE rag_chunks
        SET is_active = false,
            updated_at = now()
        WHERE source_table = %s
          AND (source_row_id = %s OR source_row_id LIKE %s)
          AND NOT (source_row_id = ANY(%s))
        """,
        (
            source_table,
            source_prefix,
            f"{source_prefix}#part%",
            active_ids,
        ),
    )


async def soft_deactivate_missing_parts_async(
    cur: Any,
    *,
    source_table: str,
    source_prefix: str,
    active_source_row_ids: Sequence[str],
) -> None:
    """Async variant of ``soft_deactivate_missing_parts`` for the ingest router."""
    if not source_prefix:
        return
    if source_table == "game_summary":
        return
    active_ids = [value for value in active_source_row_ids if value]
    if len(active_ids) == 1 and active_ids[0] == source_prefix:
        return
    await cur.execute(
        """
        UPDATE rag_chunks
        SET is_active = false,
            updated_at = now()
        WHERE source_table = %s
          AND (source_row_id = %s OR source_row_id LIKE %s)
          AND NOT (source_row_id = ANY(%s))
        """,
        (
            source_table,
            source_prefix,
            f"{source_prefix}#part%",
            active_ids,
        ),
    )


def soft_deactivate_missing_source_rows(
    cur: Any,
    *,
    source_table: str,
    active_source_row_ids: Sequence[str],
    dry_run: bool = True,
) -> int:
    """Soft deactivate active chunks whose source row vanished in a full ingest."""
    active_ids = sorted({value for value in active_source_row_ids if value})
    if not source_table or not active_ids:
        return 0

    predicate = """
        source_table = %s
          AND COALESCE(is_active, true) = true
          AND NOT (
              source_row_id = ANY(%s)
              OR regexp_replace(source_row_id, '#part[0-9]+$', '') = ANY(%s)
          )
    """
    params = (source_table, active_ids, active_ids)
    if dry_run:
        cur.execute("SELECT count(*) FROM rag_chunks WHERE " + predicate, params)
        row = cur.fetchone()
        if isinstance(row, dict):
            return int(row.get("count", 0) or 0)
        if row:
            return int(row[0] or 0)
        return 0

    cur.execute(
        """
        UPDATE rag_chunks
        SET is_active = false,
            valid_to = COALESCE(valid_to, now()),
            updated_at = now()
        WHERE
        """ + predicate,
        params,
    )
    return int(getattr(cur, "rowcount", 0) or 0)
