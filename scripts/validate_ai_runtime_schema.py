"""Run the same read-only schema contract used by managed AI startup."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping

import psycopg

# Import only the schema contract; never initialize the FastAPI app from this
# operator preflight process.
os.environ.setdefault("BEGA_SKIP_APP_INIT", "1")

from app.db.schema_contract import SchemaContractError, validate_schema_contract


def _env_flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def resolve_db_conninfo(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the same DB precedence used by Settings.source_db_url."""

    values = environ if environ is not None else os.environ
    for name in ("AI_SCHEMA_DB_URL", "OCI_DB_URL", "POSTGRES_DB_URL", "SUPABASE_DB_URL"):
        value = values.get(name, "").strip()
        if value:
            return value
    return ""


async def _validate(conninfo: str, *, require_vector_index: bool) -> None:
    conn = await psycopg.AsyncConnection.connect(conninfo, autocommit=True)
    try:
        await validate_schema_contract(
            conn,
            require_vector_index=require_vector_index,
        )
    finally:
        await conn.close()


def main() -> int:
    conninfo = resolve_db_conninfo()
    if not conninfo:
        print(
            "AI_SCHEMA_PREFLIGHT_FAILED: AI_SCHEMA_DB_URL, OCI_DB_URL, "
            "or POSTGRES_DB_URL is required",
            file=sys.stderr,
        )
        return 2

    try:
        asyncio.run(
            _validate(
                conninfo,
                require_vector_index=_env_flag(
                    "CHAT_SEMANTIC_CACHE_VECTOR_INDEX_ENABLED"
                ),
            )
        )
    except SchemaContractError as exc:
        print(f"AI_SCHEMA_CONTRACT_INVALID: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"AI_SCHEMA_PREFLIGHT_FAILED: {exc}", file=sys.stderr)
        return 1

    print("AI_SCHEMA_CONTRACT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
