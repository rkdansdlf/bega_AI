from __future__ import annotations

import pytest

from app.config import Settings




def test_oracle_rag_can_start_with_manual_ingest_path() -> None:
    settings = Settings(
        _env_file=None,
        app_env="local",
        rag_profile="local-oracle",
        rag_backend="oracle",
        postgres_db_url="postgresql://user:pass@localhost/ai",
        ai_rag_db_url="oracle+oracledb://rag_user:password@adb_high",
        ingest_worker_enabled=False,
    )

    assert settings.rag_db_url.startswith("oracle+oracledb://")
