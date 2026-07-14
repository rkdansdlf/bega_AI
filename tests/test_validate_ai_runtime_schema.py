from scripts.validate_ai_runtime_schema import _env_flag, resolve_db_conninfo


def test_preflight_disables_fastapi_app_initialization():
    import os

    assert os.environ.get("BEGA_SKIP_APP_INIT") == "1"


def test_env_flag_accepts_common_true_values(monkeypatch):
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("FLAG", value)
        assert _env_flag("FLAG") is True


def test_env_flag_defaults_to_false(monkeypatch):
    monkeypatch.delenv("FLAG", raising=False)

    assert _env_flag("FLAG") is False


def test_resolve_db_conninfo_matches_runtime_precedence():
    assert resolve_db_conninfo(
        {
            "OCI_DB_URL": "postgresql://oci/db",
            "POSTGRES_DB_URL": "postgresql://postgres/db",
        }
    ) == "postgresql://oci/db"


def test_resolve_db_conninfo_allows_explicit_rehearsal_override():
    assert resolve_db_conninfo(
        {
            "AI_SCHEMA_DB_URL": "postgresql://temporary/rehearsal",
            "OCI_DB_URL": "postgresql://oci/db",
        }
    ) == "postgresql://temporary/rehearsal"
