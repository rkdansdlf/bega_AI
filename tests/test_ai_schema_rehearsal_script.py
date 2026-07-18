from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rehearse_ai_schema_migration.sh"
MIGRATION_SCRIPT = SCRIPT.with_name("migrate_ai_runtime_schema.sh")


def test_ai_schema_rehearsal_requires_disposable_database_confirmation():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "AI_SCHEMA_REHEARSAL_DB_NAME" in text
    assert "AI_SCHEMA_REHEARSAL_CONFIRM" in text
    assert "I_UNDERSTAND_TEMP_DB" in text
    assert "Refusing to run" in text


def test_ai_schema_rehearsal_runs_migration_and_checks_contract_objects():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "migrate_ai_runtime_schema.sh" in text
    assert "scripts.validate_ai_runtime_schema" in text
    assert "AI_PYTHON" in text


def test_ai_schema_migration_wrapper_is_valid_and_matches_runtime_precedence():
    result = subprocess.run(
        ["bash", "-n", str(MIGRATION_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    text = MIGRATION_SCRIPT.read_text(encoding="utf-8")
    assert (
        'DB_URL="${AI_SCHEMA_DB_URL:-${OCI_DB_URL:-${POSTGRES_DB_URL:-${SUPABASE_DB_URL:-}}}}"'
        in text
    )
