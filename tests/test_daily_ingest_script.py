from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "daily_ingest_kbo.sh"


def test_daily_ingest_script_is_documented_as_manual_recovery_only() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "MANUAL RECOVERY ONLY" in script
    assert "DO NOT install a second cron" in script


def test_daily_ingest_script_keeps_operator_recovery_command() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "scripts/ingest_from_kbo.py" in script
    assert "--source-db-url" in script
    assert "game_metadata" in script
    assert "game_summary" in script
