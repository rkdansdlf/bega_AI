"""BEGA_EXPLICIT_ENV_ONLY=true 일 때 어떤 모듈도 .env 를 프로세스 환경에 주입하지 않는다."""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from app.config import explicit_env_only_enabled, get_settings

REPO_ROOT = Path(__file__).resolve().parents[1]

# chat_stream 을 import 하면서 dotenv.load_dotenv 호출 여부를 관찰한다.
_PROBE = textwrap.dedent("""
    import dotenv

    calls = []
    dotenv.load_dotenv = lambda *a, **k: calls.append((a, k)) or True

    import app.routers.chat_stream  # noqa: F401

    print("LOAD_DOTENV_CALLS=%d" % len(calls))
    """)


def _probe(explicit_env_only: str | None) -> int:
    env = {k: v for k, v in os.environ.items() if k != "BEGA_EXPLICIT_ENV_ONLY"}
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["EMBED_PROVIDER"] = "local"
    env["POSTGRES_DB_URL"] = "postgresql://explicit:pw@localhost:5432/explicit"
    if explicit_env_only is not None:
        env["BEGA_EXPLICIT_ENV_ONLY"] = explicit_env_only
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("LOAD_DOTENV_CALLS=")
    )
    return int(line.split("=", 1)[1])


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_explicit_env_only_enabled_truthy(monkeypatch, value):
    monkeypatch.setenv("BEGA_EXPLICIT_ENV_ONLY", value)
    assert explicit_env_only_enabled() is True


@pytest.mark.parametrize("value", [None, "", "0", "false", "off"])
def test_explicit_env_only_enabled_falsy(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("BEGA_EXPLICIT_ENV_ONLY", raising=False)
    else:
        monkeypatch.setenv("BEGA_EXPLICIT_ENV_ONLY", value)
    assert explicit_env_only_enabled() is False


def test_chat_stream_import_skips_load_dotenv_when_explicit_env_only():
    assert _probe("true") == 0


def test_chat_stream_import_still_loads_dotenv_by_default():
    assert _probe(None) == 1


def test_explicit_postgres_url_wins_when_env_file_is_not_read(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "OCI_DB_URL=postgresql://leaked:pw@remote:5432/leaked\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OCI_DB_URL", raising=False)
    monkeypatch.setenv("BEGA_EXPLICIT_ENV_ONLY", "true")
    monkeypatch.setenv(
        "POSTGRES_DB_URL", "postgresql://explicit:pw@localhost:5432/explicit"
    )
    get_settings.cache_clear()
    try:
        assert (
            get_settings().database_url
            == "postgresql://explicit:pw@localhost:5432/explicit"
        )
    finally:
        get_settings.cache_clear()


_LOAD_DOTENV_CALL = re.compile(r"^(?P<indent>\s*)(?:\w+\.)?load_dotenv\(", re.MULTILINE)


def _unguarded_load_dotenv_calls() -> list[str]:
    """app/ 과 scripts/ 에서 explicit_env_only_enabled() 가드 없는 load_dotenv( 호출을 찾는다.

    가드는 같은 함수/모듈 안에서 `explicit_env_only_enabled()` 를 호출한 뒤(= 호출보다
    위쪽 어딘가) load_dotenv 를 부르는 형태만 인정한다.
    """
    offenders: list[str] = []
    for base in ("app", "scripts"):
        for path in sorted((REPO_ROOT / base).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            for match in _LOAD_DOTENV_CALL.finditer(source):
                preceding = source[: match.start()]
                if "explicit_env_only_enabled()" not in preceding:
                    line = preceding.count("\n") + 1
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{line}")
    return offenders


def test_every_load_dotenv_call_follows_explicit_env_only_flag():
    assert _unguarded_load_dotenv_calls() == []
