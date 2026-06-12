import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from app.config import get_settings

# Pillow/PIL is an optional runtime dependency (vision endpoints only).
# Provide a lightweight stub so tests can import app.routers.vision
# in environments where Pillow is not installed.
if "PIL" not in sys.modules:
    _pil_stub = ModuleType("PIL")
    _pil_stub.Image = MagicMock()
    sys.modules["PIL"] = _pil_stub
    sys.modules["PIL.Image"] = MagicMock()


# ---------------------------------------------------------------------------
# Settings/env isolation
#
# Some modules mutate global ``os.environ`` as an *import-time* side effect.
# In particular ``scripts.batch_coach_matchup_cache`` runs
# ``_preload_workspace_env()`` at import, which ``setdefault()``s every key
# from the workspace ``.env.prod`` (including ``AI_VECTOR_QUANTIZATION=halfvec``)
# into ``os.environ``. That import happens during *collection* of tests such as
# ``tests/test_batch_auto_brief_helpers.py``, so by the time other modules
# import ``app.routers.vision`` (which calls ``get_settings()`` at import) the
# cached ``Settings`` singleton is poisoned with prod values.
#
# This breaks tests that assume default settings — e.g.
# ``test_retrieval.py::test_similarity_search_without_keyword_keeps_vector_path``
# expects the plain ``embedding <=> %s::vector`` path but gets the
# ``halfvec(256)`` path. The failure only reproduces in the *full* suite (where
# the polluting module is collected), not in isolation.
#
# We snapshot a clean baseline for these keys before any test module is
# collected, then restore it and reset the cached settings around every test.
_ENV_BASELINE_KEYS = ("AI_VECTOR_QUANTIZATION",)
_env_baseline: dict[str, str | None] = {}


def _clear_settings_cache() -> None:
    try:
        get_settings.cache_clear()
    except AttributeError:
        # Keep tests resilient across versions of get_settings helper.
        pass


def pytest_configure(config):
    os.environ.setdefault("POSTGRES_DB_URL", "postgresql://localhost:5432/postgres")

    # pytest_configure runs before test modules are collected/imported, so the
    # environment here is still free of import-time pollution.
    for key in _ENV_BASELINE_KEYS:
        _env_baseline[key] = os.environ.get(key)

    _clear_settings_cache()


@pytest.fixture(autouse=True)
def _reset_settings_state():
    """Restore the clean env baseline and reset cached settings per test.

    Guards against import-time ``os.environ`` pollution (see module docstring)
    leaking into tests that read the cached ``get_settings()`` singleton.
    """
    for key, value in _env_baseline.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    _clear_settings_cache()
    yield
    _clear_settings_cache()
