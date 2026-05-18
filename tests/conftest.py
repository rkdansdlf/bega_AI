import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

from app.config import get_settings

# Pillow/PIL is an optional runtime dependency (vision endpoints only).
# Provide a lightweight stub so tests can import app.routers.vision
# in environments where Pillow is not installed.
if "PIL" not in sys.modules:
    _pil_stub = ModuleType("PIL")
    _pil_stub.Image = MagicMock()
    sys.modules["PIL"] = _pil_stub
    sys.modules["PIL.Image"] = MagicMock()


def pytest_configure(config):
    os.environ.setdefault("POSTGRES_DB_URL", "postgresql://localhost:5432/postgres")

    try:
        get_settings.cache_clear()
    except AttributeError:
        # Keep tests resilient across versions of get_settings helper.
        pass
