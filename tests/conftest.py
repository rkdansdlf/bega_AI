import os

from app.config import get_settings


def pytest_configure(config):
    os.environ.setdefault("POSTGRES_DB_URL", "postgresql://localhost:5432/postgres")

    try:
        get_settings.cache_clear()
    except AttributeError:
        # Keep tests resilient across versions of get_settings helper.
        pass
