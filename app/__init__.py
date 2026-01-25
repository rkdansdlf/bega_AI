"""FastAPI 애플리케이션을 지연 생성 방식으로 노출하는 초기화 모듈."""

from __future__ import annotations

import os
import sys
from functools import lru_cache


@lru_cache(maxsize=1)
def get_app():
    from .main import create_app

    return create_app()


def _should_init_app() -> bool:
    return not (os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules)


app = get_app() if _should_init_app() else None
