"""FastAPI 애플리케이션을 지연 생성 방식으로 노출하는 초기화 모듈.

외부 엔트리포인트가 `app` 객체를 임포트할 때 불필요한 부수 효과 없이
서비스를 사용할 수 있도록 해 주는 래퍼 역할을 한다.
"""

from functools import lru_cache

from .main import create_app


@lru_cache(maxsize=1)
def get_app():
    return create_app()


app = get_app()
