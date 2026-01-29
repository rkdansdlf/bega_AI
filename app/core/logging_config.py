import logging
import sys
import structlog
from app.config import get_settings


def configure_logging():
    settings = get_settings()

    # 기본 로그 레벨 설정
    log_level = logging.DEBUG if settings.debug else logging.INFO

    # 1. Standard Library Logging 설정 (Uvicorn 등 외부 라이브러리 로그 캡처용)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # 2. Structlog 프로세서 체인 구성
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # 운영 환경에서는 JSON 포맷터 사용
    if not settings.debug:
        processors.extend(
            [
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ]
        )
    else:
        # 디버그 모드에서는 개발자 친화적인 콘솔 출력
        processors.extend(
            [
                structlog.dev.ConsoleRenderer(),
            ]
        )

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
