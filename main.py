import logging
import sys

# 🔥 전체 추가
logging.basicConfig(
    level=logging.INFO,  # 또는 DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)],
    force=True
)

# 특정 모듈 로거도 명시적으로 설정
logging.getLogger("app.agents").setLevel(logging.DEBUG)
logging.getLogger("app").setLevel(logging.DEBUG)

from app import app  # noqa: F401

