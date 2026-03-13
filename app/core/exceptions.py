"""공통 예외 클래스 정의 모듈."""


class DBRetrievalError(Exception):
    """pgvector 유사도 검색 중 DB 연결 실패 또는 쿼리 실행 오류 시 발생합니다.

    similarity_search()에서 발생시키며, RAGPipeline이 포착하여
    LLM 일반지식 모드 폴백으로 전환합니다.
    """

    def __init__(self, message: str, cause: Exception) -> None:
        super().__init__(message)
        self.cause = cause
