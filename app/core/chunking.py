"""문장을 일정 길이로 나누는 텍스트 청킹 유틸리티."""

from typing import Iterable, List


def smart_chunks(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    if not text:
        return []
    if len(text) <= max_chars:
        return [text.strip()]

    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end]
        chunks.append(chunk.strip())

        # 끝에 도달했으면 종료
        if end == length:
            break

        # overlap 적용하여 다음 start 위치 계산
        new_start = end - overlap
        if new_start <= start:
            # 진전이 없으면 overlap 없이 계속
            start = end
        else:
            start = new_start
    return [c for c in chunks if c]
