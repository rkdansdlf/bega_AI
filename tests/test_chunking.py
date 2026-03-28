from __future__ import annotations

from types import SimpleNamespace

from app.core.chunking import smart_chunks


def _settings(
    *,
    target: int = 90,
    max_chars: int = 110,
    min_chars: int = 40,
    overlap: int = 10,
) -> SimpleNamespace:
    return SimpleNamespace(
        rag_chunk_target_chars=target,
        rag_chunk_max_chars=max_chars,
        rag_chunk_min_chars=min_chars,
        rag_chunk_overlap_chars=overlap,
    )


def test_smart_chunks_preserves_heading_table_and_list_boundaries() -> None:
    text = "\n\n".join(
        [
            "# 규정 안내\n타이브레이크 제도는 동률 순위를 정할 때 상대 전적과 세부 기준을 함께 본다.",
            "| 항목 | 설명 |\n| --- | --- |\n| 적용 | 동률 순위 결정 |\n| 기준 | 상대 전적 |",
            "- 첫째 기준은 상대 전적이다.\n- 둘째 기준은 다득점이다.",
        ]
    )

    chunks = smart_chunks(text, settings=_settings(target=70, min_chars=20))

    assert len(chunks) == 3
    assert chunks[0].startswith("# 규정 안내")
    assert "| 항목 | 설명 |" in chunks[1]
    assert chunks[2].startswith("- 첫째 기준은 상대 전적이다.")


def test_smart_chunks_prefers_sentence_fallback_for_long_paragraph() -> None:
    text = " ".join(
        [
            "첫 번째 문장은 타이브레이크 절차의 개요를 설명하고 기준의 순서를 정리한다.",
            "두 번째 문장은 상대 전적과 다득점 여부처럼 실제 비교 항목을 이어서 설명한다.",
            "세 번째 문장은 예외 상황과 시즌 운영상 주의점을 덧붙여 맥락을 마무리한다.",
        ]
    )

    chunks = smart_chunks(
        text,
        settings=_settings(target=70, max_chars=90, min_chars=30, overlap=12),
    )

    assert len(chunks) >= 2
    assert all(len(chunk) <= 90 for chunk in chunks)
    assert chunks[0].endswith(".")
    assert any("두 번째 문장" in chunk for chunk in chunks[1:])


def test_smart_chunks_hard_split_keeps_overlap() -> None:
    text = "A" * 220

    chunks = smart_chunks(
        text,
        settings=_settings(target=60, max_chars=80, min_chars=30, overlap=10),
    )

    assert len(chunks) >= 3
    assert chunks[0][-10:] == chunks[1][:10]


def test_smart_chunks_does_not_overlap_clean_paragraph_boundaries() -> None:
    paragraph_a = "A" * 70
    paragraph_b = "B" * 70

    chunks = smart_chunks(
        f"{paragraph_a}\n\n{paragraph_b}",
        settings=_settings(target=80, max_chars=90, min_chars=20, overlap=12),
    )

    assert chunks == [paragraph_a, paragraph_b]
