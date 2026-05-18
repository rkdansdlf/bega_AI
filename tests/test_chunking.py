"""chunking.py 단위 테스트.

ChunkingConfig 기본값, resolve_chunking_config 유효성 검사,
legacy_window_chunks 경계값, smart_chunks 마크다운 구조 인식을 검증한다.
외부 의존성 없음.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from app.core.chunking import (
    ChunkingConfig,
    legacy_window_chunks,
    resolve_chunking_config,
    smart_chunks,
)


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


# ── TestChunkingConfigDefaults ────────────────────────────────────────────────

class TestChunkingConfigDefaults:
    def test_default_target_chars(self):
        assert ChunkingConfig().target_chars == 650

    def test_default_max_chars(self):
        assert ChunkingConfig().max_chars == 900

    def test_default_min_chars(self):
        assert ChunkingConfig().min_chars == 180

    def test_default_overlap_chars(self):
        assert ChunkingConfig().overlap_chars == 80

    def test_frozen_raises_on_setattr(self):
        cfg = ChunkingConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.target_chars = 999  # type: ignore[misc]


# ── TestResolveChunkingConfig ─────────────────────────────────────────────────

class TestResolveChunkingConfig:
    def test_defaults_returned_when_no_args(self):
        cfg = resolve_chunking_config()
        assert cfg.target_chars == 650
        assert cfg.max_chars == 900

    def test_explicit_params_override_defaults(self):
        cfg = resolve_chunking_config(target_chars=200, max_chars=400, min_chars=50, overlap_chars=20)
        assert cfg.target_chars == 200
        assert cfg.max_chars == 400
        assert cfg.min_chars == 50
        assert cfg.overlap_chars == 20

    def test_settings_object_used_as_fallback(self):
        s = SimpleNamespace(
            rag_chunk_target_chars=300,
            rag_chunk_max_chars=500,
            rag_chunk_min_chars=100,
            rag_chunk_overlap_chars=30,
        )
        cfg = resolve_chunking_config(settings=s)
        assert cfg.target_chars == 300
        assert cfg.max_chars == 500

    def test_explicit_param_overrides_settings(self):
        s = SimpleNamespace(
            rag_chunk_target_chars=300,
            rag_chunk_max_chars=500,
            rag_chunk_min_chars=100,
            rag_chunk_overlap_chars=30,
        )
        cfg = resolve_chunking_config(settings=s, target_chars=250)
        assert cfg.target_chars == 250

    def test_target_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="target_chars"):
            resolve_chunking_config(target_chars=1000, max_chars=500)

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="min_chars"):
            resolve_chunking_config(target_chars=100, max_chars=200, min_chars=300)

    def test_overlap_equal_to_max_raises(self):
        with pytest.raises(ValueError, match="overlap_chars"):
            resolve_chunking_config(target_chars=100, max_chars=200, min_chars=50, overlap_chars=200)

    def test_target_zero_raises(self):
        with pytest.raises(ValueError):
            resolve_chunking_config(target_chars=0, max_chars=200, min_chars=50)

    def test_overlap_negative_raises(self):
        with pytest.raises(ValueError, match="overlap_chars"):
            resolve_chunking_config(target_chars=100, max_chars=200, min_chars=50, overlap_chars=-1)

    def test_returns_chunking_config_instance(self):
        assert isinstance(resolve_chunking_config(), ChunkingConfig)


# ── TestLegacyWindowChunks ────────────────────────────────────────────────────

class TestLegacyWindowChunks:
    def test_empty_string_returns_empty(self):
        assert legacy_window_chunks("") == []

    def test_whitespace_only_returns_empty(self):
        assert legacy_window_chunks("   \n  ") == []

    def test_short_text_returns_single_chunk(self):
        result = legacy_window_chunks("가나다", max_chars=100)
        assert result == ["가나다"]

    def test_exact_max_chars_not_split(self):
        text = "A" * 100
        result = legacy_window_chunks(text, max_chars=100)
        assert len(result) == 1

    def test_long_text_produces_multiple_chunks(self):
        text = "가" * 2000
        result = legacy_window_chunks(text, max_chars=500, overlap_chars=50)
        assert len(result) > 1

    def test_all_chunks_respect_max_chars(self):
        text = "B" * 3000
        result = legacy_window_chunks(text, max_chars=500, overlap_chars=50)
        assert all(len(c) <= 500 for c in result)

    def test_overlap_creates_shared_content(self):
        text = "X" * 200
        result = legacy_window_chunks(text, max_chars=100, overlap_chars=20)
        assert len(result) >= 2
        assert result[0][-20:] == result[1][:20]

    def test_zero_overlap_no_shared_content(self):
        text = "Y" * 200
        result = legacy_window_chunks(text, max_chars=100, overlap_chars=0)
        assert len(result) >= 2
        total_chars = sum(len(c) for c in result)
        assert total_chars == 200

    def test_returns_list_of_strings(self):
        result = legacy_window_chunks("텍스트", max_chars=100)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)


# ── TestSmartChunks ───────────────────────────────────────────────────────────

class TestSmartChunks:
    def test_empty_string_returns_empty(self):
        assert smart_chunks("") == []

    def test_whitespace_only_returns_empty(self):
        assert smart_chunks("   \n  ") == []

    def test_newlines_only_returns_empty(self):
        assert smart_chunks("\n\n\n") == []

    def test_short_text_returned_as_single_chunk(self):
        text = "짧은 한국어 텍스트"
        result = smart_chunks(text, max_chars=900)
        assert result == [text]

    def test_returns_list_of_strings(self):
        result = smart_chunks("테스트 텍스트", max_chars=900)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_all_chunks_respect_max_chars(self):
        long_text = ("가나다라마바사아자차카타파하 " * 100).strip()
        result = smart_chunks(long_text, target_chars=150, max_chars=200)
        assert all(len(c) <= 200 for c in result)

    def test_markdown_heading_creates_chunk_boundary(self):
        section_a = "내용 " * 40  # ~200 chars
        section_b = "다른 내용 " * 40  # ~240 chars
        md = f"## 섹션1\n{section_a}\n\n## 섹션2\n{section_b}"
        # max_chars=300 forces split (total ~458 > 300)
        result = smart_chunks(md, settings=_settings(target=200, max_chars=300, min_chars=50, overlap=20))
        assert len(result) >= 2

    def test_table_block_kept_as_atomic_chunk(self):
        table = "| 항목 | 설명 |\n| --- | --- |\n| A | B |\n| C | D |"
        prefix = "설명 " * 5
        text = f"{prefix}\n\n{table}"
        result = smart_chunks(text, max_chars=900)
        assert any("|" in chunk for chunk in result)

    def test_list_block_kept_as_atomic_chunk(self):
        lst = "- 첫 번째\n- 두 번째\n- 세 번째"
        prefix = "항목 목록:" * 5
        text = f"{prefix}\n\n{lst}"
        result = smart_chunks(text, max_chars=900)
        assert any("- 첫 번째" in chunk for chunk in result)

    def test_preserves_heading_table_and_list_boundaries(self):
        text = "\n\n".join([
            "# 규정 안내\n타이브레이크 제도는 동률 순위를 정할 때 상대 전적과 세부 기준을 함께 본다.",
            "| 항목 | 설명 |\n| --- | --- |\n| 적용 | 동률 순위 결정 |\n| 기준 | 상대 전적 |",
            "- 첫째 기준은 상대 전적이다.\n- 둘째 기준은 다득점이다.",
        ])
        result = smart_chunks(text, settings=_settings(target=70, min_chars=20))
        assert len(result) == 3
        assert result[0].startswith("# 규정 안내")
        assert "| 항목 | 설명 |" in result[1]
        assert result[2].startswith("- 첫째 기준은 상대 전적이다.")

    def test_sentence_split_for_long_paragraph(self):
        text = " ".join([
            "첫 번째 문장은 타이브레이크 절차의 개요를 설명하고 기준의 순서를 정리한다.",
            "두 번째 문장은 상대 전적과 다득점 여부처럼 실제 비교 항목을 이어서 설명한다.",
            "세 번째 문장은 예외 상황과 시즌 운영상 주의점을 덧붙여 맥락을 마무리한다.",
        ])
        result = smart_chunks(
            text,
            settings=_settings(target=70, max_chars=90, min_chars=30, overlap=12),
        )
        assert len(result) >= 2
        assert all(len(c) <= 90 for c in result)
        assert result[0].endswith(".")
        assert any("두 번째 문장" in chunk for chunk in result[1:])

    def test_hard_split_keeps_overlap(self):
        text = "A" * 220
        result = smart_chunks(
            text,
            settings=_settings(target=60, max_chars=80, min_chars=30, overlap=10),
        )
        assert len(result) >= 3
        assert result[0][-10:] == result[1][:10]

    def test_clean_paragraph_boundaries_no_overlap(self):
        paragraph_a = "A" * 70
        paragraph_b = "B" * 70
        result = smart_chunks(
            f"{paragraph_a}\n\n{paragraph_b}",
            settings=_settings(target=80, max_chars=90, min_chars=20, overlap=12),
        )
        assert result == [paragraph_a, paragraph_b]
