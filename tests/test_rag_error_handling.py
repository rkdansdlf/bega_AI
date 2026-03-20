"""RAG 파이프라인 예외 처리 및 폴백 전략 테스트."""

import asyncio
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import psycopg
import psycopg.errors
import pytest

from app.core.exceptions import DBRetrievalError
from app.core.retrieval import similarity_search
from app.core.context_formatter import ContextFormatter

# ---------------------------------------------------------------------------
# 헬퍼 픽스처
# ---------------------------------------------------------------------------


def _make_entity_filter(
    player_name=None,
    team_id=None,
    season_year=None,
    stat_type=None,
    position_type=None,
    award_type=None,
    game_date=None,
):
    """EntityFilter 형태의 SimpleNamespace를 생성합니다."""
    from types import SimpleNamespace

    return SimpleNamespace(
        player_name=player_name,
        team_id=team_id,
        season_year=season_year,
        stat_type=stat_type,
        position_type=position_type,
        award_type=award_type,
        game_date=game_date,
    )


def _make_dummy_conn(error_cls=None):
    """similarity_search에 전달할 더미 psycopg 연결 객체를 생성합니다.

    error_cls가 주어지면 cursor().execute()에서 해당 예외를 발생시킵니다.
    """
    mock_cursor = MagicMock()
    if error_cls is not None:
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.execute = MagicMock(side_effect=error_cls("mock db error"))
    else:
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.execute = MagicMock()
        mock_cursor.fetchall = MagicMock(return_value=[])

    mock_conn = MagicMock(spec=psycopg.Connection)
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    return mock_conn


# ---------------------------------------------------------------------------
# Group A: similarity_search() 예외 분류
# ---------------------------------------------------------------------------


class TestSimilaritySearchExceptions:

    def test_operational_error_raises_db_retrieval_error(self):
        """OperationalError → DBRetrievalError로 변환되어야 한다."""
        conn = _make_dummy_conn(psycopg.OperationalError)
        # _rag_chunks_exists() 가 True를 반환하도록 패치
        with patch("app.core.retrieval._rag_chunks_exists", return_value=True):
            with pytest.raises(DBRetrievalError) as exc_info:
                similarity_search(conn, [0.1] * 768, limit=5)
        assert isinstance(exc_info.value.cause, psycopg.OperationalError)

    def test_interface_error_raises_db_retrieval_error(self):
        """InterfaceError → DBRetrievalError로 변환되어야 한다."""
        conn = _make_dummy_conn(psycopg.InterfaceError)
        with patch("app.core.retrieval._rag_chunks_exists", return_value=True):
            with pytest.raises(DBRetrievalError) as exc_info:
                similarity_search(conn, [0.1] * 768, limit=5)
        assert isinstance(exc_info.value.cause, psycopg.InterfaceError)

    def test_undefined_table_returns_empty_list(self):
        """UndefinedTable은 기존처럼 예외 없이 [] 반환이어야 한다."""
        conn = _make_dummy_conn(psycopg.errors.UndefinedTable)
        with patch("app.core.retrieval._rag_chunks_exists", return_value=True):
            result = similarity_search(conn, [0.1] * 768, limit=5)
        assert result == []


# ---------------------------------------------------------------------------
# Group B: RAGPipeline.retrieve() 플래그 설정
# ---------------------------------------------------------------------------


class TestRetrieveFlag:

    def _make_pipeline(self):
        """최소한의 RAGPipeline 인스턴스를 생성합니다."""
        from app.core.rag import RAGPipeline
        from app.config import get_settings

        settings = get_settings()
        mock_conn = MagicMock(spec=psycopg.Connection)
        pipeline = RAGPipeline(settings=settings, connection=mock_conn)
        return pipeline

    def test_retrieve_sets_retrieval_error_flag_on_db_error(self):
        """similarity_search가 DBRetrievalError를 발생시키면 _retrieval_error가 설정되어야 한다."""
        pipeline = self._make_pipeline()

        async def run():
            with patch(
                "app.core.rag.async_embed_query",
                new_callable=AsyncMock,
                return_value=[0.1] * 768,
            ):
                with patch(
                    "app.core.rag.similarity_search",
                    side_effect=DBRetrievalError(
                        "mock", cause=psycopg.OperationalError("err")
                    ),
                ):
                    result = await pipeline.retrieve("테스트 쿼리")
            return result

        result = asyncio.run(run())
        assert result == []
        assert pipeline._retrieval_error is not None
        assert len(pipeline._retrieval_error) > 0

    def test_retrieve_returns_docs_on_success(self):
        """정상 검색 시 docs를 반환하고 _retrieval_error는 None이어야 한다."""
        pipeline = self._make_pipeline()
        fake_doc = {"id": 1, "title": "테스트", "content": "내용", "meta": {}}

        async def run():
            with patch(
                "app.core.rag.async_embed_query",
                new_callable=AsyncMock,
                return_value=[0.1] * 768,
            ):
                with patch(
                    "app.core.rag.similarity_search",
                    return_value=[fake_doc],
                ):
                    result = await pipeline.retrieve("테스트 쿼리")
            return result

        result = asyncio.run(run())
        assert result == [fake_doc]
        assert pipeline._retrieval_error is None


# ---------------------------------------------------------------------------
# Group C: RAGPipeline.run() DB-down 경로
# ---------------------------------------------------------------------------


class TestRunDbDownPath:

    def _make_pipeline(self):
        from app.core.rag import RAGPipeline
        from app.config import get_settings

        settings = get_settings()
        mock_conn = MagicMock(spec=psycopg.Connection)
        pipeline = RAGPipeline(settings=settings, connection=mock_conn)
        return pipeline

    def test_run_uses_db_down_strategy_when_retrieval_error_set(self):
        """_retrieval_error가 설정되면 run()은 llm_knowledge_db_unavailable 전략을 반환해야 한다."""
        pipeline = self._make_pipeline()

        async def run():
            with patch(
                "app.core.rag.async_embed_query",
                new_callable=AsyncMock,
                return_value=[0.1] * 768,
            ):
                with patch(
                    "app.core.rag.similarity_search",
                    side_effect=DBRetrievalError(
                        "mock", cause=psycopg.OperationalError("conn failed")
                    ),
                ):
                    with patch.object(
                        pipeline,
                        "_generate",
                        new_callable=AsyncMock,
                        return_value="DB 장애로 인한 일반 지식 기반 답변입니다.",
                    ):
                        return await pipeline.run("2024년 홈런왕은 누구야?")

        result = asyncio.run(run())
        assert result["strategy"] == "llm_knowledge_db_unavailable"
        assert result["citations"] == []
        assert "DB 장애로 인한 일반 지식 기반 답변입니다." in result["answer"]

    def test_run_resets_retrieval_error_flag_at_start(self):
        """run() 시작 시 _retrieval_error가 None으로 초기화되어야 한다."""
        pipeline = self._make_pipeline()
        pipeline._retrieval_error = "leftover error from previous request"

        captured_flag = {}

        original_process = pipeline._process_and_enrich_docs

        async def spy_process(docs, year):
            captured_flag["value"] = pipeline._retrieval_error
            return await original_process(docs, year)

        async def run():
            with patch(
                "app.core.rag.async_embed_query",
                new_callable=AsyncMock,
                return_value=[0.1] * 768,
            ):
                with patch(
                    "app.core.rag.similarity_search",
                    return_value=[],
                ):
                    with patch.object(
                        pipeline, "_process_and_enrich_docs", spy_process
                    ):
                        with patch.object(
                            pipeline,
                            "_generate",
                            new_callable=AsyncMock,
                            return_value="답변",
                        ):
                            return await pipeline.run("테스트")

        asyncio.run(run())
        # _retrieval_error는 run() 시작 시 None으로 리셋되어야 함
        assert captured_flag.get("value") is None


# ---------------------------------------------------------------------------
# Group D: ContextFormatter.format_zero_hit_guidance()
# ---------------------------------------------------------------------------


class TestFormatZeroHitGuidance:

    def setup_method(self):
        self.formatter = ContextFormatter()

    def test_player_name_included_in_guidance(self):
        """player_name이 있으면 가이드 텍스트에 이름이 포함되어야 한다."""
        ef = _make_entity_filter(player_name="김도영", season_year=2024)
        result = self.formatter.format_zero_hit_guidance("김도영 타율", ef, 2024, {})
        assert "김도영" in result
        assert "가능한 원인" in result
        assert "대안적 접근" in result

    def test_adjacent_year_suggestions_present(self):
        """player_name이 있으면 year-1, year+1 대안 연도가 포함되어야 한다."""
        ef = _make_entity_filter(player_name="이정후", season_year=2023)
        result = self.formatter.format_zero_hit_guidance("이정후 OPS", ef, 2023, {})
        assert "2022" in result  # year - 1
        assert "2024" in result  # year + 1

    def test_future_year_message(self):
        """미래 연도(>2025)이면 '아직 수집되지 않았을 수 있음' 메시지가 있어야 한다."""
        ef = _make_entity_filter(season_year=2030)
        result = self.formatter.format_zero_hit_guidance("2030년 홈런왕", ef, 2030, {})
        assert "아직 수집되지 않았을 수 있음" in result

    def test_no_entities_shows_fallback_label(self):
        """엔티티가 없으면 '특정 조건 없음' 문구가 포함되어야 한다."""
        ef = _make_entity_filter()
        result = self.formatter.format_zero_hit_guidance("야구 얘기", ef, 2024, {})
        assert "특정 조건 없음" in result

    def test_returns_non_empty_string(self):
        """어떤 입력이든 비어 있지 않은 문자열을 반환해야 한다."""
        ef = _make_entity_filter(team_id="LG", season_year=2025)
        result = self.formatter.format_zero_hit_guidance("LG 성적", ef, 2025, {})
        assert isinstance(result, str)
        assert len(result) > 50


# ---------------------------------------------------------------------------
# Group E: run()의 zero-hit 컨텍스트 경로
# ---------------------------------------------------------------------------


class TestRunZeroHitContext:

    def _make_pipeline(self):
        from app.core.rag import RAGPipeline
        from app.config import get_settings

        settings = get_settings()
        mock_conn = MagicMock(spec=psycopg.Connection)
        pipeline = RAGPipeline(settings=settings, connection=mock_conn)
        return pipeline

    def test_run_injects_zero_hit_guidance_into_prompt(self):
        """docs가 [] 이면 format_zero_hit_guidance 결과가 LLM 프롬프트에 포함되어야 한다."""
        pipeline = self._make_pipeline()
        captured_messages = {}

        async def mock_generate(messages):
            captured_messages["messages"] = messages
            return "가이드 기반 답변"

        async def run():
            with patch(
                "app.core.rag.async_embed_query",
                new_callable=AsyncMock,
                return_value=[0.1] * 768,
            ):
                with patch("app.core.rag.similarity_search", return_value=[]):
                    with patch.object(pipeline, "_generate", side_effect=mock_generate):
                        return await pipeline.run(
                            "없는선수123 2099년 타율은?",
                            intent="stats_lookup",
                        )

        result = asyncio.run(run())

        # LLM에 전달된 마지막 user 메시지에 zero-hit 가이드가 포함되어야 함
        user_content = captured_messages["messages"][-1]["content"]
        assert "가능한 원인" in user_content or "검색 결과 없음" in user_content
