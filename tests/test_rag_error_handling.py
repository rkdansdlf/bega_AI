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
    movement_type=None,
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
        movement_type=movement_type,
    )


def _make_dummy_conn(error_cls=None):
    """similarity_search에 전달할 더미 psycopg 연결 객체를 생성합니다.

    error_cls가 주어지면 cursor().execute()에서 해당 예외를 발생시킵니다.
    """
    mock_cursor = MagicMock()
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)
    if error_cls is not None:
        mock_cursor.execute = AsyncMock(side_effect=error_cls("mock db error"))
        mock_cursor.fetchall = AsyncMock(return_value=[])
        mock_cursor.fetchone = AsyncMock(return_value=None)
    else:
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[])
        mock_cursor.fetchone = AsyncMock(return_value=None)

    mock_conn = MagicMock(spec=psycopg.AsyncConnection)
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
                asyncio.run(similarity_search(conn, [0.1] * 768, limit=5))
        assert isinstance(exc_info.value.cause, psycopg.OperationalError)

    def test_interface_error_raises_db_retrieval_error(self):
        """InterfaceError → DBRetrievalError로 변환되어야 한다."""
        conn = _make_dummy_conn(psycopg.InterfaceError)
        with patch("app.core.retrieval._rag_chunks_exists", return_value=True):
            with pytest.raises(DBRetrievalError) as exc_info:
                asyncio.run(similarity_search(conn, [0.1] * 768, limit=5))
        assert isinstance(exc_info.value.cause, psycopg.InterfaceError)

    def test_query_canceled_raises_db_retrieval_error(self):
        """QueryCanceled → DBRetrievalError로 변환되어야 한다."""
        conn = _make_dummy_conn(psycopg.errors.QueryCanceled)
        with patch("app.core.retrieval._rag_chunks_exists", return_value=True):
            with pytest.raises(DBRetrievalError) as exc_info:
                asyncio.run(similarity_search(conn, [0.1] * 768, limit=5))
        assert isinstance(exc_info.value.cause, psycopg.errors.QueryCanceled)

    def test_undefined_table_returns_empty_list(self):
        """UndefinedTable은 기존처럼 예외 없이 [] 반환이어야 한다."""
        conn = _make_dummy_conn(psycopg.errors.UndefinedTable)
        with patch("app.core.retrieval._rag_chunks_exists", return_value=True):
            result = asyncio.run(similarity_search(conn, [0.1] * 768, limit=5))
        assert result == []


# ---------------------------------------------------------------------------
# Group B: RAGPipeline.retrieve() 플래그 설정
# ---------------------------------------------------------------------------


class TestRetrieveState:

    def _make_pipeline(self):
        """최소한의 RAGPipeline 인스턴스를 생성합니다."""
        from app.core.rag import RAGPipeline
        from app.config import get_settings

        settings = get_settings()
        mock_conn = MagicMock(spec=psycopg.Connection)
        pipeline = RAGPipeline(settings=settings, connection=mock_conn)
        return pipeline

    def test_retrieve_sets_request_state_on_db_error(self):
        """similarity_search가 DBRetrievalError를 발생시키면 request-local state에 기록되어야 한다."""
        pipeline = self._make_pipeline()
        retrieval_state: Dict[str, Any] = {}

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
                    result = await pipeline.retrieve(
                        "테스트 쿼리", retrieval_state=retrieval_state
                    )
            return result

        result = asyncio.run(run())
        assert result == []
        assert retrieval_state["error_type"] == "db_unavailable"
        assert retrieval_state["db_error"]

    def test_retrieve_returns_docs_on_success(self):
        """정상 검색 시 docs를 반환하고 request-local error state는 비어 있어야 한다."""
        pipeline = self._make_pipeline()
        fake_doc = {"id": 1, "title": "테스트", "content": "내용", "meta": {}}
        retrieval_state: Dict[str, Any] = {}

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
                    result = await pipeline.retrieve(
                        "테스트 쿼리", retrieval_state=retrieval_state
                    )
            return result

        result = asyncio.run(run())
        assert result == [fake_doc]
        assert retrieval_state == {}

    def test_retrieve_sets_request_state_on_embedding_error(self):
        """query embedding 실패는 db_unavailable/zero_hit가 아니라 embedding_failed로 기록한다."""
        pipeline = self._make_pipeline()
        retrieval_state: Dict[str, Any] = {}

        async def run():
            with patch(
                "app.core.rag.async_embed_query",
                new_callable=AsyncMock,
                side_effect=RuntimeError("embed down"),
            ):
                return await pipeline.retrieve(
                    "테스트 쿼리", retrieval_state=retrieval_state
                )

        result = asyncio.run(run())
        assert result == []
        assert retrieval_state["error_type"] == "embedding_failed"
        assert "embed down" in retrieval_state["embedding_error"]

    def test_retrieve_skips_hyde_by_default(self):
        """retrieve() 기본 경로는 HyDE를 사용하지 않아야 한다."""
        pipeline = self._make_pipeline()
        fake_doc = {"id": 2, "title": "기본", "content": "기본 내용", "meta": {}}

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
                    with patch.object(
                        pipeline,
                        "_generate",
                        new_callable=AsyncMock,
                        return_value="가설 문서",
                    ) as mock_generate:
                        result = await pipeline.retrieve("테스트 쿼리")
                        mock_generate.assert_not_called()
            return result

        result = asyncio.run(run())
        assert result == [fake_doc]

    def test_retrieve_enables_hyde_only_when_requested(self):
        """use_hyde=True일 때만 가설 문서 생성이 호출되어야 한다."""
        pipeline = self._make_pipeline()
        fake_doc = {"id": 3, "title": "HyDE", "content": "HyDE 내용", "meta": {}}

        async def run():
            with patch.object(
                pipeline,
                "_generate",
                new_callable=AsyncMock,
                return_value="가설 문서 결과",
            ) as mock_generate:
                with patch(
                    "app.core.rag.async_embed_query",
                    new_callable=AsyncMock,
                    return_value=[0.1] * 768,
                ) as mock_embed:
                    with patch(
                        "app.core.rag.similarity_search",
                        return_value=[fake_doc],
                    ):
                        result = await pipeline.retrieve("테스트 쿼리", use_hyde=True)
            return result, mock_generate, mock_embed

        result, mock_generate, mock_embed = asyncio.run(run())
        assert result == [fake_doc]
        mock_generate.assert_awaited_once()
        mock_embed.assert_awaited_once()
        assert mock_embed.await_args.args[0] == "가설 문서 결과"


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

    def test_general_conversation_keeps_volatile_rules_out_of_hardcoded_answers(self):
        pipeline = self._make_pipeline()

        async def run():
            return await pipeline._handle_general_conversation("FA 자격 조건은?")

        result = asyncio.run(run())
        assert result["strategy"] == "conversation_handler"
        assert "프로 경력 9년" not in result["answer"]
        assert "보상선수" not in result["answer"]

    def test_run_uses_db_down_strategy_when_retrieval_error_set(self):
        """DB retrieval error가 request-local state에 있으면 DB 장애 전략을 반환해야 한다."""
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
                    with patch(
                        "app.core.rag.record_retrieval_event"
                    ) as mock_event:
                        with patch.object(
                            pipeline,
                            "_generate",
                            new_callable=AsyncMock,
                            return_value="DB 장애로 인한 일반 지식 기반 답변입니다.",
                        ):
                            result = await pipeline.run("2024년 홈런왕은 누구야?")
                            await asyncio.sleep(0)
            return result, mock_event

        result, mock_event = asyncio.run(run())
        assert result["strategy"] == "llm_knowledge_db_unavailable"
        assert result["citations"] == []
        assert result["answer"].startswith("⚠️ 현재 KBO 통계 DB에 일시적으로 접속할 수 없어")
        assert "DB 장애로 인한 일반 지식 기반 답변입니다." in result["answer"]
        assert mock_event.call_args.kwargs["success"] is False
        assert mock_event.call_args.kwargs["error_type"] == "db_unavailable"

    def test_run_uses_embedding_failed_strategy_when_query_embedding_fails(self):
        """query embedding 실패는 zero-hit가 아니라 embedding_failed 이벤트와 전략으로 분리한다."""
        pipeline = self._make_pipeline()

        async def run():
            with patch(
                "app.core.rag.async_embed_query",
                new_callable=AsyncMock,
                side_effect=RuntimeError("embedding provider down"),
            ):
                with patch("app.core.rag.record_retrieval_event") as mock_event:
                    with patch.object(
                        pipeline,
                        "_generate",
                        new_callable=AsyncMock,
                        return_value="제한적인 참고 답변입니다.",
                    ):
                        result = await pipeline.run("2024년 홈런왕은 누구야?")
                        await asyncio.sleep(0)
            return result, mock_event

        result, mock_event = asyncio.run(run())
        assert result["strategy"] == "llm_knowledge_embedding_failed"
        assert result["citations"] == []
        assert result["answer"].startswith("검색용 임베딩 생성에 실패해")
        assert mock_event.call_args.kwargs["success"] is False
        assert mock_event.call_args.kwargs["error_type"] == "embedding_failed"

    def test_run_ignores_stale_instance_retrieval_error(self):
        """run()은 과거 instance error flag가 아니라 request-local state만 사용해야 한다."""
        pipeline = self._make_pipeline()
        pipeline._retrieval_error = "leftover error from previous request"

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
                            pipeline,
                            "_generate",
                            new_callable=AsyncMock,
                            return_value="답변",
                        ):
                            return await pipeline.run("테스트")

        result = asyncio.run(run())
        assert result["strategy"] == "rag_v3_enhanced"
        assert result["answer"].startswith("저장된 KBO 데이터에서는 관련 근거를 찾지 못했습니다.")

    def test_run_records_actual_fallback_filters_and_expanded_citation(self):
        """fallback이 필터를 완화하면 최초/실제 필터와 확장 citation을 남긴다."""
        pipeline = self._make_pipeline()
        fake_doc = {
            "id": 42,
            "title": "LG 2025",
            "content": "LG 2025 summary",
            "meta": {"topic_key": "kbo.team.2025.lg"},
            "source_table": "team_summary",
            "source_row_id": "team_id=LG|season_year=2025",
            "source_type": "kbo_db_table",
            "source_uri": "db:team_summary:team_id=LG|season_year=2025",
            "topic_key": "kbo.team.2025.lg",
            "similarity": 0.81,
            "combined_score": 0.05,
            "quality_score": 0.85,
            "valid_from": None,
            "valid_to": None,
        }
        calls: List[Dict[str, Any]] = []

        from types import SimpleNamespace

        entity_filter = SimpleNamespace(
            player_name=None,
            team_id="LG",
            season_year=2025,
            stat_type=None,
            position_type=None,
            award_type=None,
            game_date=None,
            movement_type=None,
        )
        search_strategy = {
            "entity_filter": entity_filter,
            "db_filters": {
                "source_table": "markdown_docs",
                "team_id": "LG",
                "season_year": 2025,
            },
            "search_limit": 5,
            "is_ranking_query": False,
        }

        async def fake_multi_query(*_args, **_kwargs):
            return []

        async def fake_retrieve(_query, *, filters=None, **_kwargs):
            calls.append(dict(filters or {}))
            if filters == {"season_year": 2025}:
                return [fake_doc]
            return []

        async def run():
            with patch("app.core.rag.enhance_search_strategy", return_value=search_strategy):
                with patch.object(
                    pipeline, "_is_statistical_query", return_value=False
                ):
                    with patch.object(
                        pipeline, "_is_general_conversation", return_value=False
                    ):
                        with patch.object(
                            pipeline, "_is_regulation_query", return_value=False
                        ):
                            with patch.object(pipeline, "_is_game_query", return_value=False):
                                with patch.object(
                                    pipeline,
                                    "_is_game_flow_narrative_query",
                                    return_value=False,
                                ):
                                    with patch.object(
                                        pipeline,
                                        "retrieve_with_multi_query",
                                        side_effect=fake_multi_query,
                                    ):
                                        with patch.object(
                                            pipeline,
                                            "retrieve",
                                            side_effect=fake_retrieve,
                                        ):
                                            with patch(
                                                "app.core.rag.record_retrieval_event"
                                            ) as mock_event:
                                                with patch.object(
                                                    pipeline,
                                                    "_generate",
                                                    new_callable=AsyncMock,
                                                    return_value="근거 기반 답변",
                                                ):
                                                    result = await pipeline.run(
                                                        "LG 요약", intent="stats_lookup"
                                                    )
                                                    await asyncio.sleep(0)
            return result, mock_event

        result, mock_event = asyncio.run(run())
        assert calls == [
            {"source_table": "markdown_docs", "team_id": "LG", "season_year": 2025},
            {"team_id": "LG", "season_year": 2025},
            {"season_year": 2025},
        ]
        metadata_filter = mock_event.call_args.kwargs["metadata_filter"]
        assert metadata_filter["original_filters"] == search_strategy["db_filters"]
        assert metadata_filter["actual_filters"] == {"season_year": 2025}
        assert metadata_filter["fallback_used"] is True
        assert metadata_filter["fallback_stage"] == "without_team_id"
        citation = result["citations"][0]
        assert citation["id"] == 42
        assert citation["title"] == "LG 2025"
        assert citation["source_table"] == "team_summary"
        assert citation["source_row_id"] == "team_id=LG|season_year=2025"
        assert citation["source_uri"] == "db:team_summary:team_id=LG|season_year=2025"
        assert citation["similarity"] == 0.81
        assert citation["combined_score"] == 0.05
        assert citation["quality_score"] == 0.85
        assert citation["topic_key"] == "kbo.team.2025.lg"

    def test_default_season_year_prefers_setting_then_current_year(self):
        """기준 시즌은 설정값을 우선하고, 없으면 현재 연도로 폴백해야 한다."""
        from app.config import Settings
        from app.core.rag import _resolve_default_season_year

        assert (
            _resolve_default_season_year(Settings(default_kbo_season_year=2031))
            == 2031
        )

        with patch("app.core.rag.datetime") as mock_datetime:
            mock_datetime.now.return_value.year = 2042
            assert (
                _resolve_default_season_year(Settings(default_kbo_season_year=None))
                == 2042
            )

    def test_process_and_enrich_docs_excludes_sample_filtered_docs_from_raw_docs(self):
        """표본 부족 stat doc은 raw_docs에도 남지 않아야 한다."""
        pipeline = self._make_pipeline()
        low_sample_doc = {
            "id": 99,
            "source_table": "player_season_pitching",
            "content_hash": "low-sample-hash",
            "meta": {
                "source_row_id": "player_id=low|season_year=2026",
                "player_name": "표본부족",
                "innings_pitched": 1,
                "games_started": 0,
                "era": 5.0,
                "whip": 1.5,
            },
        }

        result = asyncio.run(pipeline._process_and_enrich_docs([low_sample_doc], 2026))

        assert result["raw_docs"] == []
        assert result["pitchers"] == []
        assert any("표본 부족" in warning for warning in result["warnings"])


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

    def test_zero_hit_prefix_accepts_any_stored_prefix(self):
        """이미 '저장된'으로 시작하는 답변에는 prefix를 중복 적용하지 않는다."""
        from app.core.rag import _ensure_zero_hit_answer_prefix

        answer = "저장된 데이터 기준으로 관련 기록이 없습니다."

        assert _ensure_zero_hit_answer_prefix(answer) == answer

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
                    with patch(
                        "app.core.rag.record_retrieval_event"
                    ) as mock_event:
                        with patch.object(
                            pipeline, "_generate", side_effect=mock_generate
                        ):
                            result = await pipeline.run(
                                "없는선수123 2099년 타율은?",
                                intent="stats_lookup",
                            )
                            await asyncio.sleep(0)
            return result, mock_event

        result, mock_event = asyncio.run(run())

        # LLM에 전달된 마지막 user 메시지에 zero-hit 가이드가 포함되어야 함
        user_content = captured_messages["messages"][-1]["content"]
        assert "가능한 원인" in user_content or "검색 결과 없음" in user_content
        assert result["answer"].startswith("저장된 KBO 데이터에서는 관련 근거를 찾지 못했습니다.")
        assert mock_event.call_args.kwargs["success"] is False
        assert mock_event.call_args.kwargs["error_type"] == "zero_hit"
