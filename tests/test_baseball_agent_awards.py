from types import SimpleNamespace

from app.agents.baseball_agent import BaseballStatisticsAgent
from app.agents.chat_intent_router import ChatIntentRouter
from app.agents.tool_caller import ToolResult
from app.tools.database_query import DatabaseQueryTool


def _build_agent() -> BaseballStatisticsAgent:
    agent = BaseballStatisticsAgent.__new__(BaseballStatisticsAgent)
    agent.fast_path_enabled = True
    agent.fast_path_scope = "all"
    agent.fast_path_tool_cap = 2
    agent._detect_team_alias_from_query = lambda query: None
    agent.chat_intent_router = ChatIntentRouter(
        resolve_reference_year=agent._resolve_reference_year,
        detect_team_alias=agent._detect_team_alias_from_query,
        resolve_award_query_type=agent._resolve_award_query_type,
        build_team_tool_calls=lambda query, team_name, season_year: [],
        fast_path_enabled=agent.fast_path_enabled,
        fast_path_scope=agent.fast_path_scope,
    )
    return agent


def test_reference_fast_path_routes_mvp_query_to_awards() -> None:
    agent = _build_agent()
    entity_filter = SimpleNamespace(
        season_year=2025,
        team_id=None,
        player_name=None,
        award_type="mvp",
    )

    plan = agent._build_reference_fast_path_plan(
        "25년 mvp 선수가 누구야", entity_filter
    )

    assert plan is not None
    assert plan["intent"] == "season_result_lookup"
    assert len(plan["tool_calls"]) == 1
    assert plan["tool_calls"][0].tool_name == "get_award_winners"
    assert plan["tool_calls"][0].parameters == {"year": 2025, "award_type": "mvp"}


def test_resolve_award_query_type_prefers_korean_series_mvp() -> None:
    agent = _build_agent()
    entity_filter = SimpleNamespace(award_type="mvp")

    resolved = agent._resolve_award_query_type(
        "2025 한국시리즈 MVP는 누구야?", entity_filter
    )

    assert resolved == "korean_series_mvp"


def test_resolve_award_query_type_handles_korean_spelling() -> None:
    agent = _build_agent()
    entity_filter = SimpleNamespace(award_type="mvp")

    resolved = agent._resolve_award_query_type(
        "2025 한국시리즈 엠브이피는 누구야?", entity_filter
    )

    assert resolved == "korean_series_mvp"


def test_build_structured_deterministic_answer_for_single_award() -> None:
    agent = _build_agent()
    tool_results = [
        ToolResult(
            success=True,
            data={
                "year": 2025,
                "award_type": "mvp",
                "awards": [
                    {
                        "year": 2025,
                        "award_type": "mvp",
                        "player_name": "폰세",
                        "team_name": None,
                        "position": "P",
                    }
                ],
                "found": True,
            },
            message="2025년 MVP 수상자는 폰세입니다.",
        )
    ]

    answer = agent._build_structured_deterministic_answer(
        "25년 mvp 선수가 누구야", tool_results
    )

    assert answer is not None
    assert "2025년 KBO MVP 수상자는 폰세" in answer
    assert "| MVP | 폰세 | - | P |" in answer


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, query, params=None):
        self.query = query
        self.params = params

    def fetchall(self):
        return self.rows

    def close(self):
        return None


class _FakeConnection:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self, row_factory=None):
        return _FakeCursor(self.rows)


def test_database_query_awards_dedupes_duplicate_rows() -> None:
    tool = DatabaseQueryTool.__new__(DatabaseQueryTool)
    tool.connection = _FakeConnection(
        [
            {
                "player_name": "폰세",
                "award_type": "MVP",
                "award_year": 2025,
                "position": "P",
                "team_name": None,
            },
            {
                "player_name": "폰세",
                "award_type": "MVP",
                "award_year": 2025,
                "position": "P",
                "team_name": None,
            },
        ]
    )
    tool._table_columns_cache = {}
    tool._get_table_columns = lambda table_name: {"award_year", "position"}

    result = DatabaseQueryTool.get_award_winners(tool, 2025, "mvp")

    assert result["found"] is True
    assert len(result["awards"]) == 1
    assert result["awards"][0]["player_name"] == "폰세"
    assert result["awards"][0]["award_type"] == "mvp"
