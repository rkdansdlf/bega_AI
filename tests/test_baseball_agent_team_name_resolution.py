from app.agents.baseball_agent import BaseballStatisticsAgent


class _StubTeamResolver:
    def __init__(self) -> None:
        self.code_to_name = {
            "SSG": "SSG 랜더스",
            "SK": "SK 와이번스",
            "KIA": "KIA 타이거즈",
        }
        self.name_to_canonical = {
            "SSG": "SSG",
            "SK": "SSG",
            "와이번스": "SSG",
            "기아": "KIA",
        }

    def display_name(self, team_code: str) -> str:
        return {
            "SSG": "SSG 랜더스",
            "SK": "SK 와이번스",
            "와이번스": "SSG 랜더스",
            "기아": "KIA 타이거즈",
        }.get(team_code, team_code)


class _StubDbQueryTool:
    def __init__(self) -> None:
        self.team_resolver = _StubTeamResolver()

    def get_team_name(self, team_code: str) -> str:
        return self.team_resolver.display_name(team_code)


def _build_agent() -> BaseballStatisticsAgent:
    agent = BaseballStatisticsAgent.__new__(BaseballStatisticsAgent)
    agent.db_query_tool = _StubDbQueryTool()
    agent._team_name_cache = None
    agent.chat_tool_result_max_items = 8
    agent.chat_tool_result_max_chars = 2200
    return agent


def test_load_team_name_mapping_merges_resolver_aliases() -> None:
    agent = _build_agent()

    mapping = agent._load_team_name_mapping()

    assert mapping["SK"] == "SK 와이번스"
    assert mapping["와이번스"] == "SSG 랜더스"
    assert mapping["기아"] == "KIA 타이거즈"


def test_convert_team_id_to_name_prefers_database_query_tool() -> None:
    agent = _build_agent()

    assert agent._convert_team_id_to_name("SK") == "SK 와이번스"


def test_convert_team_id_to_name_keeps_static_fallback_for_missing_tool_mapping() -> (
    None
):
    agent = _build_agent()

    assert agent._convert_team_id_to_name("HH") == "한화 이글스"


def test_serialize_tool_data_for_prompt_uses_resolver_backed_team_names() -> None:
    agent = _build_agent()

    serialized = agent._serialize_tool_data_for_prompt(
        {
            "home_team": "SK",
            "away_team": "HH",
            "winning_team": "SK",
        }
    )

    assert "SK 와이번스" in serialized
    assert "한화 이글스" in serialized
