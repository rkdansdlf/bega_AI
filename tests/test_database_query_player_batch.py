from unittest.mock import Mock

from app.tools.database_query import DatabaseQueryTool


def test_get_player_season_stats_batch_preserves_order_and_partial_miss() -> None:
    mock_connection = Mock()
    mock_cursor = Mock()
    mock_connection.cursor.return_value = mock_cursor
    mock_cursor.fetchall.side_effect = [
        [
            {
                "player_order": 0,
                "requested_name": "안현민",
                "resolved_player_name": "안현민",
                "team_name": "KT",
                "season_year": 2025,
                "plate_appearances": 210,
                "avg": 0.311,
                "obp": 0.392,
                "slg": 0.551,
                "ops": 0.943,
                "home_runs": 13,
                "doubles": 12,
                "walks": 24,
                "strikeouts": 39,
                "row_num": 1,
            },
            {
                "player_order": 1,
                "requested_name": "윤도현",
                "resolved_player_name": "윤도현",
                "team_name": "KIA",
                "season_year": 2025,
                "plate_appearances": 188,
                "avg": 0.287,
                "obp": 0.349,
                "slg": 0.448,
                "ops": 0.797,
                "home_runs": 7,
                "doubles": 14,
                "walks": 18,
                "strikeouts": 41,
                "row_num": 1,
            },
        ],
        [],
    ]

    tool = DatabaseQueryTool.__new__(DatabaseQueryTool)
    tool.connection = mock_connection

    results = tool.get_player_season_stats_batch(
        ["안현민", "윤도현", "김도영"],
        2025,
        "both",
    )

    assert [result["requested_player_name"] for result in results] == [
        "안현민",
        "윤도현",
        "김도영",
    ]
    assert results[0]["player_name"] == "안현민"
    assert results[1]["player_name"] == "윤도현"
    assert results[2]["player_name"] == "김도영"
    assert results[0]["found"] is True
    assert results[1]["found"] is True
    assert results[2]["found"] is False
    assert results[2]["batting_stats"] is None
    assert results[2]["pitching_stats"] is None
    assert results[0]["batch_lookup"] is True
    assert mock_cursor.execute.call_count == 2


def test_get_player_season_stats_batch_sets_error_on_query_failure() -> None:
    mock_connection = Mock()
    mock_cursor = Mock()
    mock_connection.cursor.return_value = mock_cursor
    mock_cursor.execute.side_effect = RuntimeError("db down")

    tool = DatabaseQueryTool.__new__(DatabaseQueryTool)
    tool.connection = mock_connection

    results = tool.get_player_season_stats_batch(["문동주", "김택연"], 2025, "pitching")

    assert len(results) == 2
    assert all(result["found"] is False for result in results)
    assert all(result["error"] == "db down" for result in results)
