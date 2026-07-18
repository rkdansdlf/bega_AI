"""Request-model and deterministic AI stream contract export tests."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from app.contracts.stream_requests import ChatStreamRequest, CoachAnalyzeRequest
from scripts.export_ai_stream_contract import (
    CONTRACT_PATH,
    build_contract_document,
    render_contract_json,
)


def test_chat_request_preserves_current_extra_field_policy() -> None:
    request = ChatStreamRequest.model_validate(
        {
            "question": "테스트",
            "history": None,
            "client_marker": "legacy",
        }
    )

    assert request.question == "테스트"
    assert request.model_extra == {"client_marker": "legacy"}


def test_chat_request_types_history_and_style() -> None:
    request = ChatStreamRequest.model_validate(
        {
            "question": "테스트",
            "history": [{"role": "user", "content": "이전 질문"}],
            "style": "compact",
            "cache_bypass": True,
        }
    )

    assert request.history is not None
    assert request.history[0].role == "user"
    assert request.style == "compact"
    assert request.cache_bypass is True


def test_coach_request_backfills_legacy_fields() -> None:
    request = CoachAnalyzeRequest.model_validate(
        {
            "team_id": "HT",
            "analysisType": "game_review",
            "request_mode": "manual_detail",
        }
    )

    assert request.home_team_id == "HT"
    assert request.analysis_type == "game_review"


def test_coach_auto_brief_rejects_question_override() -> None:
    with pytest.raises(ValidationError, match="question_override"):
        CoachAnalyzeRequest.model_validate(
            {
                "home_team_id": "HT",
                "request_mode": "auto_brief",
                "question_override": "상세 분석",
            }
        )


def test_coach_request_trims_optional_signatures() -> None:
    request = CoachAnalyzeRequest.model_validate(
        {
            "home_team_id": "HT",
            "request_mode": "manual_detail",
            "starter_signature": " starter ",
            "lineup_signature": "  ",
            "expected_cache_key": " cache-key ",
        }
    )

    assert request.starter_signature == "starter"
    assert request.lineup_signature is None
    assert request.expected_cache_key == "cache-key"


def test_coach_router_uses_the_canonical_request_model() -> None:
    from app.routers.coach import AnalyzeRequest

    assert AnalyzeRequest is CoachAnalyzeRequest


def test_contract_document_contains_named_requests_and_event_union() -> None:
    document = build_contract_document()
    schemas = document["components"]["schemas"]

    assert document["openapi"] == "3.1.0"
    assert document["info"]["version"] == "2.1.0"
    assert "ChatStreamRequest" in schemas
    assert "CoachAnalyzeRequest" in schemas
    assert "AiStreamV2Event" in schemas
    assert "AiStreamHttpError" in schemas
    http_error = schemas["AiStreamHttpError"]
    assert set(http_error["required"]) == {"code", "message", "retryable"}
    assert http_error["additionalProperties"] is False
    discriminator = schemas["AiStreamV2Event"]["discriminator"]
    assert discriminator["propertyName"] == "type"
    assert set(discriminator["mapping"]) == {
        "chat.status",
        "chat.queue",
        "chat.message.delta",
        "chat.meta",
        "coach.status",
        "coach.preview.chunk",
        "coach.preview.reset",
        "coach.message.delta",
        "coach.meta",
        "stream.error",
        "stream.done",
    }


def test_contract_render_is_sorted_and_has_one_trailing_newline() -> None:
    rendered = render_contract_json(build_contract_document())

    assert rendered.endswith("\n")
    assert not rendered.endswith("\n\n")
    assert rendered.index('"components"') < rendered.index('"info"')


def test_committed_contract_is_current() -> None:
    expected = render_contract_json(build_contract_document())

    assert Path(CONTRACT_PATH).read_text(encoding="utf-8") == expected


def test_exporter_cli_imports_app_from_repository_root() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/export_ai_stream_contract.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
