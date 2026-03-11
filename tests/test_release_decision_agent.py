from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.agents.release_decision_agent import (
    ReleaseDecisionDraft,
    ResponsesReleaseDecisionAgent,
    WorkspaceDocumentTools,
    _extract_json_object,
    render_release_decision_markdown,
)


def test_extract_json_object_accepts_code_fence() -> None:
    payload = _extract_json_object(
        """```json
        {"decision":"GO","summary":"ok"}
        ```"""
    )
    assert payload["decision"] == "GO"


def test_workspace_document_tools_search_and_read(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    doc_path = docs_dir / "gate.md"
    doc_path.write_text(
        "# Gate\nPrediction Stage 2는 표본 부족으로 Pending 상태입니다.\n",
        encoding="utf-8",
    )

    tools = WorkspaceDocumentTools(tmp_path, ["docs"])
    listed = tools.list_documents("docs/*.md")
    assert listed["matches"] == ["docs/gate.md"]

    searched = tools.search_documents("표본 부족", path_prefix="docs")
    assert searched["results"][0]["path"] == "docs/gate.md"

    read = tools.read_document("docs/gate.md", start_line=2, max_lines=1)
    assert read["excerpt"][0]["text"] == "Prediction Stage 2는 표본 부족으로 Pending 상태입니다."


def test_workspace_document_tools_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (tmp_path / "docs" / "gate.md").write_text("# Gate\n", encoding="utf-8")

    tools = WorkspaceDocumentTools(tmp_path, ["../../", "docs"])
    assert all(str(path).startswith(str(tmp_path)) for path in tools.allowed_roots)

    try:
        tools.read_document("../outside.md")
    except ValueError as exc:
        assert "escapes workspace" in str(exc)
    else:
        raise AssertionError("expected traversal path to be rejected")


def test_render_release_decision_markdown_contains_key_sections() -> None:
    draft = ReleaseDecisionDraft.model_validate(
        {
            "title": "Stage 2 Decision",
            "decision": "NO_GO",
            "summary": "표본 부족으로 보류합니다.",
            "blockers": ["24시간 모니터링 미완료"],
            "risks": ["운영 데이터 부족"],
            "next_actions": ["체크포인트 완료"],
            "evidence": [
                {
                    "claim": "모든 체크포인트가 Pending",
                    "source": "docs/qa/prediction_rollout_gate_latest.md",
                    "excerpt": "모든 체크포인트가 Pending",
                },
                {
                    "claim": "Stage2 Recommendation이 NO-GO",
                    "source": "docs/qa/prediction_stage1_sync_summary_latest.md",
                    "excerpt": "추천 결정: NO-GO (보류)",
                },
            ],
            "confidence": "medium",
        }
    )
    markdown = render_release_decision_markdown(draft)
    assert "## Evidence" in markdown
    assert "`NO_GO`" in markdown


class _FakeResponsesAPI:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = _FakeResponsesAPI(responses)


def test_agent_executes_tool_loop_and_parses_final_output(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs" / "qa"
    docs_dir.mkdir(parents=True)
    (docs_dir / "decision.md").write_text(
        "# Decision\n- 자동 판정: `NO-GO (보류)`\n",
        encoding="utf-8",
    )
    first_response = SimpleNamespace(
        id="resp_1",
        output=[
            SimpleNamespace(
                type="function_call",
                name="read_document",
                arguments=json.dumps({"path": "docs/qa/decision.md"}),
                call_id="call_1",
            )
        ],
        output_text="",
    )
    final_response = SimpleNamespace(
        id="resp_2",
        output=[],
        output_text=json.dumps(
            {
                "title": "Prediction Decision",
                "decision": "NO_GO",
                "summary": "문서상 보류입니다.",
                "blockers": ["자동 판정이 NO-GO"],
                "risks": ["운영 윈도우 부족"],
                "next_actions": ["체크포인트 완료"],
                "evidence": [
                    {
                        "claim": "자동 판정이 NO-GO",
                        "source": "docs/qa/decision.md",
                        "excerpt": "- 자동 판정: `NO-GO (보류)`",
                    },
                    {
                        "claim": "문서가 rollout 결정을 다룸",
                        "source": "docs/qa/decision.md",
                        "excerpt": "# Decision",
                    },
                ],
                "confidence": "high",
            },
            ensure_ascii=False,
        ),
    )
    client = _FakeClient([first_response, final_response])
    agent = ResponsesReleaseDecisionAgent(
        workspace_root=tmp_path,
        allowed_roots=["docs"],
        client=client,
        model="gpt-test",
    )
    result = agent.draft(
        scenario="prediction_stage2",
        task_prompt="Draft the current decision.",
        seed_paths=["docs/qa/decision.md"],
    )
    assert result.draft.decision == "NO_GO"
    assert len(result.tool_trace) == 1
    assert client.responses.calls[1]["previous_response_id"] == "resp_1"
