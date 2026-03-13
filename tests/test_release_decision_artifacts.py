from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.agents.release_decision_agent import ReleaseDecisionRunResult
from app.agents.release_decision_artifacts import (
    ReleaseDecisionArtifactStore,
    ReleaseDecisionEvalCaseSummary,
    ReleaseDecisionEvaluateResponse,
)
from app.agents.release_decision_eval import ReleaseDecisionEvalResult


def _sample_run_result() -> ReleaseDecisionRunResult:
    return ReleaseDecisionRunResult.model_validate(
        {
            "scenario": "prediction_stage2",
            "model": "gpt-test",
            "task_prompt": "Draft the rollout decision.",
            "seed_paths": ["docs/qa/prediction_rollout_gate_latest.md"],
            "generated_at_utc": "2026-03-07T00:00:00Z",
            "response_id": "resp_test",
            "raw_response_text": "{}",
            "draft": {
                "title": "Prediction Gate",
                "decision": "NO_GO",
                "summary": "표본 부족으로 보류합니다.",
                "blockers": ["운영 표본 부족"],
                "risks": ["24시간 모니터링 미완료"],
                "next_actions": ["체크포인트 완료"],
                "evidence": [
                    {
                        "claim": "자동 판정이 NO-GO",
                        "source": "docs/qa/prediction_rollout_gate_latest.md",
                        "excerpt": "모든 체크포인트가 Pending",
                    },
                    {
                        "claim": "전체 체크포인트 Pending",
                        "source": "docs/qa/prediction_stage1_sync_summary_latest.md",
                        "excerpt": "Pending",
                    },
                ],
                "confidence": "high",
            },
            "tool_trace": [],
        }
    )


def _sample_evaluation() -> ReleaseDecisionEvaluateResponse:
    return ReleaseDecisionEvaluateResponse(
        case=ReleaseDecisionEvalCaseSummary(
            case_id="prediction_stage2_hold",
            scenario="prediction_stage2",
            expected_decision="NO_GO",
            required_keywords=["표본 부족"],
            required_sources=["docs/qa/prediction_rollout_gate_latest.md"],
        ),
        evaluation=ReleaseDecisionEvalResult(
            case_id="prediction_stage2_hold",
            status="PASS",
            decision_ok=True,
            keyword_hits={"표본 부족": True},
            source_hits={"docs/qa/prediction_rollout_gate_latest.md": True},
            missing_keywords=[],
            missing_sources=[],
        ),
    )


def test_save_artifact_writes_json_and_markdown(tmp_path: Path) -> None:
    store = ReleaseDecisionArtifactStore(tmp_path / "reports" / "release-decision")
    summary = store.save_artifact(
        scenario="prediction_stage2",
        task_prompt="Draft the rollout decision.",
        seed_paths=["docs/qa/prediction_rollout_gate_latest.md"],
        allowed_roots=["docs/qa"],
        draft_response=_sample_run_result(),
        markdown="# Draft\n",
        evaluation=None,
        saved_at=datetime(2026, 3, 7, 0, 0, tzinfo=timezone.utc),
    )

    scenario_dir = tmp_path / "reports" / "release-decision" / "prediction_stage2"
    assert (scenario_dir / summary.json_filename).exists()
    assert (scenario_dir / summary.markdown_filename).exists()
    assert summary.eval_status is None


def test_save_artifact_persists_evaluation(tmp_path: Path) -> None:
    store = ReleaseDecisionArtifactStore(tmp_path / "reports" / "release-decision")
    summary = store.save_artifact(
        scenario="prediction_stage2",
        task_prompt="Draft the rollout decision.",
        seed_paths=["docs/qa/prediction_rollout_gate_latest.md"],
        allowed_roots=["docs/qa"],
        draft_response=_sample_run_result(),
        markdown="# Draft\n",
        evaluation=_sample_evaluation(),
        saved_at=datetime(2026, 3, 7, 0, 0, tzinfo=timezone.utc),
    )

    record = store.load_artifact(summary.artifact_id)
    assert record.evaluation is not None
    assert record.evaluation.evaluation.status == "PASS"


def test_list_artifacts_returns_newest_first(tmp_path: Path) -> None:
    store = ReleaseDecisionArtifactStore(tmp_path / "reports" / "release-decision")
    store.save_artifact(
        scenario="prediction_stage2",
        task_prompt="older",
        seed_paths=[],
        allowed_roots=[],
        draft_response=_sample_run_result(),
        markdown="# older\n",
        evaluation=None,
        saved_at=datetime(2026, 3, 7, 0, 0, tzinfo=timezone.utc),
    )
    latest = store.save_artifact(
        scenario="prediction_stage2",
        task_prompt="newer",
        seed_paths=[],
        allowed_roots=[],
        draft_response=_sample_run_result(),
        markdown="# newer\n",
        evaluation=_sample_evaluation(),
        saved_at=datetime(2026, 3, 7, 1, 0, tzinfo=timezone.utc),
    )

    summaries = store.list_artifacts()
    assert summaries[0].artifact_id == latest.artifact_id
    assert summaries[0].eval_status == "PASS"


def test_load_artifact_raises_for_unknown_id(tmp_path: Path) -> None:
    store = ReleaseDecisionArtifactStore(tmp_path / "reports" / "release-decision")

    try:
        store.load_artifact("unknown_artifact")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected unknown artifact to raise FileNotFoundError")


def test_save_artifact_keeps_files_within_root(tmp_path: Path) -> None:
    store = ReleaseDecisionArtifactStore(tmp_path / "reports" / "release-decision")
    summary = store.save_artifact(
        scenario="../prediction_stage2",
        task_prompt="Draft the rollout decision.",
        seed_paths=[],
        allowed_roots=[],
        draft_response=_sample_run_result(),
        markdown="# Draft\n",
        evaluation=None,
        saved_at=datetime(2026, 3, 7, 0, 0, tzinfo=timezone.utc),
    )

    scenario_dir = (tmp_path / "reports" / "release-decision").resolve()
    json_path = next(scenario_dir.rglob(summary.json_filename)).resolve()
    markdown_path = next(scenario_dir.rglob(summary.markdown_filename)).resolve()
    assert scenario_dir in json_path.parents
    assert scenario_dir in markdown_path.parents
