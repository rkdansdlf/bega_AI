from __future__ import annotations

from app.agents.release_decision_agent import ReleaseDecisionDraft
from app.agents.release_decision_eval import (
    ReleaseDecisionEvalCase,
    evaluate_release_decision,
)


def test_evaluate_release_decision_passes_when_all_checks_match() -> None:
    case = ReleaseDecisionEvalCase.model_validate(
        {
            "case_id": "stadium_release_go",
            "scenario": "stadium_release",
            "task_prompt": "Draft the release gate decision.",
            "expected_decision": "GO",
            "required_keywords": ["pass", "5 passing, 0 failing"],
            "required_sources": [
                "docs/qa/stadium-release-gate-20260304.md",
                "docs/qa/stadium-guide-smoke-20260304.md",
            ],
        }
    )
    draft = ReleaseDecisionDraft.model_validate(
        {
            "title": "Stadium Gate",
            "decision": "GO",
            "summary": "모든 핵심 검증이 pass이며 5 passing, 0 failing 입니다.",
            "blockers": [],
            "risks": ["후속 문의 리마인드 필요"],
            "next_actions": ["2026-03-06 리마인드 실행"],
            "evidence": [
                {
                    "claim": "Backend/Frontend/metadata verify 모두 PASS",
                    "source": "docs/qa/stadium-release-gate-20260304.md",
                    "excerpt": "Backend stadium tests: PASS",
                },
                {
                    "claim": "수동 스모크 5 passing, 0 failing",
                    "source": "docs/qa/stadium-guide-smoke-20260304.md",
                    "excerpt": "결과: `5 passing, 0 failing`",
                },
            ],
            "confidence": "high",
        }
    )
    result = evaluate_release_decision(draft, case)
    assert result.status == "PASS"
    assert result.decision_ok is True


def test_evaluate_release_decision_fails_on_missing_source() -> None:
    case = ReleaseDecisionEvalCase.model_validate(
        {
            "case_id": "prediction_stage2_hold",
            "scenario": "prediction_stage2",
            "task_prompt": "Draft the rollout decision.",
            "expected_decision": "NO_GO",
            "required_keywords": ["표본 부족"],
            "required_sources": ["docs/qa/prediction_rollout_gate_latest.md"],
        }
    )
    draft = ReleaseDecisionDraft.model_validate(
        {
            "title": "Prediction Gate",
            "decision": "NO_GO",
            "summary": "표본 부족으로 보류합니다.",
            "blockers": ["운영 표본 부족"],
            "risks": [],
            "next_actions": ["모니터링 완료"],
            "evidence": [
                {
                    "claim": "표본 부족",
                    "source": "docs/qa/prediction_stage1_sync_summary_latest.md",
                    "excerpt": "전체 체크포인트가 Pending",
                },
                {
                    "claim": "윈도우 미도래",
                    "source": "docs/qa/prediction_stage1_monitoring_20260305_20260306.md",
                    "excerpt": "24시간 모니터링",
                },
            ],
            "confidence": "medium",
        }
    )
    result = evaluate_release_decision(draft, case)
    assert result.status == "FAIL"
    assert result.missing_sources == ["docs/qa/prediction_rollout_gate_latest.md"]
