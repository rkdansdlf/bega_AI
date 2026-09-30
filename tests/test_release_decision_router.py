from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.routers import release_decision
from app.core.errors import (
    AIDependencyUnavailable,
    ai_dependency_unavailable_handler,
)


class _FakeReleaseDecisionAgent:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def draft(self, *, scenario: str, task_prompt: str, seed_paths: list[str]):
        draft = release_decision.ReleaseDecisionRunResult.model_validate(
            {
                "scenario": scenario,
                "model": self.kwargs.get("model") or "gpt-test",
                "task_prompt": task_prompt,
                "seed_paths": seed_paths,
                "generated_at_utc": "2026-03-07T00:00:00Z",
                "response_id": "resp_test",
                "raw_response_text": "{}",
                "draft": {
                    "title": "Prediction Gate",
                    "decision": "NO_GO",
                    "summary": "전체 체크포인트가 Pending이어서 보류합니다.",
                    "blockers": ["운영 표본 부족"],
                    "risks": ["24시간 모니터링 미완료"],
                    "next_actions": ["체크포인트 완료"],
                    "evidence": [
                        {
                            "claim": "자동 판정이 NO-GO",
                            "source": "docs/qa/prediction_rollout_gate_latest.md",
                            "excerpt": "NO-GO (보류)",
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
        return draft


class _FailingReleaseDecisionAgent(_FakeReleaseDecisionAgent):
    def draft(self, *, scenario: str, task_prompt: str, seed_paths: list[str]):
        raise RuntimeError("provider unavailable")


def _build_client() -> TestClient:
    test_app = FastAPI()
    test_app.add_exception_handler(
        AIDependencyUnavailable,
        ai_dependency_unavailable_handler,
    )
    test_app.include_router(release_decision.router)
    return TestClient(test_app)


def test_bounded_model_executor_retains_capacity_until_timed_out_threads_exit() -> None:
    blocker = threading.Event()
    lock = threading.Lock()
    active = 0
    max_active = 0

    def blocking_call() -> str:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            blocker.wait(timeout=1)
            return "done"
        finally:
            with lock:
                active -= 1

    async def exercise() -> None:
        executor = release_decision._BoundedModelExecutor(2)

        async def time_out() -> None:
            with pytest.raises(TimeoutError):
                await executor.run(blocking_call, timeout=0.02)

        try:
            await asyncio.gather(time_out(), time_out())
            with pytest.raises(AIDependencyUnavailable) as exc_info:
                await executor.run(blocking_call, timeout=0.02)
            assert exc_info.value.retryable is True
            assert max_active == 2
        finally:
            blocker.set()
            await asyncio.sleep(0.05)

    asyncio.run(exercise())


def test_list_release_decision_presets_requires_internal_token(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.internal_auth.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr(
        "app.internal_auth.record_security_event", lambda *args, **kwargs: None
    )

    with _build_client() as client:
        response = client.get("/ai/release-decision/presets")

    assert response.status_code == 401


def test_list_release_decision_presets_returns_known_scenarios(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.internal_auth.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr(
        "app.internal_auth.record_security_event", lambda *args, **kwargs: None
    )

    with _build_client() as client:
        response = client.get(
            "/ai/release-decision/presets",
            headers={"X-Internal-Api-Key": "expected-token"},
        )

    assert response.status_code == 200
    scenarios = {item["scenario"] for item in response.json()}
    assert {"prediction_stage2", "stadium_release"} <= scenarios


def test_list_release_decision_eval_cases_returns_case_summaries(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.internal_auth.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr(
        "app.internal_auth.record_security_event", lambda *args, **kwargs: None
    )

    with _build_client() as client:
        response = client.get(
            "/ai/release-decision/eval-cases",
            headers={"X-Internal-Api-Key": "expected-token"},
        )

    assert response.status_code == 200
    body = response.json()
    assert any(item["case_id"] == "prediction_stage2_hold" for item in body)
    assert any(item["case_id"] == "stadium_release_go" for item in body)


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("get", "/ai/release-decision/eval-cases", None),
        (
            "post",
            "/ai/release-decision/evaluate",
            {
                "case_id": "missing",
                "draft": {
                    "title": "Release gate",
                    "decision": "PENDING",
                    "summary": "Evidence is unavailable.",
                    "blockers": [],
                    "risks": [],
                    "next_actions": [],
                    "evidence": [
                        {
                            "claim": "Evidence one",
                            "source": "docs/qa/one.md",
                            "excerpt": "one",
                        },
                        {
                            "claim": "Evidence two",
                            "source": "docs/qa/two.md",
                            "excerpt": "two",
                        },
                    ],
                    "confidence": "low",
                },
            },
        ),
    ],
)
def test_eval_case_load_failure_uses_standard_503_contract(
    monkeypatch, method: str, path: str, payload
) -> None:
    monkeypatch.setattr(
        "app.internal_auth.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr(
        "app.internal_auth.record_security_event", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.routers.release_decision.load_eval_cases",
        lambda _path: (_ for _ in ()).throw(OSError("missing evidence")),
    )

    with _build_client() as client:
        response = client.request(
            method,
            path,
            headers={"X-Internal-Api-Key": "expected-token"},
            json=payload,
        )

    assert response.status_code == 503
    assert response.json() == {
        "code": "AI_DEPENDENCY_UNAVAILABLE",
        "message": "release eval cases unavailable: missing evidence",
        "retryable": True,
    }


def test_draft_release_decision_returns_markdown_and_result(monkeypatch) -> None:
    router_settings = SimpleNamespace(openai_api_key="test-openai-key")
    monkeypatch.setattr(
        "app.internal_auth.get_settings",
        lambda: SimpleNamespace(
            resolved_ai_internal_token="expected-token",
            openai_api_key="test-openai-key",
        ),
    )
    monkeypatch.setattr(
        "app.routers.release_decision.get_settings", lambda: router_settings
    )
    monkeypatch.setattr(
        "app.internal_auth.record_security_event", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.routers.release_decision.ResponsesReleaseDecisionAgent",
        _FakeReleaseDecisionAgent,
    )

    with _build_client() as client:
        response = client.post(
            "/ai/release-decision/draft",
            headers={"X-Internal-Api-Key": "expected-token"},
            json={
                "scenario": "prediction_stage2",
                "evidence_profile": "release_gate",
                "model": "gpt-4.1-mini",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["scenario"] == "prediction_stage2"
    assert "docs/qa/prediction_rollout_gate_latest.md" in body["result"]["seed_paths"]
    assert body["result"]["draft"]["decision"] == "NO_GO"
    assert "## Evidence" in body["markdown"]


def test_draft_release_decision_rejects_request_supplied_document_roots(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.internal_auth.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr(
        "app.internal_auth.record_security_event", lambda *args, **kwargs: None
    )

    with _build_client() as client:
        response = client.post(
            "/ai/release-decision/draft",
            headers={"X-Internal-Api-Key": "expected-token"},
            json={
                "scenario": "prediction_stage2",
                "allowed_roots": ["."],
            },
        )

    assert response.status_code == 422


def test_evaluate_release_decision_returns_pass(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.internal_auth.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr(
        "app.internal_auth.record_security_event", lambda *args, **kwargs: None
    )

    with _build_client() as client:
        response = client.post(
            "/ai/release-decision/evaluate",
            headers={"X-Internal-Api-Key": "expected-token"},
            json={
                "case_id": "prediction_stage2_hold",
                "draft": {
                    "title": "Prediction Gate",
                    "decision": "NO_GO",
                    "summary": "전체 체크포인트가 pending 이고 표본 부족으로 24시간 윈도우가 완료되지 않아 보류합니다.",
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
                            "claim": "Stage2 recommendation is NO-GO",
                            "source": "docs/qa/prediction_stage1_sync_summary_latest.md",
                            "excerpt": "표본 부족/윈도우 미도래",
                        },
                    ],
                    "confidence": "high",
                },
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["evaluation"]["status"] == "PASS"
    assert body["case"]["case_id"] == "prediction_stage2_hold"


def test_draft_release_decision_returns_503_without_openai_key(monkeypatch) -> None:
    router_settings = SimpleNamespace(openai_api_key=None)
    monkeypatch.setattr(
        "app.internal_auth.get_settings",
        lambda: SimpleNamespace(
            resolved_ai_internal_token="expected-token",
            openai_api_key=None,
        ),
    )
    monkeypatch.setattr(
        "app.routers.release_decision.get_settings", lambda: router_settings
    )
    monkeypatch.setattr(
        "app.internal_auth.record_security_event", lambda *args, **kwargs: None
    )

    with _build_client() as client:
        response = client.post(
            "/ai/release-decision/draft",
            headers={"X-Internal-Api-Key": "expected-token"},
            json={"scenario": "prediction_stage2"},
        )

    assert response.status_code == 503
    assert response.json()["code"] == "AI_DEPENDENCY_UNAVAILABLE"
    assert "OPENAI_API_KEY" in response.json()["message"]
    assert response.json()["retryable"] is True


def test_draft_release_decision_maps_provider_failure_to_standard_503(
    monkeypatch,
) -> None:
    router_settings = SimpleNamespace(openai_api_key="test-openai-key")
    monkeypatch.setattr(
        "app.internal_auth.get_settings",
        lambda: SimpleNamespace(
            resolved_ai_internal_token="expected-token",
            openai_api_key="test-openai-key",
        ),
    )
    monkeypatch.setattr(
        "app.routers.release_decision.get_settings", lambda: router_settings
    )
    monkeypatch.setattr(
        "app.internal_auth.record_security_event", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.routers.release_decision.ResponsesReleaseDecisionAgent",
        _FailingReleaseDecisionAgent,
    )

    with _build_client() as client:
        response = client.post(
            "/ai/release-decision/draft",
            headers={"X-Internal-Api-Key": "expected-token"},
            json={"scenario": "prediction_stage2"},
        )

    assert response.status_code == 503
    assert response.json()["code"] == "AI_DEPENDENCY_UNAVAILABLE"


def test_save_release_decision_artifact_and_list_and_detail(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "app.internal_auth.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr(
        "app.internal_auth.record_security_event", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.routers.release_decision.ARTIFACTS_ROOT", tmp_path / "reports"
    )

    payload = {
        "scenario": "prediction_stage2",
        "task_prompt": "Draft the rollout decision.",
        "seed_paths": ["docs/qa/prediction_rollout_gate_latest.md"],
        "allowed_roots": ["docs/qa"],
        "draft_response": {
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
        },
        "markdown": "# Prediction Gate\n",
        "evaluation": None,
    }

    with _build_client() as client:
        save_response = client.post(
            "/ai/release-decision/save",
            headers={"X-Internal-Api-Key": "expected-token"},
            json=payload,
        )
        assert save_response.status_code == 200
        saved = save_response.json()

        list_response = client.get(
            "/ai/release-decision/artifacts",
            headers={"X-Internal-Api-Key": "expected-token"},
        )
        assert list_response.status_code == 200
        listed = list_response.json()
        assert listed[0]["artifact_id"] == saved["artifact_id"]

        detail_response = client.get(
            f"/ai/release-decision/artifacts/{saved['artifact_id']}",
            headers={"X-Internal-Api-Key": "expected-token"},
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["artifact_id"] == saved["artifact_id"]
        assert detail["draft_response"]["draft"]["decision"] == "NO_GO"


def test_get_release_decision_artifact_returns_404_for_unknown_id(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "app.internal_auth.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr(
        "app.internal_auth.record_security_event", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.routers.release_decision.ARTIFACTS_ROOT", tmp_path / "reports"
    )

    with _build_client() as client:
        response = client.get(
            "/ai/release-decision/artifacts/unknown_artifact",
            headers={"X-Internal-Api-Key": "expected-token"},
        )

    assert response.status_code == 404
