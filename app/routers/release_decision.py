"""릴리스/롤아웃 결정 초안 생성 라우터."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import partial
import os
from pathlib import Path
from threading import BoundedSemaphore
from typing import Callable, TypeVar

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..agents.release_decision_agent import (
    ReleaseDecisionRunResult,
    ResponsesReleaseDecisionAgent,
    SCENARIO_PRESETS,
    ReleaseDecisionDraft,
    render_release_decision_markdown,
)
from ..agents.release_decision_artifacts import (
    ReleaseDecisionArtifactRecord,
    ReleaseDecisionArtifactStore,
    ReleaseDecisionArtifactSummary,
    ReleaseDecisionEvalCaseSummary,
    ReleaseDecisionEvaluateResponse,
)
from ..agents.release_decision_eval import (
    evaluate_release_decision,
    load_eval_cases,
)
from ..config import get_settings
from ..core.errors import (
    AIDependencyUnavailable,
    AIDependencyUnavailableResponse,
)
from ..core.model_circuit import release_decision_circuit
from ..internal_auth import require_ai_internal_token

_SERVICE_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_EVIDENCE_ROOT = (
    _LOCAL_REPOSITORY_ROOT
    if (_LOCAL_REPOSITORY_ROOT / "docs" / "qa").is_dir()
    else _SERVICE_ROOT
)
WORKSPACE_ROOT = Path(
    os.getenv("AI_RELEASE_EVIDENCE_ROOT", str(_DEFAULT_EVIDENCE_ROOT))
).resolve()
EVAL_CASES_PATH = _SERVICE_ROOT / "evals" / "release_decision_cases.json"
ARTIFACTS_ROOT = _SERVICE_ROOT / "reports" / "release-decision"
router = APIRouter(prefix="/ai/release-decision", tags=["release-decision"])
_MODEL_EXECUTORS: dict[int, "_BoundedModelExecutor"] = {}
_T = TypeVar("_T")


class EvidenceProfile(str, Enum):
    RELEASE_GATE = "release_gate"


EVIDENCE_PROFILE_ROOTS: dict[EvidenceProfile, tuple[str, ...]] = {
    EvidenceProfile.RELEASE_GATE: ("docs/qa",),
}


def _dependency_unavailable(message: str) -> AIDependencyUnavailable:
    return AIDependencyUnavailable(message)


class ReleaseDecisionPresetResponse(BaseModel):
    scenario: str
    task_prompt: str
    seed_paths: list[str]
    allowed_roots: list[str]


class ReleaseDecisionDraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str = Field(..., description="Built-in scenario preset name.")
    evidence_profile: EvidenceProfile = EvidenceProfile.RELEASE_GATE
    task_prompt: str | None = Field(
        None, description="Optional override prompt for the selected scenario."
    )
    model: str | None = Field(None, description="Optional Responses API model.")
    max_tool_rounds: int = Field(3, ge=1, le=3)
    max_output_tokens: int = Field(2200, ge=400, le=8000)


class ReleaseDecisionDraftResponse(BaseModel):
    result: ReleaseDecisionRunResult
    markdown: str


class ReleaseDecisionEvaluateRequest(BaseModel):
    case_id: str
    draft: ReleaseDecisionDraft


class ReleaseDecisionSaveRequest(BaseModel):
    scenario: str
    task_prompt: str | None = None
    seed_paths: list[str] = Field(default_factory=list)
    allowed_roots: list[str] = Field(default_factory=list)
    draft_response: ReleaseDecisionRunResult
    markdown: str = Field(..., min_length=1)
    evaluation: ReleaseDecisionEvaluateResponse | None = None


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


class _BoundedModelExecutor:
    """Reject excess work and retain capacity until timed-out threads really exit."""

    def __init__(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        self._permits = BoundedSemaphore(self.limit)
        self._executor = ThreadPoolExecutor(
            max_workers=self.limit,
            thread_name_prefix="release-decision-model",
        )

    async def run(self, call: Callable[[], _T], *, timeout: float) -> _T:
        if not self._permits.acquire(blocking=False):
            raise _dependency_unavailable(
                "release decision model capacity is exhausted"
            )
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, call)
        future.add_done_callback(lambda _future: self._permits.release())
        return await asyncio.wait_for(
            asyncio.shield(future),
            timeout=max(0.001, float(timeout)),
        )


def _model_executor(limit: int) -> _BoundedModelExecutor:
    normalized = max(1, int(limit))
    executor = _MODEL_EXECUTORS.get(normalized)
    if executor is None:
        executor = _BoundedModelExecutor(normalized)
        _MODEL_EXECUTORS[normalized] = executor
    return executor


def _load_release_eval_cases():
    try:
        return load_eval_cases(EVAL_CASES_PATH)
    except Exception as exc:
        raise _dependency_unavailable(
            f"release eval cases unavailable: {exc}"
        ) from exc


def _artifact_store() -> ReleaseDecisionArtifactStore:
    return ReleaseDecisionArtifactStore(ARTIFACTS_ROOT)


@router.get("/presets", response_model=list[ReleaseDecisionPresetResponse])
async def list_release_decision_presets(
    _: None = Depends(require_ai_internal_token),
):
    return [
        ReleaseDecisionPresetResponse(
            scenario=name,
            task_prompt=preset.task_prompt,
            seed_paths=list(preset.seed_paths),
            allowed_roots=list(preset.allowed_roots),
        )
        for name, preset in SCENARIO_PRESETS.items()
    ]


@router.get(
    "/eval-cases",
    response_model=list[ReleaseDecisionEvalCaseSummary],
    responses={503: {"model": AIDependencyUnavailableResponse}},
)
async def list_release_decision_eval_cases(
    _: None = Depends(require_ai_internal_token),
):
    return [
        ReleaseDecisionEvalCaseSummary(
            case_id=case.case_id,
            scenario=case.scenario,
            expected_decision=case.expected_decision,
            required_keywords=case.required_keywords,
            required_sources=case.required_sources,
        )
        for case in _load_release_eval_cases()
    ]


@router.post(
    "/draft",
    response_model=ReleaseDecisionDraftResponse,
    responses={503: {"model": AIDependencyUnavailableResponse}},
)
async def draft_release_decision(
    payload: ReleaseDecisionDraftRequest,
    _: None = Depends(require_ai_internal_token),
):
    preset = SCENARIO_PRESETS.get(payload.scenario)
    if preset is None:
        raise HTTPException(status_code=404, detail="Unknown release decision scenario")

    settings = get_settings()
    if not settings.openai_api_key:
        raise _dependency_unavailable(
            "OPENAI_API_KEY is required to use release decision drafting"
        )
    try:
        agent = ResponsesReleaseDecisionAgent(
            workspace_root=WORKSPACE_ROOT,
            allowed_roots=list(EVIDENCE_PROFILE_ROOTS[payload.evidence_profile]),
            api_key=settings.openai_api_key,
            model=payload.model,
            max_tool_rounds=min(
                payload.max_tool_rounds,
                getattr(settings, "ai_model_max_attempts", 3),
            ),
            max_output_tokens=payload.max_output_tokens,
            request_timeout_seconds=getattr(
                settings, "ai_model_request_deadline_seconds", 75.0
            ),
            max_attempts=getattr(settings, "ai_model_max_attempts", 3),
        )
    except ValueError as exc:
        raise _dependency_unavailable(str(exc)) from exc

    try:
        if not release_decision_circuit.allow_request():
            raise _dependency_unavailable(
                "release decision model circuit is open"
            )
        executor = _model_executor(
            getattr(settings, "ai_model_max_concurrency", 4)
        )
        result = await executor.run(
            partial(
                agent.draft,
                scenario=preset.name,
                task_prompt=payload.task_prompt or preset.task_prompt,
                seed_paths=list(preset.seed_paths),
            ),
            timeout=getattr(settings, "ai_model_request_deadline_seconds", 75.0),
        )
        release_decision_circuit.record_success()
    except TimeoutError as exc:
        release_decision_circuit.record_failure()
        raise _dependency_unavailable(
            "release decision model deadline exceeded"
        ) from exc
    except (HTTPException, AIDependencyUnavailable):
        raise
    except Exception as exc:
        release_decision_circuit.record_failure()
        raise _dependency_unavailable(f"release decision draft failed: {exc}") from exc

    return ReleaseDecisionDraftResponse(
        result=result,
        markdown=render_release_decision_markdown(result.draft),
    )


@router.post(
    "/evaluate",
    response_model=ReleaseDecisionEvaluateResponse,
    responses={503: {"model": AIDependencyUnavailableResponse}},
)
async def evaluate_release_decision_draft(
    payload: ReleaseDecisionEvaluateRequest,
    _: None = Depends(require_ai_internal_token),
):
    cases = _load_release_eval_cases()
    case = next((item for item in cases if item.case_id == payload.case_id), None)
    if case is None:
        raise HTTPException(
            status_code=404, detail="Unknown release decision eval case"
        )

    evaluation = evaluate_release_decision(payload.draft, case)
    return ReleaseDecisionEvaluateResponse(
        case=ReleaseDecisionEvalCaseSummary(
            case_id=case.case_id,
            scenario=case.scenario,
            expected_decision=case.expected_decision,
            required_keywords=case.required_keywords,
            required_sources=case.required_sources,
        ),
        evaluation=evaluation,
    )


@router.post("/save", response_model=ReleaseDecisionArtifactSummary)
async def save_release_decision_artifact(
    payload: ReleaseDecisionSaveRequest,
    _: None = Depends(require_ai_internal_token),
):
    try:
        return _artifact_store().save_artifact(
            scenario=payload.scenario,
            task_prompt=payload.task_prompt,
            seed_paths=_dedupe(payload.seed_paths),
            allowed_roots=_dedupe(payload.allowed_roots),
            draft_response=payload.draft_response,
            markdown=payload.markdown,
            evaluation=payload.evaluation,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"release decision save failed: {exc}",
        ) from exc


@router.get("/artifacts", response_model=list[ReleaseDecisionArtifactSummary])
async def list_release_decision_artifacts(
    _: None = Depends(require_ai_internal_token),
):
    try:
        return _artifact_store().list_artifacts()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"release decision artifact listing failed: {exc}",
        ) from exc


@router.get("/artifacts/{artifact_id}", response_model=ReleaseDecisionArtifactRecord)
async def get_release_decision_artifact(
    artifact_id: str,
    _: None = Depends(require_ai_internal_token),
):
    try:
        return _artifact_store().load_artifact(artifact_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="Unknown release decision artifact"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"release decision artifact load failed: {exc}",
        ) from exc
