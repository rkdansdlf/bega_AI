"""릴리스/롤아웃 결정 초안 생성 라우터."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

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
from ..internal_auth import require_ai_internal_token

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
EVAL_CASES_PATH = WORKSPACE_ROOT / "bega_AI" / "evals" / "release_decision_cases.json"
ARTIFACTS_ROOT = WORKSPACE_ROOT / "reports" / "release-decision"
router = APIRouter(prefix="/ai/release-decision", tags=["release-decision"])


class ReleaseDecisionPresetResponse(BaseModel):
    scenario: str
    task_prompt: str
    seed_paths: list[str]
    allowed_roots: list[str]


class ReleaseDecisionDraftRequest(BaseModel):
    scenario: str = Field(..., description="Built-in scenario preset name.")
    task_prompt: str | None = Field(
        None, description="Optional override prompt for the selected scenario."
    )
    seed_paths: list[str] = Field(
        default_factory=list,
        description="Additional repo-relative seed paths.",
    )
    allowed_roots: list[str] = Field(
        default_factory=list,
        description="Additional repo-relative allowed roots inside the workspace.",
    )
    model: str | None = Field(None, description="Optional Responses API model.")
    max_tool_rounds: int = Field(6, ge=1, le=10)
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


def _load_release_eval_cases():
    try:
        return load_eval_cases(EVAL_CASES_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"release eval cases unavailable: {exc}"
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


@router.get("/eval-cases", response_model=list[ReleaseDecisionEvalCaseSummary])
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


@router.post("/draft", response_model=ReleaseDecisionDraftResponse)
async def draft_release_decision(
    payload: ReleaseDecisionDraftRequest,
    _: None = Depends(require_ai_internal_token),
):
    preset = SCENARIO_PRESETS.get(payload.scenario)
    if preset is None:
        raise HTTPException(status_code=404, detail="Unknown release decision scenario")

    settings = get_settings()
    try:
        agent = ResponsesReleaseDecisionAgent(
            workspace_root=WORKSPACE_ROOT,
            allowed_roots=_dedupe(list(preset.allowed_roots) + payload.allowed_roots),
            api_key=settings.openai_api_key,
            model=payload.model,
            max_tool_rounds=payload.max_tool_rounds,
            max_output_tokens=payload.max_output_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = agent.draft(
            scenario=preset.name,
            task_prompt=payload.task_prompt or preset.task_prompt,
            seed_paths=_dedupe(list(preset.seed_paths) + payload.seed_paths),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"release decision draft failed: {exc}",
        ) from exc

    return ReleaseDecisionDraftResponse(
        result=result,
        markdown=render_release_decision_markdown(result.draft),
    )


@router.post("/evaluate", response_model=ReleaseDecisionEvaluateResponse)
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
