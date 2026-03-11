"""릴리스 결정 초안 아티팩트 저장/조회 유틸리티."""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .release_decision_agent import ReleaseDecisionRunResult
from .release_decision_eval import ReleaseDecisionEvalResult

_SAFE_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")
_SAFE_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class ReleaseDecisionEvalCaseSummary(BaseModel):
    case_id: str
    scenario: str
    expected_decision: str
    required_keywords: list[str] = Field(default_factory=list)
    required_sources: list[str] = Field(default_factory=list)


class ReleaseDecisionEvaluateResponse(BaseModel):
    case: ReleaseDecisionEvalCaseSummary
    evaluation: ReleaseDecisionEvalResult


class ReleaseDecisionArtifactRecord(BaseModel):
    artifact_id: str
    saved_at_utc: str
    scenario: str
    task_prompt: str | None = None
    seed_paths: list[str] = Field(default_factory=list)
    allowed_roots: list[str] = Field(default_factory=list)
    draft_response: ReleaseDecisionRunResult
    markdown: str
    evaluation: ReleaseDecisionEvaluateResponse | None = None


class ReleaseDecisionArtifactSummary(BaseModel):
    artifact_id: str
    scenario: str
    decision: Literal["GO", "NO_GO", "PENDING"]
    eval_status: Literal["PASS", "FAIL"] | None = None
    saved_at_utc: str
    markdown_filename: str
    json_filename: str


def _slugify_component(value: str) -> str:
    cleaned = _SAFE_COMPONENT_PATTERN.sub("_", (value or "").strip())
    collapsed = re.sub(r"_+", "_", cleaned).strip("_")
    return collapsed or "artifact"


def _utc_isoformat(now: datetime | None = None) -> str:
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_saved_at(value: str) -> datetime:
    text = (value or "").strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    return datetime.fromisoformat(text)


def _atomic_write_text(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_path, target)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


class ReleaseDecisionArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _scenario_dir(self, scenario: str) -> Path:
        return self.root / _slugify_component(scenario)

    def _json_path(self, scenario: str, artifact_id: str) -> Path:
        return self._scenario_dir(scenario) / f"{artifact_id}.json"

    def _markdown_path(self, scenario: str, artifact_id: str) -> Path:
        return self._scenario_dir(scenario) / f"{artifact_id}.md"

    def _allocate_artifact_id(
        self,
        *,
        scenario: str,
        decision: Literal["GO", "NO_GO", "PENDING"],
        now: datetime | None = None,
    ) -> str:
        base_timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        base = (
            f"{base_timestamp.strftime('%Y%m%dT%H%M%SZ')}"
            f"_{_slugify_component(scenario)}"
            f"_{_slugify_component(decision)}"
        )
        candidate = base
        while (
            self._json_path(scenario, candidate).exists()
            or self._markdown_path(scenario, candidate).exists()
        ):
            candidate = f"{base}_{secrets.token_hex(3)}"
        return candidate

    def save_artifact(
        self,
        *,
        scenario: str,
        task_prompt: str | None,
        seed_paths: list[str],
        allowed_roots: list[str],
        draft_response: ReleaseDecisionRunResult,
        markdown: str,
        evaluation: ReleaseDecisionEvaluateResponse | None,
        saved_at: datetime | None = None,
    ) -> ReleaseDecisionArtifactSummary:
        saved_at_value = (saved_at or datetime.now(timezone.utc)).astimezone(
            timezone.utc
        )
        artifact_id = self._allocate_artifact_id(
            scenario=scenario,
            decision=draft_response.draft.decision,
            now=saved_at_value,
        )
        record = ReleaseDecisionArtifactRecord(
            artifact_id=artifact_id,
            saved_at_utc=_utc_isoformat(saved_at_value),
            scenario=scenario,
            task_prompt=task_prompt,
            seed_paths=list(seed_paths),
            allowed_roots=list(allowed_roots),
            draft_response=draft_response,
            markdown=markdown,
            evaluation=evaluation,
        )
        json_path = self._json_path(scenario, artifact_id)
        markdown_path = self._markdown_path(scenario, artifact_id)
        json_text = (
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n"
        )
        markdown_text = markdown if markdown.endswith("\n") else f"{markdown}\n"

        _atomic_write_text(json_path, json_text)
        _atomic_write_text(markdown_path, markdown_text)
        return self._build_summary(record)

    def list_artifacts(self) -> list[ReleaseDecisionArtifactSummary]:
        if not self.root.exists():
            return []

        summaries: list[ReleaseDecisionArtifactSummary] = []
        for json_path in self.root.rglob("*.json"):
            try:
                record = ReleaseDecisionArtifactRecord.model_validate_json(
                    json_path.read_text(encoding="utf-8")
                )
            except Exception:
                continue
            summaries.append(self._build_summary(record))

        summaries.sort(
            key=lambda item: _parse_saved_at(item.saved_at_utc),
            reverse=True,
        )
        return summaries

    def load_artifact(self, artifact_id: str) -> ReleaseDecisionArtifactRecord:
        if not _SAFE_ARTIFACT_ID_PATTERN.fullmatch(artifact_id or ""):
            raise FileNotFoundError(artifact_id)
        if not self.root.exists():
            raise FileNotFoundError(artifact_id)

        matches = sorted(self.root.rglob(f"{artifact_id}.json"))
        if not matches:
            raise FileNotFoundError(artifact_id)
        return ReleaseDecisionArtifactRecord.model_validate_json(
            matches[0].read_text(encoding="utf-8")
        )

    def _build_summary(
        self, record: ReleaseDecisionArtifactRecord
    ) -> ReleaseDecisionArtifactSummary:
        return ReleaseDecisionArtifactSummary(
            artifact_id=record.artifact_id,
            scenario=record.scenario,
            decision=record.draft_response.draft.decision,
            eval_status=(
                record.evaluation.evaluation.status
                if record.evaluation is not None
                else None
            ),
            saved_at_utc=record.saved_at_utc,
            markdown_filename=f"{record.artifact_id}.md",
            json_filename=f"{record.artifact_id}.json",
        )
