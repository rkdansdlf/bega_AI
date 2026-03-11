"""Responses API 기반 릴리스/롤아웃 결정 초안 에이전트."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from openai import OpenAI
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    claim: str
    source: str
    excerpt: str


class ReleaseDecisionDraft(BaseModel):
    title: str
    decision: Literal["GO", "NO_GO", "PENDING"]
    summary: str
    blockers: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list, min_length=2)
    confidence: Literal["low", "medium", "high"]


class ToolTraceItem(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    result_preview: str


class ReleaseDecisionRunResult(BaseModel):
    scenario: str
    model: str
    task_prompt: str
    seed_paths: list[str]
    generated_at_utc: str
    response_id: str | None = None
    raw_response_text: str
    draft: ReleaseDecisionDraft
    tool_trace: list[ToolTraceItem] = Field(default_factory=list)


@dataclass(frozen=True)
class ScenarioPreset:
    name: str
    task_prompt: str
    seed_paths: tuple[str, ...]
    allowed_roots: tuple[str, ...]


SCENARIO_PRESETS: dict[str, ScenarioPreset] = {
    "prediction_stage2": ScenarioPreset(
        name="prediction_stage2",
        task_prompt=(
            "Prediction Stage 2 진행 여부를 판단하는 최신 rollout decision 초안을 작성하라. "
            "문서 근거만 사용하고, 표본 부족이면 PENDING 대신 명시적 기준에 따라 "
            "NO_GO 또는 PENDING을 구분하라."
        ),
        seed_paths=(
            "docs/qa/prediction_rollout_decision_20260306.md",
            "docs/qa/prediction_rollout_gate_latest.md",
            "docs/qa/prediction_stage1_sync_summary_latest.md",
            "docs/qa/prediction_stage1_monitoring_20260305_20260306.md",
        ),
        allowed_roots=("docs/qa", "scripts"),
    ),
    "stadium_release": ScenarioPreset(
        name="stadium_release",
        task_prompt=(
            "Stadium 기능의 현재 release gate 결정을 작성하라. 테스트 통과 여부, "
            "운영 상태, 남은 후속 작업을 근거와 함께 정리하라."
        ),
        seed_paths=(
            "docs/qa/stadium-release-gate-20260304.md",
            "docs/qa/stadium-guide-smoke-20260304.md",
            "docs/qa/stadium-guide-smoke-staging-20260304.md",
            "docs/qa/stadium-food-response-runbook-20260304.md",
        ),
        allowed_roots=("docs/qa", "reports", "bega_backend/docs"),
    ),
}


def render_release_decision_markdown(draft: ReleaseDecisionDraft) -> str:
    lines = [
        f"# {draft.title}",
        "",
        f"- Decision: `{draft.decision}`",
        f"- Confidence: `{draft.confidence}`",
        "",
        "## Summary",
        draft.summary,
        "",
        "## Blockers",
    ]
    blockers = draft.blockers or ["없음"]
    for item in blockers:
        lines.append(f"- {item}")

    lines.extend(["", "## Risks"])
    risks = draft.risks or ["없음"]
    for item in risks:
        lines.append(f"- {item}")

    lines.extend(["", "## Next Actions"])
    actions = draft.next_actions or ["없음"]
    for item in actions:
        lines.append(f"- {item}")

    lines.extend(["", "## Evidence"])
    for item in draft.evidence:
        lines.append(f"- {item.claim} ({item.source})")
        lines.append(f"  - Excerpt: {item.excerpt}")

    return "\n".join(lines) + "\n"


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty response text")

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
    if fenced_match:
        raw = fenced_match.group(1).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise
        parsed = json.loads(raw[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("response did not contain a JSON object")
    return parsed


class WorkspaceDocumentTools:
    """LLM이 안전하게 접근할 수 있는 워크스페이스 문서 도구."""

    def __init__(
        self,
        workspace_root: Path,
        allowed_roots: Sequence[Path | str],
        *,
        max_file_bytes: int = 512_000,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.max_file_bytes = max_file_bytes
        self.allowed_extensions = {".md", ".txt", ".json", ".log", ".ndjson"}
        self.allowed_roots = self._normalize_allowed_roots(allowed_roots)
        self._text_cache: dict[Path, str] = {}

    def _is_within_workspace(self, path: Path) -> bool:
        resolved = path.resolve()
        return (
            resolved == self.workspace_root or self.workspace_root in resolved.parents
        )

    def _normalize_allowed_roots(self, roots: Sequence[Path | str]) -> list[Path]:
        normalized: list[Path] = []
        for root in roots:
            if isinstance(root, Path) and root.is_absolute():
                candidate = root.resolve()
            else:
                candidate = (self.workspace_root / Path(root)).resolve()
            if candidate.exists() and self._is_within_workspace(candidate):
                normalized.append(candidate)
        if not normalized:
            normalized.append(self.workspace_root)
        return normalized

    def _is_allowed(self, path: Path) -> bool:
        resolved = path.resolve()
        if resolved.suffix.lower() not in self.allowed_extensions:
            return False
        try:
            if resolved.stat().st_size > self.max_file_bytes:
                return False
        except FileNotFoundError:
            return False
        return any(
            root == resolved or root in resolved.parents for root in self.allowed_roots
        )

    def _candidate_files(self, path_prefix: str | None = None) -> Iterable[Path]:
        prefix_path: Path | None = None
        if path_prefix:
            prefix_candidate = (self.workspace_root / path_prefix).resolve()
            if prefix_candidate.exists() and self._is_within_workspace(
                prefix_candidate
            ):
                prefix_path = prefix_candidate

        search_roots = [prefix_path] if prefix_path is not None else self.allowed_roots
        seen: set[Path] = set()
        for root in search_roots:
            if root is None or not root.exists():
                continue
            if root.is_file():
                candidates = [root]
            else:
                candidates = sorted(root.rglob("*"))
            for candidate in candidates:
                if candidate in seen:
                    continue
                if candidate.is_file() and self._is_allowed(candidate):
                    seen.add(candidate)
                    yield candidate

    def _relative_path(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.workspace_root))

    def _resolve_user_path(self, raw_path: str) -> Path:
        candidate = (self.workspace_root / raw_path).resolve()
        if not self._is_within_workspace(candidate):
            raise ValueError(f"path escapes workspace: {raw_path}")
        if not self._is_allowed(candidate):
            raise ValueError(f"path is not allowed: {raw_path}")
        return candidate

    def _read_text(self, path: Path) -> str:
        cached = self._text_cache.get(path)
        if cached is not None:
            return cached
        text = path.read_text(encoding="utf-8", errors="replace")
        self._text_cache[path] = text
        return text

    def list_documents(self, pattern: str, limit: int = 20) -> dict[str, Any]:
        limit = max(1, min(int(limit), 50))
        matched: list[str] = []
        for candidate in self._candidate_files():
            rel_path = self._relative_path(candidate)
            if Path(rel_path).match(pattern):
                matched.append(rel_path)
            if len(matched) >= limit:
                break
        return {"pattern": pattern, "matches": matched, "count": len(matched)}

    def search_documents(
        self,
        query: str,
        limit: int = 8,
        path_prefix: str | None = None,
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), 20))
        normalized_query = query.strip().lower()
        tokens = [
            token for token in re.split(r"\W+", normalized_query) if len(token) >= 2
        ]
        matches: list[dict[str, Any]] = []

        for candidate in self._candidate_files(path_prefix=path_prefix):
            rel_path = self._relative_path(candidate)
            score = 0
            excerpts: list[dict[str, Any]] = []
            for line_number, raw_line in enumerate(
                self._read_text(candidate).splitlines(), start=1
            ):
                line = raw_line.strip()
                if not line:
                    continue
                line_lower = line.lower()
                line_hits = sum(1 for token in tokens if token in line_lower)
                if normalized_query and normalized_query in line_lower:
                    line_hits += max(3, len(tokens))
                if line_hits:
                    score += line_hits
                    if len(excerpts) < 3:
                        excerpts.append(
                            {
                                "line_number": line_number,
                                "text": line[:240],
                            }
                        )
            if normalized_query and normalized_query in rel_path.lower():
                score += 4
            if score:
                matches.append(
                    {
                        "path": rel_path,
                        "score": score,
                        "matches": excerpts,
                    }
                )

        matches.sort(key=lambda item: (-int(item["score"]), str(item["path"])))
        return {
            "query": query,
            "results": matches[:limit],
            "count": len(matches[:limit]),
        }

    def read_document(
        self,
        path: str,
        start_line: int = 1,
        max_lines: int = 80,
    ) -> dict[str, Any]:
        resolved = self._resolve_user_path(path)
        start_line = max(1, int(start_line))
        max_lines = max(1, min(int(max_lines), 160))
        lines = self._read_text(resolved).splitlines()
        start_index = start_line - 1
        end_index = min(len(lines), start_index + max_lines)
        excerpt = [
            {
                "line_number": idx,
                "text": lines[idx - 1][:300],
            }
            for idx in range(start_line, end_index + 1)
        ]
        return {
            "path": self._relative_path(resolved),
            "start_line": start_line,
            "end_line": end_index,
            "line_count": len(lines),
            "excerpt": excerpt,
        }


class ResponsesReleaseDecisionAgent:
    """OpenAI Responses API와 로컬 문서 도구를 사용하는 좁은 업무형 에이전트."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        allowed_roots: Sequence[Path | str],
        client: OpenAI | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_tool_rounds: int = 6,
        max_output_tokens: int = 2200,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.tools = WorkspaceDocumentTools(
            workspace_root=self.workspace_root,
            allowed_roots=allowed_roots,
        )
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if client is None and not resolved_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required to use ResponsesReleaseDecisionAgent"
            )
        self.client = client or OpenAI(api_key=resolved_api_key)
        self.model = model or os.environ.get("OPENAI_RESPONSES_MODEL", "gpt-4.1-mini")
        self.max_tool_rounds = max(1, int(max_tool_rounds))
        self.max_output_tokens = max(400, int(max_output_tokens))

    def _tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "list_documents",
                "description": "List repo-relative document paths by glob pattern.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "search_documents",
                "description": "Search allowed documents and return matching lines.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                        "path_prefix": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "read_document",
                "description": "Read a repo-relative document excerpt with line numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "max_lines": {"type": "integer", "minimum": 1, "maximum": 160},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        ]

    def _schema_text(self) -> str:
        schema = ReleaseDecisionDraft.model_json_schema()
        return json.dumps(schema, ensure_ascii=False, indent=2)

    def _instructions(self) -> str:
        return (
            "You are drafting a release or rollout decision memo for the KBO Platform repository.\n"
            "Use repository documents only. Never invent evidence.\n"
            "Inspect documents with tools before finalizing.\n"
            "Decision policy:\n"
            "- GO: explicit pass/green evidence and no blocking failures.\n"
            "- NO_GO: explicit blocker, failure, or explicit no-go decision.\n"
            "- PENDING: evidence is incomplete or inconclusive without a clear blocker.\n"
            "Citations:\n"
            "- Use repo-relative paths in evidence[].source.\n"
            "- Keep excerpts short and factual.\n"
            "Output JSON only. It must match this schema exactly:\n"
            f"{self._schema_text()}"
        )

    def _build_initial_input(
        self,
        *,
        scenario: str,
        task_prompt: str,
        seed_paths: Sequence[str],
    ) -> str:
        seed_block = (
            "\n".join(f"- {path}" for path in seed_paths) if seed_paths else "- 없음"
        )
        return (
            f"Scenario: {scenario}\n"
            f"Workspace root: {self.workspace_root}\n"
            f"Task: {task_prompt}\n"
            "Start by inspecting the seed paths, then expand only if needed.\n"
            "Seed paths:\n"
            f"{seed_block}\n"
            "Return only the final JSON object after using tools."
        )

    def _execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        if tool_name == "list_documents":
            return self.tools.list_documents(**arguments)
        if tool_name == "search_documents":
            return self.tools.search_documents(**arguments)
        if tool_name == "read_document":
            return self.tools.read_document(**arguments)
        raise ValueError(f"unknown tool: {tool_name}")

    def draft(
        self,
        *,
        scenario: str,
        task_prompt: str,
        seed_paths: Sequence[str] | None = None,
    ) -> ReleaseDecisionRunResult:
        normalized_seed_paths = list(seed_paths or [])
        response = self.client.responses.create(
            model=self.model,
            instructions=self._instructions(),
            input=self._build_initial_input(
                scenario=scenario,
                task_prompt=task_prompt,
                seed_paths=normalized_seed_paths,
            ),
            tools=self._tool_specs(),
            tool_choice="auto",
            parallel_tool_calls=False,
            max_output_tokens=self.max_output_tokens,
            temperature=0,
        )
        trace: list[ToolTraceItem] = []

        for _ in range(self.max_tool_rounds):
            function_calls = [
                item
                for item in getattr(response, "output", [])
                if getattr(item, "type", None) == "function_call"
            ]
            if not function_calls:
                break

            follow_up_items: list[dict[str, Any]] = []
            for call in function_calls:
                try:
                    arguments = json.loads(getattr(call, "arguments", "") or "{}")
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"tool arguments were not valid JSON for {call.name}"
                    ) from exc

                result = self._execute_tool(call.name, arguments)
                preview = json.dumps(result, ensure_ascii=False)[:400]
                trace.append(
                    ToolTraceItem(
                        tool_name=call.name,
                        arguments=arguments,
                        result_preview=preview,
                    )
                )
                follow_up_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )

            response = self.client.responses.create(
                model=self.model,
                instructions=self._instructions(),
                previous_response_id=getattr(response, "id", None),
                input=follow_up_items,
                tools=self._tool_specs(),
                tool_choice="auto",
                parallel_tool_calls=False,
                max_output_tokens=self.max_output_tokens,
                temperature=0,
            )

        raw_response_text = getattr(response, "output_text", "") or ""
        parsed = _extract_json_object(raw_response_text)
        draft = ReleaseDecisionDraft.model_validate(parsed)
        return ReleaseDecisionRunResult(
            scenario=scenario,
            model=self.model,
            task_prompt=task_prompt,
            seed_paths=normalized_seed_paths,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            response_id=getattr(response, "id", None),
            raw_response_text=raw_response_text,
            draft=draft,
            tool_trace=trace,
        )
