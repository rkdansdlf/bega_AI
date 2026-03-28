#!/usr/bin/env python3
"""Run live smoke checks for chatbot and coach endpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chatbot live smoke checks.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001",
        help="Target API base URL.",
    )
    parser.add_argument(
        "--season-year",
        type=int,
        default=datetime.now().year,
        help="Season year for /coach/analyze smoke payload.",
    )
    parser.add_argument(
        "--coach-home-team",
        default="LG",
        help="home_team_id for /coach/analyze.",
    )
    parser.add_argument(
        "--coach-away-team",
        default="SSG",
        help="away_team_id for /coach/analyze.",
    )
    parser.add_argument(
        "--coach-request-mode",
        default="auto_brief",
        choices=["auto_brief", "manual_detail"],
        help="Request mode for /coach/analyze.",
    )
    parser.add_argument(
        "--coach-focus",
        default=None,
        help="Comma-separated focus list for /coach/analyze (manual_detail only).",
    )
    parser.add_argument(
        "--coach-question-override",
        default=None,
        help="Optional question_override for /coach/analyze (manual_detail).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON report output path.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="Optional compact summary JSON output path.",
    )
    parser.add_argument(
        "--chat-question-list",
        type=str,
        default=None,
        help="Path to newline-separated chat questions for /ai/chat/completion batch test.",
    )
    parser.add_argument(
        "--chat-batch-size",
        type=int,
        default=0,
        help="Number of batch questions to send to /ai/chat/completion. 0 disables batch run.",
    )
    parser.add_argument(
        "--chat-request-delay",
        type=float,
        default=0.0,
        help="Delay in seconds between batch completion requests.",
    )
    parser.add_argument(
        "--rate-limit-max-retries",
        type=int,
        default=8,
        help="Retry attempts when HTTP 429 Rate Limit is returned.",
    )
    parser.add_argument(
        "--rate-limit-base-delay",
        type=float,
        default=1.5,
        help="Minimum delay in seconds used when retrying rate-limited requests.",
    )
    parser.add_argument(
        "--internal-api-key",
        default=os.getenv("AI_INTERNAL_TOKEN", ""),
        help="Value for X-Internal-Api-Key when calling protected AI endpoints.",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Exit with code 1 if any endpoint fails (default).",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Always exit with code 0 even if checks fail.",
    )
    return parser.parse_args()


def _default_chat_questions() -> List[str]:
    teams = ["LG", "KT", "SSG", "두산", "한화", "롯데", "삼성", "KIA", "키움", "NC"]
    topics = [
        "2025 시즌 타격 라인업 장단점",
        "최근 5경기 득점 흐름",
        "상대 전적을 기준으로한 승리 포인트",
        "핵심 타자 3인과 컨디션",
        "선발 투수 라인업 분석",
        "불펜 안정성 평가",
        "수비 실책 리스크 정리",
        "플레이오프 가능성",
        "강점과 약점 한 줄 요약",
        "다음 경기 주요 주전 변수",
    ]

    return [f"{team} 야구부 팀 분석: {topic}" for team in teams for topic in topics]


def _load_chat_questions(path: str | None, limit: int) -> List[str]:
    if path:
        p = Path(path)
        questions = [
            line.strip() for line in p.read_text(encoding="utf-8").splitlines()
        ]
        questions = [line for line in questions if line]
        if not questions:
            raise ValueError(f"No questions found in {p}")
        source = "file"
    else:
        questions = _default_chat_questions()
        source = "default"

    if limit <= 0:
        raise ValueError("chat_batch_size must be > 0 when batch mode is enabled.")
    if len(questions) < limit:
        raise ValueError(
            f"{source} question source only has {len(questions)} questions, but {limit} were requested."
        )
    return questions[:limit]


def _build_internal_headers(token: str) -> Dict[str, str]:
    return {"X-Internal-Api-Key": token} if token else {}


def _get_retry_delay(response: httpx.Response, fallback: float) -> float:
    raw = response.headers.get("Retry-After")
    if not raw:
        return fallback
    try:
        return float(raw)
    except ValueError:
        # If Retry-After is a full date format, fallback to configured delay.
        return fallback


def _truncate(text: str, limit: int = 600) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text


def check_health(client: httpx.Client, base_url: str) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.get(f"{base_url}/health")
        payload = _safe_json(response)
        ok = (
            response.status_code == 200
            and isinstance(payload, dict)
            and payload.get("status") == "ok"
        )
        return {
            "endpoint": "/health",
            "status_code": response.status_code,
            "ok": ok,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": (
                None if ok else f"unexpected_response:{_truncate(str(payload), 240)}"
            ),
            "sample_response": payload,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "endpoint": "/health",
            "status_code": None,
            "ok": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": str(exc),
            "sample_response": None,
        }


def check_completion(
    client: httpx.Client,
    base_url: str,
    question: str,
    *,
    index: int | None = None,
    headers: Dict[str, str] | None = None,
    rate_limit_retries: int = 8,
    rate_limit_base_delay: float = 1.5,
) -> Dict[str, Any]:
    attempts_left = max(1, rate_limit_retries + 1)
    payload = {"question": question}
    last_exception: str | None = None

    for attempt in range(1, attempts_left + 1):
        started = time.perf_counter()
        try:
            response = client.post(
                f"{base_url}/ai/chat/completion",
                json=payload,
                headers=headers,
            )

            if response.status_code == 429 and attempt <= rate_limit_retries:
                delay = _get_retry_delay(response, rate_limit_base_delay)
                time.sleep(max(delay, rate_limit_base_delay))
                continue

            body = _safe_json(response)
            ok = (
                response.status_code == 200
                and isinstance(body, dict)
                and isinstance(body.get("answer"), str)
            )
            return {
                "endpoint": "/ai/chat/completion",
                "question_index": index,
                "question": question,
                "status_code": response.status_code,
                "ok": ok,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": (
                    None if ok else f"unexpected_response:{_truncate(str(body), 240)}"
                ),
                "sample_response": body,
                "attempt": attempt,
            }
        except Exception as exc:  # noqa: BLE001
            last_exception = str(exc)
            if attempt > rate_limit_retries:
                return {
                    "endpoint": "/ai/chat/completion",
                    "question_index": index,
                    "question": question,
                    "status_code": None,
                    "ok": False,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error": last_exception,
                    "sample_response": None,
                    "attempt": attempt,
                }
            time.sleep(rate_limit_base_delay)

    return {
        "endpoint": "/ai/chat/completion",
        "question_index": index,
        "question": question,
        "status_code": None,
        "ok": False,
        "latency_ms": 0.0,
        "error": last_exception or "unexpected_error",
        "sample_response": None,
        "attempt": attempts_left,
    }


def _check_stream_endpoint(
    client: httpx.Client,
    *,
    endpoint_url: str,
    endpoint_name: str,
    payload: Dict[str, Any],
    require_order: bool = False,
    headers: Dict[str, str] | None = None,
    rate_limit_retries: int = 8,
    rate_limit_base_delay: float = 1.5,
) -> Dict[str, Any]:
    attempts_left = max(1, rate_limit_retries + 1)
    last_exception: str | None = None
    for attempt in range(1, attempts_left + 1):
        started = time.perf_counter()
        seen_message = False
        seen_meta = False
        seen_done = False
        event_positions: Dict[str, int] = {}
        status_code: int | None = None
        sample_lines: List[str] = []
        error: str | None = None

        try:
            request_headers = {"Accept": "text/event-stream"}
            if headers:
                request_headers.update(headers)
            with client.stream(
                "POST",
                endpoint_url,
                json=payload,
                headers=request_headers,
            ) as response:
                status_code = response.status_code

                if status_code == 429 and attempt <= rate_limit_retries:
                    delay = _get_retry_delay(response, rate_limit_base_delay)
                    time.sleep(max(delay, rate_limit_base_delay))
                    continue

                if status_code != 200:
                    error = f"unexpected_status:{status_code}"
                else:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        sample_lines.append(line)
                        if len(sample_lines) > 24:
                            sample_lines = sample_lines[-24:]

                        if line.startswith("event: "):
                            event_name = line.split(":", 1)[1].strip()
                            if event_name not in event_positions:
                                event_positions[event_name] = len(sample_lines)
                            if event_name == "message":
                                seen_message = True
                            elif event_name == "meta":
                                seen_meta = True
                            elif event_name == "done":
                                seen_done = True
                                break

                    if require_order:
                        status_pos = event_positions.get("status")
                        message_pos = event_positions.get("message")
                        meta_pos = event_positions.get("meta")
                        done_pos = event_positions.get("done")
                        if status_pos is None:
                            error = "missing_status_event"
                        elif message_pos is None:
                            error = "missing_message_event"
                        elif meta_pos is None:
                            error = "missing_meta_event"
                        elif done_pos is None:
                            error = "missing_done_event"
                        elif not (status_pos < message_pos < meta_pos < done_pos):
                            error = "invalid_event_order"
                    elif not seen_done:
                        error = "missing_done_event"
                    elif not seen_message:
                        error = "missing_message_event"
                    elif not seen_meta:
                        error = "missing_meta_event"
        except Exception as exc:  # noqa: BLE001
            last_exception = str(exc)
            if attempt > rate_limit_retries:
                error = str(exc)
            else:
                time.sleep(rate_limit_base_delay)

        if error is None:
            return {
                "endpoint": endpoint_name,
                "status_code": status_code,
                "ok": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": None,
                "sample_response": _truncate("\n".join(sample_lines)),
                "attempt": attempt,
            }
        if status_code != 429:
            break

    return {
        "endpoint": endpoint_name,
        "status_code": status_code,
        "ok": False,
        "latency_ms": (
            None
            if last_exception is None
            else round((time.perf_counter() - started) * 1000, 2)
        ),
        "error": error or last_exception,
        "sample_response": _truncate("\n".join(sample_lines)),
        "attempt": attempt,
    }


def check_chat_stream(
    client: httpx.Client,
    base_url: str,
    headers: Dict[str, str] | None = None,
    *,
    rate_limit_retries: int = 8,
    rate_limit_base_delay: float = 1.5,
) -> Dict[str, Any]:
    return _check_stream_endpoint(
        client,
        endpoint_url=f"{base_url}/ai/chat/stream?style=compact",
        endpoint_name="/ai/chat/stream",
        payload={"question": "LG 트윈스 2025 시즌 요약해줘"},
        headers=headers,
        rate_limit_retries=rate_limit_retries,
        rate_limit_base_delay=rate_limit_base_delay,
    )


def check_coach_stream(
    client: httpx.Client,
    base_url: str,
    season_year: int,
    *,
    require_order: bool = True,
    home_team: str = "LG",
    away_team: str | None = None,
    request_mode: str = "auto_brief",
    focus: str | None = None,
    question_override: str | None = None,
    headers: Dict[str, str] | None = None,
    rate_limit_retries: int = 8,
    rate_limit_base_delay: float = 1.5,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "home_team_id": home_team,
        "league_context": {
            "season_year": season_year,
            "league_type": "REGULAR",
        },
        "request_mode": request_mode,
    }
    if away_team:
        payload["away_team_id"] = away_team
    if focus:
        payload["focus"] = [item.strip() for item in focus.split(",") if item.strip()]
    if question_override:
        payload["question_override"] = question_override
    return _check_stream_endpoint(
        client,
        endpoint_url=f"{base_url}/coach/analyze",
        endpoint_name="/coach/analyze",
        payload=payload,
        require_order=require_order,
        headers=headers,
        rate_limit_retries=rate_limit_retries,
        rate_limit_base_delay=rate_limit_base_delay,
    )


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, int]:
    total = len(results)
    passed = sum(1 for item in results if item.get("ok"))
    failed = total - passed
    return {"total": total, "passed": passed, "failed": failed}


def _build_console_summary(summary: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    return {"summary": dict(summary)}


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    timeout = httpx.Timeout(args.timeout, connect=min(10.0, args.timeout))
    default_headers = _build_internal_headers(args.internal_api_key)

    with httpx.Client(timeout=timeout) as client:
        results = [
            check_health(client, base_url),
            check_completion(
                client,
                base_url,
                "LG 트윈스 2025 시즌 요약해줘",
                headers=default_headers,
                rate_limit_retries=args.rate_limit_max_retries,
                rate_limit_base_delay=args.rate_limit_base_delay,
            ),
            check_chat_stream(
                client,
                base_url,
                headers=default_headers,
                rate_limit_retries=args.rate_limit_max_retries,
                rate_limit_base_delay=args.rate_limit_base_delay,
            ),
            check_coach_stream(
                client,
                base_url,
                args.season_year,
                home_team=args.coach_home_team,
                away_team=args.coach_away_team,
                request_mode=args.coach_request_mode,
                focus=args.coach_focus,
                question_override=args.coach_question_override,
                headers=default_headers,
                rate_limit_retries=args.rate_limit_max_retries,
                rate_limit_base_delay=args.rate_limit_base_delay,
            ),
        ]

        if args.chat_batch_size > 0:
            batch_questions = _load_chat_questions(
                args.chat_question_list,
                args.chat_batch_size,
            )
            for i, question in enumerate(batch_questions):
                results.append(
                    check_completion(
                        client,
                        base_url,
                        question,
                        index=i + 1,
                        headers=default_headers,
                        rate_limit_retries=args.rate_limit_max_retries,
                        rate_limit_base_delay=args.rate_limit_base_delay,
                    )
                )
                if args.chat_request_delay > 0:
                    time.sleep(args.chat_request_delay)

    summary = build_summary(results)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "base_url": base_url,
            "season_year": args.season_year,
            "timeout": args.timeout,
            "chat_batch_size": args.chat_batch_size,
            "chat_question_list": args.chat_question_list,
            "chat_request_delay": args.chat_request_delay,
            "rate_limit_max_retries": args.rate_limit_max_retries,
            "rate_limit_base_delay": args.rate_limit_base_delay,
            "strict": args.strict,
        },
        "results": results,
        "summary": summary,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.summary_output:
        summary_output = Path(args.summary_output)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(_build_console_summary(summary), ensure_ascii=False))

    if args.output:
        print(f"Wrote report: {output_path}")

    if args.summary_output:
        print(f"Wrote summary: {summary_output}")

    if args.strict and summary["failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
