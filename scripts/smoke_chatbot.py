#!/usr/bin/env python3
"""Run live smoke checks for chatbot and coach endpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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


def check_completion(client: httpx.Client, base_url: str) -> Dict[str, Any]:
    started = time.perf_counter()
    payload = {"question": "LG 트윈스 2025 시즌 요약해줘"}
    try:
        response = client.post(f"{base_url}/ai/chat/completion", json=payload)
        body = _safe_json(response)
        ok = (
            response.status_code == 200
            and isinstance(body, dict)
            and isinstance(body.get("answer"), str)
        )
        return {
            "endpoint": "/ai/chat/completion",
            "status_code": response.status_code,
            "ok": ok,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": None if ok else f"unexpected_response:{_truncate(str(body), 240)}",
            "sample_response": body,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "endpoint": "/ai/chat/completion",
            "status_code": None,
            "ok": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": str(exc),
            "sample_response": None,
        }


def _check_stream_endpoint(
    client: httpx.Client,
    *,
    endpoint_url: str,
    endpoint_name: str,
    payload: Dict[str, Any],
    require_order: bool = False,
) -> Dict[str, Any]:
    started = time.perf_counter()
    seen_message = False
    seen_meta = False
    seen_done = False
    event_positions: Dict[str, int] = {}
    status_code: int | None = None
    sample_lines: List[str] = []
    error: str | None = None

    try:
        with client.stream(
            "POST",
            endpoint_url,
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            status_code = response.status_code
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

            if status_code != 200:
                error = f"unexpected_status:{status_code}"
            elif require_order:
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
        error = str(exc)

    return {
        "endpoint": endpoint_name,
        "status_code": status_code,
        "ok": error is None,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "error": error,
        "sample_response": _truncate("\n".join(sample_lines)),
    }


def check_chat_stream(client: httpx.Client, base_url: str) -> Dict[str, Any]:
    return _check_stream_endpoint(
        client,
        endpoint_url=f"{base_url}/ai/chat/stream?style=compact",
        endpoint_name="/ai/chat/stream",
        payload={"question": "LG 트윈스 2025 시즌 요약해줘"},
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
    )


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, int]:
    total = len(results)
    passed = sum(1 for item in results if item.get("ok"))
    failed = total - passed
    return {"total": total, "passed": passed, "failed": failed}


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    timeout = httpx.Timeout(args.timeout, connect=min(10.0, args.timeout))

    with httpx.Client(timeout=timeout) as client:
        results = [
            check_health(client, base_url),
            check_completion(client, base_url),
            check_chat_stream(client, base_url),
            check_coach_stream(
                client,
                base_url,
                args.season_year,
                home_team=args.coach_home_team,
                away_team=args.coach_away_team,
                request_mode=args.coach_request_mode,
                focus=args.coach_focus,
                question_override=args.coach_question_override,
            ),
        ]

    summary = build_summary(results)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "base_url": base_url,
            "season_year": args.season_year,
            "timeout": args.timeout,
            "strict": args.strict,
        },
        "results": results,
        "summary": summary,
    }

    print(json.dumps({"summary": summary}, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote report: {output_path}")

    if args.strict and summary["failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
