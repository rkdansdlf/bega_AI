#!/usr/bin/env python3
"""Run factual canary checks against chatbot completion and stream endpoints."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx


DEFAULT_CASES_PATH = (
    Path(__file__).resolve().with_name("smoke_chatbot_factual_canary_cases.json")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run factual chatbot canary checks.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--case-file", default=str(DEFAULT_CASES_PATH))
    parser.add_argument(
        "--internal-api-key",
        default=os.getenv("AI_INTERNAL_TOKEN", ""),
        help="Value for X-Internal-Api-Key.",
    )
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument(
        "--stream-style",
        choices=("markdown", "json", "compact"),
        default="compact",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--summary-output", type=str, default=None)
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    return parser.parse_args()


def _load_cases(path: str) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("case file must be a JSON array")
    return [item for item in payload if isinstance(item, dict)]


def _build_history_payload(question: str) -> List[Dict[str, str]]:
    nonce = datetime.now(timezone.utc).isoformat()
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"cache-bypass:{nonce}"},
    ]


def _check_completion(
    client: httpx.Client,
    base_url: str,
    question: str,
    *,
    headers: Dict[str, str],
    history_payload: List[Dict[str, str]] | None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"question": question}
    if history_payload:
        payload["history"] = history_payload

    start = time.perf_counter()
    response = client.post(
        f"{base_url}/ai/chat/completion",
        json=payload,
        headers=headers,
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    answer = None
    error = None
    if response.status_code == 200:
        body = response.json()
        answer = body.get("answer")
    else:
        error = f"status={response.status_code}"

    return {
        "endpoint": "/ai/chat/completion",
        "status_code": response.status_code,
        "latency_ms": latency_ms,
        "answer": answer,
        "error": error,
    }


def _check_stream(
    client: httpx.Client,
    base_url: str,
    question: str,
    *,
    headers: Dict[str, str],
    history_payload: List[Dict[str, str]] | None,
    stream_style: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"question": question}
    if history_payload:
        payload["history"] = history_payload

    start = time.perf_counter()
    message_chunks: List[str] = []
    status_code: int | None = None
    current_event = ""
    seen_done = False
    error = None

    request_headers = {"Accept": "text/event-stream"}
    request_headers.update(headers)

    with client.stream(
        "POST",
        f"{base_url}/ai/chat/stream?style={stream_style}",
        json=payload,
        headers=request_headers,
    ) as response:
        status_code = response.status_code
        if response.status_code != 200:
            error = f"status={response.status_code}"
        else:
            for line in response.iter_lines():
                if not line:
                    continue
                stripped = line.strip()
                if stripped.startswith("event:"):
                    current_event = stripped.split(":", 1)[1].strip()
                    continue
                if not stripped.startswith("data:"):
                    continue
                data = stripped[5:].strip()
                if data == "[DONE]":
                    seen_done = True
                    continue
                if current_event == "message":
                    try:
                        payload_data = json.loads(data)
                        delta = payload_data.get("delta")
                        if delta:
                            message_chunks.append(str(delta))
                    except Exception:
                        message_chunks.append(data)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    if status_code == 200 and not seen_done:
        error = "stream_ended_without_done"

    return {
        "endpoint": "/ai/chat/stream",
        "status_code": status_code,
        "latency_ms": latency_ms,
        "answer": "".join(message_chunks) if not error else None,
        "error": error,
    }


def _match_substrings(answer: str, needles: List[str]) -> Tuple[bool, List[str]]:
    missing = [needle for needle in needles if needle not in answer]
    return not missing, missing


def _match_any_substrings(answer: str, needles: List[str]) -> Tuple[bool, List[str]]:
    if not needles:
        return True, []
    if any(needle in answer for needle in needles):
        return True, []
    return False, needles


def _forbidden_substrings(answer: str, needles: List[str]) -> Tuple[bool, List[str]]:
    found = [needle for needle in needles if needle in answer]
    return not found, found


def _evaluate_answer(case: Dict[str, Any], answer: str | None) -> Dict[str, Any]:
    if not isinstance(answer, str) or not answer.strip():
        return {
            "ok": False,
            "checks": {
                "required_all_pass": False,
                "required_any_pass": False,
                "forbidden_pass": False,
            },
            "missing_required_all": list(case.get("required_all") or []),
            "missing_required_any": list(case.get("required_any") or []),
            "found_forbidden": [],
        }

    required_all = [str(item) for item in case.get("required_all") or []]
    required_any = [str(item) for item in case.get("required_any") or []]
    forbidden = [str(item) for item in case.get("forbidden") or []]

    required_all_pass, missing_required_all = _match_substrings(answer, required_all)
    required_any_pass, missing_required_any = _match_any_substrings(answer, required_any)
    forbidden_pass, found_forbidden = _forbidden_substrings(answer, forbidden)

    checks = {
        "required_all_pass": required_all_pass,
        "required_any_pass": required_any_pass,
        "forbidden_pass": forbidden_pass,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "missing_required_all": missing_required_all,
        "missing_required_any": missing_required_any,
        "found_forbidden": found_forbidden,
    }


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    passed = sum(1 for item in records if item.get("ok"))
    by_endpoint: Dict[str, Dict[str, int]] = {}
    for endpoint in ("/ai/chat/completion", "/ai/chat/stream"):
        endpoint_rows = [item for item in records if item.get("endpoint") == endpoint]
        endpoint_total = len(endpoint_rows)
        endpoint_passed = sum(1 for item in endpoint_rows if item.get("ok"))
        by_endpoint[endpoint] = {
            "total": endpoint_total,
            "passed": endpoint_passed,
            "failed": endpoint_total - endpoint_passed,
        }
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "completion": by_endpoint["/ai/chat/completion"],
        "stream": by_endpoint["/ai/chat/stream"],
    }


def main() -> int:
    args = parse_args()
    cases = _load_cases(args.case_file)
    headers = {}
    if args.internal_api_key:
        headers["X-Internal-Api-Key"] = args.internal_api_key

    records: List[Dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout) as client:
        for index, case in enumerate(cases, start=1):
            question = str(case["question"])
            history_payload = (
                _build_history_payload(question) if args.disable_cache else None
            )
            completion = _check_completion(
                client,
                args.base_url,
                question,
                headers=headers,
                history_payload=history_payload,
            )
            stream = _check_stream(
                client,
                args.base_url,
                question,
                headers=headers,
                history_payload=history_payload,
                stream_style=args.stream_style,
            )

            for response in (completion, stream):
                evaluation = _evaluate_answer(case, response.get("answer"))
                record = {
                    "case_id": case.get("id"),
                    "question": question,
                    "endpoint": response["endpoint"],
                    "status_code": response.get("status_code"),
                    "latency_ms": response.get("latency_ms"),
                    "error": response.get("error"),
                    "ok": response.get("error") is None and evaluation["ok"],
                    "checks": evaluation["checks"],
                    "missing_required_all": evaluation["missing_required_all"],
                    "missing_required_any": evaluation["missing_required_any"],
                    "found_forbidden": evaluation["found_forbidden"],
                    "answer": response.get("answer"),
                }
                records.append(record)

            c_ok = "O" if records[-2]["ok"] else "X"
            s_ok = "O" if records[-1]["ok"] else "X"
            print(
                f"[{index}/{len(cases)}] completion={c_ok}({completion.get('latency_ms')}ms) "
                f"stream={s_ok}({stream.get('latency_ms')}ms) {question}"
            )

    summary = _summarize(records)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "case_file": args.case_file,
            "base_url": args.base_url,
            "disable_cache": args.disable_cache,
            "stream_style": args.stream_style,
        },
        "summary": summary,
        "results": records,
    }

    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report saved: {output_path}")

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"summary saved: {summary_path}")

    return 0 if (summary["failed"] == 0 or not args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
