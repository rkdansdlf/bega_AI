#!/usr/bin/env python3
"""Run live chatbot quality checks for completion and stream endpoints.

Runs 100 sequential chatbot questions (or a provided question list) against
`/ai/chat/completion` and `/ai/chat/stream`.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx


SECTION_HEADERS = (
    "핵심",
    "요약",
    "전망",
    "분석",
    "정리",
    "결론",
    "핵심 결론",
    "요점",
)


TABLE_PATTERN = re.compile(r"(^|\n)\s*\|.+\|.+\|", re.MULTILINE)
SECTION_PATTERN = re.compile(r"(^|\n)\s*#{1,3}\s*([^\n]+)")
SOURCE_PATTERN = re.compile(r"(^|\n)\s*출처\s*[:：]", re.MULTILINE)
BRIEFING_INTRO_PATTERN = re.compile(r"^\s*질문하신\s*`", re.MULTILINE)
BRIEFING_SECTION_LINE_PATTERN = re.compile(
    r"^\s*(?:#{1,3}\s*)?"
    r"(핵심(?:\s*(?:요약|분석|결론|지표))?|요약|전망|분석|정리|결론|요점|상세 내역|핵심 지표|주요 지표|인사이트)"
    r"\s*(?:[:：]|$)"
)
RAW_CHUNK_MARKERS = (
    "README.md 쪽에서는",
    ".md 쪽에서는",
    "```",
)
LOW_DATA_FALLBACK_MARKERS = (
    "이번 조회만으로는 질문에 시원하게 답할 만큼 데이터가 충분히 안 붙었습니다.",
    "지금 조회로는 질문에 딱 맞는 규정 조항이 충분히 안 잡혔습니다.",
    "지금 일정 조회로는 질문에 바로 꽂히는 경기 데이터가 충분히 안 잡혔습니다.",
    "괜히 아는 척하지 않고 확인된 범위에서만 말씀드리겠습니다.",
)
DEFAULT_QUESTION_LIST_PATH = (
    Path(__file__).resolve().with_name("smoke_chatbot_quality_regmix_100.txt")
)
DEFAULT_REGULATION_CANARY_PATH = (
    Path(__file__).resolve().with_name("smoke_chatbot_quality_regulations_20.txt")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run live quality checks for chatbot endpoints. "
            "Default question set is regmix 100 (team 80 + regulation 20)."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001",
        help="Base URL for AI FastAPI service.",
    )
    parser.add_argument(
        "--chat-batch-size",
        type=int,
        default=100,
        help="Number of questions to run (default: 100).",
    )
    parser.add_argument(
        "--chat-question-list",
        type=str,
        default=None,
        help=(
            "Optional path to newline-separated questions file. "
            f"Default: {DEFAULT_QUESTION_LIST_PATH}. "
            f"Regulation canary: {DEFAULT_REGULATION_CANARY_PATH}."
        ),
    )
    parser.add_argument(
        "--internal-api-key",
        default=os.getenv("AI_INTERNAL_TOKEN", ""),
        help="Value for X-Internal-Api-Key.",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Attach minimal history to bypass server cache and force live model calls.",
    )
    parser.add_argument(
        "--stream-style",
        choices=("markdown", "json", "compact"),
        default="markdown",
        help="Style parameter for /ai/chat/stream requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout seconds.",
    )
    parser.add_argument(
        "--rate-limit-max-retries",
        type=int,
        default=8,
        help="Retry count on 429 responses.",
    )
    parser.add_argument(
        "--rate-limit-base-delay",
        type=float,
        default=1.5,
        help="Base backoff for 429 and retry exceptions.",
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
        "--lock-file",
        type=str,
        default="/tmp/smoke_chatbot_quality.lock",
        help="Path to single-run lock file.",
    )
    parser.add_argument(
        "--fail-if-locked",
        dest="fail_if_locked",
        action="store_true",
        default=True,
        help="Exit with code 2 when another smoke run lock is active (default).",
    )
    parser.add_argument(
        "--wait-if-locked",
        dest="fail_if_locked",
        action="store_false",
        help="Wait until lock is released instead of exiting.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing --output report file if present.",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Exit with failure when any check fails (default).",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Always exit 0 even if checks fail.",
    )
    parser.add_argument(
        "--stream-fallback-ratio-max",
        type=float,
        default=0.40,
        help="Maximum allowed fallback ratio for stream endpoint (default: 0.40).",
    )
    return parser.parse_args()


def _default_questions() -> List[str]:
    if DEFAULT_QUESTION_LIST_PATH.exists():
        questions = [
            line.strip()
            for line in DEFAULT_QUESTION_LIST_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if questions:
            return questions

    teams = [
        "LG",
        "KT",
        "SSG",
        "두산",
        "한화",
        "롯데",
        "삼성",
        "KIA",
        "키움",
        "NC",
    ]
    templates = [
        "{team} 요즘 시즌 흐름 팬 눈높이로 한 번에 정리해줘.",
        "{team} 최근 5경기 폼, 솔직히 올라오는 중이야 내려가는 중이야?",
        "{team} 최근 10경기 득점 보면 타선 살아난 거야 식은 거야?",
        "{team} 선발진 지금 믿고 가도 되는 상태인지 냉정하게 봐줘.",
        "{team} 타선이 터질 때랑 침묵할 때 패턴 좀 현실적으로 짚어줘.",
        "{team} 불펜 요즘 갈아 넣는 중이야? 필승조 퍼진 신호 있는지 봐줘.",
        "{team} 상대 전적 보면 누구만 만나면 유독 꼬이는지 알려줘.",
        "{team} 수비 실책 때문에 날린 경기들, 뼈아픈 장면 위주로 정리해줘.",
        "{team} 큰 경기만 가면 흔들리는 구간이 어디인지 콕 집어줘.",
        "{team} 이 페이스면 가을야구 갈 수 있어? 팬 입장에서 현실적으로 말해줘.",
    ]
    return [template.format(team=team) for team in teams for template in templates]


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_lock_info(lock_path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _acquire_execution_lock(
    lock_path: Path, *, run_id: str, fail_if_locked: bool
) -> None:
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            lock_payload = {
                "pid": os.getpid(),
                "run_id": run_id,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            os.write(
                fd,
                (json.dumps(lock_payload, ensure_ascii=False) + "\n").encode("utf-8"),
            )
            os.close(fd)
            return
        except FileExistsError:
            lock_info = _read_lock_info(lock_path)
            existing_pid = lock_info.get("pid")
            if isinstance(existing_pid, int) and _pid_alive(existing_pid):
                message = (
                    f"lock exists: {lock_path} pid={existing_pid} "
                    f"run_id={lock_info.get('run_id', 'unknown')}"
                )
                if fail_if_locked:
                    raise RuntimeError(message)
                time.sleep(1.0)
                continue

            try:
                lock_path.unlink()
            except FileNotFoundError:
                continue
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"failed to recover stale lock: {lock_path}: {exc}"
                ) from exc


def _release_execution_lock(lock_path: Path) -> None:
    if not lock_path.exists():
        return
    lock_info = _read_lock_info(lock_path)
    lock_pid = lock_info.get("pid")
    if isinstance(lock_pid, int) and lock_pid != os.getpid():
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


def _load_questions(path: str | None, limit: int) -> List[str]:
    if not path:
        questions = _default_questions()
    else:
        data = Path(path).read_text(encoding="utf-8").splitlines()
        questions = [line.strip() for line in data if line.strip()]
        if not questions:
            raise ValueError(f"No questions found in {path}")
    if limit <= 0:
        raise ValueError("chat-batch-size must be greater than 0")
    if len(questions) < limit:
        raise ValueError(
            f"Question source has {len(questions)} questions, but {limit} requested."
        )
    return questions[:limit]


def _build_internal_headers(token: str) -> Dict[str, str]:
    return {"X-Internal-Api-Key": token} if token else {}


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text


def _retry_delay(response: httpx.Response, fallback: float) -> float:
    raw = response.headers.get("Retry-After")
    if not raw:
        return fallback
    try:
        return float(raw)
    except ValueError:
        return fallback


def _transient_backoff(base_delay: float, attempt: int) -> float:
    safe_base = max(0.2, float(base_delay))
    return min(safe_base * (2 ** max(0, attempt - 1)), 30.0)


def _is_transient_status(status_code: int) -> bool:
    return status_code in {429, 500, 502, 503, 504}


def _is_transient_error_message(message: str) -> bool:
    normalized = (message or "").lower()
    markers = (
        "connection refused",
        "failed to connect",
        "connection reset",
        "temporarily unavailable",
        "server disconnected",
        "timed out",
        "timeout",
        "broken pipe",
    )
    return any(marker in normalized for marker in markers)


def _has_table(text: str) -> bool:
    return bool(TABLE_PATTERN.search(text))


def _has_section(text: str) -> bool:
    normalized = re.sub(r"\*{1,2}", "", text)
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if BRIEFING_SECTION_LINE_PATTERN.match(line):
            return True
        heading_match = SECTION_PATTERN.match(line)
        if heading_match:
            title = heading_match.group(2).strip()
            if any(section in title for section in SECTION_HEADERS):
                return True
    return False


def _has_source(text: str) -> bool:
    return bool(SOURCE_PATTERN.search(text)) or "출처" in text


def _has_raw_chunk_marker(text: str) -> bool:
    return any(marker in (text or "") for marker in RAW_CHUNK_MARKERS)


def _has_briefing_intro(text: str) -> bool:
    return bool(BRIEFING_INTRO_PATTERN.search(text or ""))


def _has_low_data_fallback(text: str) -> bool:
    return any(marker in (text or "") for marker in LOW_DATA_FALLBACK_MARKERS)


def _evaluate_quality(text: str) -> Dict[str, bool]:
    no_table_markup = not _has_table(text)
    no_briefing_headers = not _has_section(text)
    no_source_line = not _has_source(text)
    no_raw_chunk_marker = not _has_raw_chunk_marker(text)
    no_briefing_intro = not _has_briefing_intro(text)
    no_low_data_fallback = not _has_low_data_fallback(text)
    natural_chat = all(
        (
            no_table_markup,
            no_briefing_headers,
            no_source_line,
            no_raw_chunk_marker,
            no_briefing_intro,
        )
    )
    return {
        "natural_chat": natural_chat,
        "no_table_markup": no_table_markup,
        "no_briefing_headers": no_briefing_headers,
        "no_source_line": no_source_line,
        "no_raw_chunk_marker": no_raw_chunk_marker,
        "no_briefing_intro": no_briefing_intro,
        "no_low_data_fallback": no_low_data_fallback,
    }


def _quality_pass(quality: Dict[str, bool]) -> bool:
    return bool(quality.get("natural_chat")) and bool(
        quality.get("no_low_data_fallback")
    )


def _is_fallback_answer(text: str) -> bool:
    return _has_low_data_fallback(text or "")


def _empty_quality() -> Dict[str, bool]:
    return _evaluate_quality("")


def _fallback_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    fallback_count = sum(
        1 for item in items if _is_fallback_answer(str(item.get("answer", "")))
    )
    ratio = (fallback_count / total) if total else 0.0
    return {
        "total": total,
        "fallback_count": fallback_count,
        "fallback_ratio": round(ratio, 4),
    }


def _check_completion(
    client: httpx.Client,
    base_url: str,
    question: str,
    *,
    history_payload: Optional[List[Dict[str, str]]],
    headers: Dict[str, str],
    rate_limit_retries: int,
    rate_limit_base_delay: float,
) -> Dict[str, Any]:
    attempts = max(1, rate_limit_retries + 1)
    payload: Dict[str, Any] = {"question": question}
    if history_payload:
        payload["history"] = history_payload
    start = 0.0
    last_error: str | None = None

    for attempt in range(1, attempts + 1):
        start = time.perf_counter()
        try:
            response = client.post(f"{base_url}/ai/chat/completion", json=payload, headers=headers)
            if _is_transient_status(response.status_code) and attempt < attempts:
                if response.status_code == 429:
                    delay = max(_retry_delay(response, rate_limit_base_delay), rate_limit_base_delay)
                else:
                    delay = _transient_backoff(rate_limit_base_delay, attempt)
                time.sleep(delay)
                continue
            body = _safe_json(response)
            if response.status_code != 200 or not isinstance(body, dict):
                last_error = f"status={response.status_code}"
                return {
                    "endpoint": "/ai/chat/completion",
                    "question": question,
                    "status_code": response.status_code,
                    "ok": False,
                    "error": last_error,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                    "answer": None,
                    "quality": _empty_quality(),
                    "quality_pass": False,
                    "attempt": attempt,
                    "cached": False,
                    "sample_response": body,
                }

            answer = body.get("answer")
            if not isinstance(answer, str):
                answer = str(answer)
            quality = _evaluate_quality(answer)
            return {
                "endpoint": "/ai/chat/completion",
                "question": question,
                "status_code": response.status_code,
                "ok": _quality_pass(quality),
                "error": None,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "answer": answer,
                "quality": quality,
                "quality_pass": _quality_pass(quality),
                "attempt": attempt,
                "cached": bool(body.get("cached", False)),
                "sample_response": body,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt >= attempts:
                break
            delay = _transient_backoff(rate_limit_base_delay, attempt)
            if not _is_transient_error_message(last_error):
                delay = max(0.2, float(rate_limit_base_delay))
            time.sleep(delay)

    return {
        "endpoint": "/ai/chat/completion",
        "question": question,
        "status_code": None,
        "ok": False,
        "error": last_error,
        "latency_ms": None,
        "answer": None,
        "quality": _empty_quality(),
        "quality_pass": False,
        "attempt": attempts,
        "cached": False,
        "sample_response": None,
    }


def _check_stream(
    client: httpx.Client,
    base_url: str,
    question: str,
    *,
    history_payload: Optional[List[Dict[str, str]]],
    headers: Dict[str, str],
    rate_limit_retries: int,
    rate_limit_base_delay: float,
    stream_style: str,
) -> Dict[str, Any]:
    attempts = max(1, rate_limit_retries + 1)
    payload: Dict[str, Any] = {"question": question}
    if history_payload:
        payload["history"] = history_payload
    last_error: str | None = None

    for attempt in range(1, attempts + 1):
        start = time.perf_counter()
        line_index = 0
        status_code: int | None = None
        event_positions: Dict[str, int] = {}
        current_event = ""
        message_chunks: List[str] = []
        sample_lines: List[str] = []
        seen_done = False
        meta_payload: Dict[str, Any] = {}

        try:
            request_headers = {"Accept": "text/event-stream"}
            request_headers.update(headers)

            with client.stream(
                "POST",
                f"{base_url}/ai/chat/stream?style={stream_style}",
                json=payload,
                headers=request_headers,
            ) as response:
                status_code = response.status_code

                if _is_transient_status(response.status_code) and attempt < attempts:
                    if response.status_code == 429:
                        delay = max(_retry_delay(response, rate_limit_base_delay), rate_limit_base_delay)
                    else:
                        delay = _transient_backoff(rate_limit_base_delay, attempt)
                    time.sleep(delay)
                    continue

                if response.status_code != 200:
                    return {
                        "endpoint": "/ai/chat/stream",
                        "question": question,
                        "status_code": response.status_code,
                        "ok": False,
                        "error": f"status={response.status_code}",
                        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                        "answer": None,
                        "quality": _empty_quality(),
                        "quality_pass": False,
                        "attempt": attempt,
                        "event_order_ok": False,
                        "cached": False,
                        "sample_response": "",
                    }

                for line in response.iter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    sample_lines.append(line)

                    if line.startswith("event:"):
                        current_event = line.split(":", 1)[1].strip()
                        event_positions.setdefault(current_event, line_index)
                        line_index += 1
                        continue

                    if line.startswith("data:"):
                        data = line[5:].strip()
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
                        elif current_event == "meta":
                            try:
                                meta_payload = json.loads(data)
                            except Exception:
                                meta_payload = {}
                        line_index += 1

                        if current_event == "done":
                            break

            if seen_done:
                text = "".join(message_chunks)
                quality = _evaluate_quality(text)
                event_order = event_positions.get("status", 1e9)
                event_message = event_positions.get("message", 1e9)
                event_meta = event_positions.get("meta", 1e9)
                event_done = event_positions.get("done", 1e9)
                event_order_ok = (
                    event_order < event_message < event_meta < event_done
                    if all(pos != 1e9 for pos in (event_order, event_message, event_meta, event_done))
                    else False
                )
                quality_ok = _quality_pass(quality)

                return {
                    "endpoint": "/ai/chat/stream",
                    "question": question,
                    "status_code": status_code,
                    "ok": event_order_ok and quality_ok,
                    "error": None,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                    "answer": text,
                    "quality": quality,
                    "quality_pass": quality_ok,
                    "attempt": attempt,
                    "event_order_ok": event_order_ok,
                    "cached": bool(meta_payload.get("cached", False)),
                    "meta": meta_payload,
                    "sample_response": "\\n".join(sample_lines[-24:]),
                }

            last_error = "stream_ended_without_done"
            if attempt < attempts:
                time.sleep(_transient_backoff(rate_limit_base_delay, attempt))
                continue
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt >= attempts:
                break
            delay = _transient_backoff(rate_limit_base_delay, attempt)
            if not _is_transient_error_message(last_error):
                delay = max(0.2, float(rate_limit_base_delay))
            time.sleep(delay)

    return {
        "endpoint": "/ai/chat/stream",
        "question": question,
        "status_code": status_code,
        "ok": False,
        "error": last_error,
        "latency_ms": None,
        "answer": None,
        "quality": _empty_quality(),
        "quality_pass": False,
        "attempt": attempts,
        "event_order_ok": False,
        "cached": False,
        "meta": {},
        "sample_response": "",
    }


def _print_progress(index: int, total: int, completion_result: Dict[str, Any], stream_result: Dict[str, Any]) -> None:
    c_ok = "O" if completion_result.get("ok") else "X"
    s_ok = "O" if stream_result.get("ok") else "X"
    print(
        f"[{index}/{total}] completion={c_ok}({completion_result.get('latency_ms')}ms) "
        f"stream={s_ok}({stream_result.get('latency_ms')}ms)"
    )


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _percentile(sorted_values: List[float], percentile: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return round(sorted_values[0], 2)

    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return round(sorted_values[lower], 2)
    weight = rank - lower
    value = sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight
    return round(value, 2)


def _is_timeout_result(item: Dict[str, Any]) -> bool:
    status_code = item.get("status_code")
    if status_code in (408, 504):
        return True
    error = str(item.get("error") or "").lower()
    timeout_tokens = ("timeout", "timed out", "read timeout", "connect timeout")
    return any(token in error for token in timeout_tokens)


def _endpoint_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    passed = sum(1 for item in items if item.get("ok"))
    failed = total - passed
    error_count = failed
    timeout_count = sum(1 for item in items if _is_timeout_result(item))
    latency_values = sorted(
        float(item["latency_ms"])
        for item in items
        if isinstance(item.get("latency_ms"), (int, float))
    )

    latency_summary = {
        "count": len(latency_values),
        "min": round(latency_values[0], 2) if latency_values else None,
        "max": round(latency_values[-1], 2) if latency_values else None,
        "avg": (
            round(sum(latency_values) / len(latency_values), 2) if latency_values else None
        ),
        "p50": _percentile(latency_values, 0.50),
        "p95": _percentile(latency_values, 0.95),
        "p99": _percentile(latency_values, 0.99),
    }

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": _ratio(passed, total),
        "error_rate": _ratio(error_count, total),
        "timeout_rate": _ratio(timeout_count, total),
        "latency_ms": latency_summary,
        "quality_natural_chat_passed": sum(
            1 for item in items if item.get("quality", {}).get("natural_chat")
        ),
        "quality_no_table_markup_passed": sum(
            1 for item in items if item.get("quality", {}).get("no_table_markup")
        ),
        "quality_no_briefing_headers_passed": sum(
            1 for item in items if item.get("quality", {}).get("no_briefing_headers")
        ),
        "quality_no_source_line_passed": sum(
            1 for item in items if item.get("quality", {}).get("no_source_line")
        ),
        "quality_no_raw_chunk_marker_passed": sum(
            1 for item in items if item.get("quality", {}).get("no_raw_chunk_marker")
        ),
        "quality_no_low_data_fallback_passed": sum(
            1 for item in items if item.get("quality", {}).get("no_low_data_fallback")
        ),
    }


def _extract_planner_mode(item: Dict[str, Any]) -> str:
    mode: Any = None
    sample_response = item.get("sample_response")
    if isinstance(sample_response, dict):
        mode = sample_response.get("planner_mode")
        if not isinstance(mode, str):
            metadata = sample_response.get("metadata")
            if isinstance(metadata, dict):
                mode = metadata.get("planner_mode")
    if not isinstance(mode, str):
        meta = item.get("meta")
        if isinstance(meta, dict):
            mode = meta.get("planner_mode")
            if not isinstance(mode, str):
                perf = meta.get("perf")
                if isinstance(perf, dict):
                    mode = perf.get("planner_mode")
    if isinstance(mode, str) and mode.strip():
        return mode.strip()
    return "unknown"


def _planner_mode_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        mode = _extract_planner_mode(item)
        grouped.setdefault(mode, []).append(item)
    return {mode: _endpoint_metrics(group) for mode, group in grouped.items()}


def _summarize_results(
    results: List[Dict[str, Any]], *, stream_fallback_ratio_max: float
) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for item in results if item.get("ok"))
    failed = total - passed
    completion = [item for item in results if item.get("endpoint") == "/ai/chat/completion"]
    stream = [item for item in results if item.get("endpoint") == "/ai/chat/stream"]
    completion_fallback_metrics = _fallback_metrics(completion)
    stream_fallback_metrics = _fallback_metrics(stream)
    stream_fallback_ratio_ok = (
        stream_fallback_metrics["fallback_ratio"] <= stream_fallback_ratio_max
    )

    summary: Dict[str, Any] = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "completion_passed": sum(1 for item in completion if item.get("ok")),
        "completion_total": len(completion),
        "stream_passed": sum(1 for item in stream if item.get("ok")),
        "stream_total": len(stream),
        "quality_natural_chat_passed": sum(
            1 for item in results if item.get("quality", {}).get("natural_chat")
        ),
        "quality_no_table_markup_passed": sum(
            1 for item in results if item.get("quality", {}).get("no_table_markup")
        ),
        "quality_no_briefing_headers_passed": sum(
            1
            for item in results
            if item.get("quality", {}).get("no_briefing_headers")
        ),
        "quality_no_source_line_passed": sum(
            1 for item in results if item.get("quality", {}).get("no_source_line")
        ),
        "quality_no_raw_chunk_marker_passed": sum(
            1
            for item in results
            if item.get("quality", {}).get("no_raw_chunk_marker")
        ),
        "quality_no_low_data_fallback_passed": sum(
            1
            for item in results
            if item.get("quality", {}).get("no_low_data_fallback")
        ),
        "completion_metrics": _endpoint_metrics(completion),
        "stream_metrics": _endpoint_metrics(stream),
        "planner_mode_metrics": {
            "overall": _planner_mode_metrics(results),
            "completion": _planner_mode_metrics(completion),
            "stream": _planner_mode_metrics(stream),
        },
        "completion_fallback_metrics": completion_fallback_metrics,
        "stream_fallback_metrics": stream_fallback_metrics,
        "stream_fallback_ratio_max": stream_fallback_ratio_max,
        "stream_fallback_ratio_ok": stream_fallback_ratio_ok,
        "overall_error_rate": _ratio(failed, total),
        "overall_timeout_rate": _ratio(
            sum(1 for item in results if _is_timeout_result(item)),
            total,
        ),
    }
    return summary


def _build_report_payload(
    *,
    base_url: str,
    total_questions: int,
    strict: bool,
    chat_batch_size: int,
    run_id: str,
    pid: int,
    lock_file: str,
    stream_fallback_ratio_max: float,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary = _summarize_results(
        results, stream_fallback_ratio_max=stream_fallback_ratio_max
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "base_url": base_url,
            "total_questions": total_questions,
            "strict": strict,
            "chat_batch_size": chat_batch_size,
            "run_id": run_id,
            "pid": pid,
            "lock_file": lock_file,
            "stream_fallback_ratio_max": stream_fallback_ratio_max,
        },
        "summary": summary,
        "results": results,
    }


def _write_output_report(
    output_path: Path,
    *,
    base_url: str,
    total_questions: int,
    strict: bool,
    chat_batch_size: int,
    run_id: str,
    pid: int,
    lock_file: str,
    stream_fallback_ratio_max: float,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    report = _build_report_payload(
        base_url=base_url,
        total_questions=total_questions,
        strict=strict,
        chat_batch_size=chat_batch_size,
        run_id=run_id,
        pid=pid,
        lock_file=lock_file,
        stream_fallback_ratio_max=stream_fallback_ratio_max,
        results=results,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _resume_results_from_output(output_path: Path) -> Tuple[List[Dict[str, Any]], int]:
    if not output_path.exists():
        return [], 0

    try:
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return [], 0

    raw_results = loaded.get("results", [])
    if not isinstance(raw_results, list) or not raw_results:
        return [], 0

    # 질문당 completion/stream 2건 단위까지만 복구합니다.
    pair_count = len(raw_results) // 2
    restored = raw_results[: pair_count * 2]
    return restored, pair_count


def main() -> int:
    args = parse_args()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    lock_path = Path(args.lock_file)

    try:
        _acquire_execution_lock(
            lock_path, run_id=run_id, fail_if_locked=args.fail_if_locked
        )
    except RuntimeError as exc:
        print(str(exc))
        return 2

    try:
        base_url = args.base_url.rstrip("/")
        questions = _load_questions(args.chat_question_list, args.chat_batch_size)
        headers = _build_internal_headers(args.internal_api_key)
        timeout = httpx.Timeout(args.timeout, connect=min(10.0, args.timeout))
        history_payload = (
            [{"role": "assistant", "content": "smoke_live_mode"}]
            if args.disable_cache
            else None
        )

        results: List[Dict[str, Any]] = []
        completed_questions = 0
        output_path = Path(args.output) if args.output else None

        if args.resume and output_path is not None:
            results, completed_questions = _resume_results_from_output(output_path)
            if completed_questions > 0:
                print(
                    f"resume from output: {completed_questions}/{len(questions)} questions"
                )
        if args.disable_cache:
            print("cache bypass enabled: history payload attached")

        with httpx.Client(timeout=timeout) as client:
            for idx, question in enumerate(
                questions[completed_questions:], start=completed_questions + 1
            ):
                completion_result = _check_completion(
                    client,
                    base_url,
                    question,
                    history_payload=history_payload,
                    headers=headers,
                    rate_limit_retries=args.rate_limit_max_retries,
                    rate_limit_base_delay=args.rate_limit_base_delay,
                )
                stream_result = _check_stream(
                    client,
                    base_url,
                    question,
                    history_payload=history_payload,
                    headers={"X-Internal-Api-Key": headers.get("X-Internal-Api-Key", "")},
                    rate_limit_retries=args.rate_limit_max_retries,
                    rate_limit_base_delay=args.rate_limit_base_delay,
                    stream_style=args.stream_style,
                )

                results.append(completion_result)
                results.append(stream_result)
                _print_progress(idx, len(questions), completion_result, stream_result)

                if output_path is not None:
                    _write_output_report(
                        output_path,
                        base_url=base_url,
                        total_questions=len(questions),
                        strict=args.strict,
                        chat_batch_size=args.chat_batch_size,
                        run_id=run_id,
                        pid=os.getpid(),
                        lock_file=str(lock_path),
                        stream_fallback_ratio_max=args.stream_fallback_ratio_max,
                        results=results,
                    )

        report = _build_report_payload(
            base_url=base_url,
            total_questions=len(questions),
            strict=args.strict,
            chat_batch_size=args.chat_batch_size,
            run_id=run_id,
            pid=os.getpid(),
            lock_file=str(lock_path),
            stream_fallback_ratio_max=args.stream_fallback_ratio_max,
            results=results,
        )
        summary = report["summary"]
        print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))

        if output_path is not None:
            _write_output_report(
                output_path,
                base_url=base_url,
                total_questions=len(questions),
                strict=args.strict,
                chat_batch_size=args.chat_batch_size,
                run_id=run_id,
                pid=os.getpid(),
                lock_file=str(lock_path),
                stream_fallback_ratio_max=args.stream_fallback_ratio_max,
                results=results,
            )
            print(f"report saved: {output_path}")

        if args.summary_output:
            summary_output = Path(args.summary_output)
            summary_output.parent.mkdir(parents=True, exist_ok=True)
            summary_payload = {
                "generated_at_utc": report["generated_at_utc"],
                "input": report["input"],
                "summary": summary,
            }
            summary_output.write_text(
                json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"summary saved: {summary_output}")

        if args.strict and (
            summary["failed"] > 0 or not summary.get("stream_fallback_ratio_ok", True)
        ):
            return 1
        return 0
    finally:
        _release_execution_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
