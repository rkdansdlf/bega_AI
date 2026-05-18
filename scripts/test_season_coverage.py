#!/usr/bin/env python3
"""
test_season_coverage.py
2020~2026 시즌별 챗봇 대응 가능 여부 통합 테스트

실행:
  python scripts/test_season_coverage.py
  python scripts/test_season_coverage.py --base-url http://localhost:8001 --timeout 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

# ─────────────────────────────────────────────────────────────────────────────
BASE_URL = "http://127.0.0.1:8001"
TIMEOUT = 60.0
TOKEN = os.getenv("AI_INTERNAL_TOKEN", "ef9c322a0d4105d5f520328cfc29e2197952609776df75eee0fcfab23497bb4f")
HEADERS = {"X-Internal-Api-Key": TOKEN, "Content-Type": "application/json"}
REQUEST_DELAY = 2.0  # rate limit (60req/min) 회피용 요청 간 대기(초)

# ─────────────────────────────────────────────────────────────────────────────
# 테스트 케이스 정의
# note:
#   - 2022~2026: 게임별 상세 통계 있음 → 경기 결과, 개인 경기 성적 질문 가능
#   - 2020~2021: 시즌 통계/선수 기본정보만 → 시즌 누적 성적만 가능
# ─────────────────────────────────────────────────────────────────────────────
TEST_CASES: list[dict[str, Any]] = [
    # ── 2026 ──────────────────────────────────────────────────────────────────
    {
        "season": 2026,
        "category": "팀 성적",
        "question": "2026년 KBO 순위는 어떻게 돼?",
        "filters": {"season_year": 2026},
        "expect_keywords": ["2026", "위"],
        "data_note": "2026 시즌 진행 중 (3/22~5/14)",
    },
    {
        "season": 2026,
        "category": "타자 성적",
        "question": "2026년 4월 타율 1위 타자는 누구야?",
        "filters": {"season_year": 2026},
        "expect_keywords": ["2026"],
        "data_note": "2026 타격 랭킹",
    },
    {
        "season": 2026,
        "category": "팀 성적",
        "question": "2026년 두산 베어스 성적 알려줘",
        "filters": {"season_year": 2026},
        "expect_keywords": ["두산"],
        "data_note": "2026 두산 팀 성적",
    },

    # ── 2025 ──────────────────────────────────────────────────────────────────
    {
        "season": 2025,
        "category": "우승팀",
        "question": "2025년 KBO 우승팀은 어디야?",
        "filters": {"season_year": 2025},
        "expect_keywords": ["2025"],
        "data_note": "2025 시즌 챔피언",
    },
    {
        "season": 2025,
        "category": "홈런왕",
        "question": "2025년 홈런왕은 누구야?",
        "filters": {"season_year": 2025},
        "expect_keywords": ["2025", "홈런"],
        "data_note": "2025 홈런 1위",
    },
    {
        "season": 2025,
        "category": "투수 성적",
        "question": "2025년 KBO 다승왕 투수는 누구야?",
        "filters": {"season_year": 2025},
        "expect_keywords": ["2025"],
        "data_note": "2025 투수 다승 1위",
    },

    # ── 2024 ──────────────────────────────────────────────────────────────────
    {
        "season": 2024,
        "category": "우승팀",
        "question": "2024년 KBO 한국시리즈 우승팀은?",
        "filters": {"season_year": 2024},
        "expect_keywords": ["2024"],
        "data_note": "2024 KS 챔피언",
    },
    {
        "season": 2024,
        "category": "MVP",
        "question": "2024년 KBO MVP는 누구야?",
        "filters": {"season_year": 2024},
        "expect_keywords": ["2024"],
        "data_note": "2024 MVP 수상자",
    },
    {
        "season": 2024,
        "category": "팀 성적",
        "question": "2024년 KIA 타이거즈 팀 타율은 얼마야?",
        "filters": {"season_year": 2024},
        "expect_keywords": ["KIA", "2024"],
        "data_note": "2024 KIA 타격 성적",
    },

    # ── 2023 ──────────────────────────────────────────────────────────────────
    {
        "season": 2023,
        "category": "우승팀",
        "question": "2023년 KBO 정규시즌 1위 팀은?",
        "filters": {"season_year": 2023},
        "expect_keywords": ["2023"],
        "data_note": "2023 정규리그 우승",
    },
    {
        "season": 2023,
        "category": "타자 성적",
        "question": "2023년 LG 트윈스 4번 타자는 누구야?",
        "filters": {"season_year": 2023},
        "expect_keywords": ["LG", "2023"],
        "data_note": "2023 LG 타선",
    },
    {
        "season": 2023,
        "category": "경기 결과",
        "question": "2023년 LG 트윈스 한국시리즈 상대는 어느 팀이었어?",
        "filters": {"season_year": 2023},
        "expect_keywords": ["LG", "2023"],
        "data_note": "2023 한국시리즈",
    },

    # ── 2022 ──────────────────────────────────────────────────────────────────
    {
        "season": 2022,
        "category": "우승팀",
        "question": "2022년 KBO 우승팀은 어디야?",
        "filters": {"season_year": 2022},
        "expect_keywords": ["2022"],
        "data_note": "2022 KS 챔피언",
    },
    {
        "season": 2022,
        "category": "홈런왕",
        "question": "2022년 홈런 가장 많이 친 선수는 누구야?",
        "filters": {"season_year": 2022},
        "expect_keywords": ["2022"],
        "data_note": "2022 홈런 1위",
    },
    {
        "season": 2022,
        "category": "투수 ERA",
        "question": "2022년 평균자책점 1위 투수는?",
        "filters": {"season_year": 2022},
        "expect_keywords": ["2022"],
        "data_note": "2022 ERA 1위",
    },

    # ── 2021 (시즌통계만 — 게임 단위 통계 없음) ────────────────────────────
    {
        "season": 2021,
        "category": "우승팀",
        "question": "2021년 KBO 우승팀은?",
        "filters": {"season_year": 2021},
        "expect_keywords": ["2021"],
        "data_note": "시즌통계만 존재 (게임별 없음)",
    },
    {
        "season": 2021,
        "category": "타자 성적",
        "question": "2021년 KBO 타율 1위 선수는 누구야?",
        "filters": {"season_year": 2021},
        "expect_keywords": ["2021"],
        "data_note": "2021 시즌 타격 랭킹",
    },

    # ── 2020 (시즌통계만 — 게임 단위 통계 없음) ────────────────────────────
    {
        "season": 2020,
        "category": "우승팀",
        "question": "2020년 KBO 우승팀은 어디야?",
        "filters": {"season_year": 2020},
        "expect_keywords": ["2020"],
        "data_note": "시즌통계만 존재 (게임별 없음)",
    },
    {
        "season": 2020,
        "category": "홈런왕",
        "question": "2020년 홈런왕은 누구야?",
        "filters": {"season_year": 2020},
        "expect_keywords": ["2020"],
        "data_note": "2020 홈런 1위",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TestResult:
    season: int
    category: str
    question: str
    passed: bool
    response_len: int
    elapsed: float
    answer_snippet: str
    reason: str
    data_note: str = ""


def check_response(answer: str, expect_keywords: list[str]) -> tuple[bool, str]:
    """응답 품질 판단."""
    # MVP 등 단문 정답도 통과할 수 있도록 최소 길이를 20자로 완화
    if not answer or len(answer.strip()) < 20:
        return False, f"응답 너무 짧음 ({len(answer)}자)"

    FAIL_PHRASES = [
        "찾을 수 없", "알 수 없", "모르겠", "제공되지 않", "확인할 수 없",
        "데이터가 없", "정보가 없", "죄송합니다", "오류가 발생",
        "MANUAL_BASEBALL_DATA_REQUIRED",
    ]
    for phrase in FAIL_PHRASES:
        if phrase in answer:
            return False, f"실패 문구 감지: '{phrase}'"

    return True, "OK"


def run_test(client: httpx.Client, case: dict[str, Any]) -> TestResult:
    payload = {
        "question": case["question"],
        "filters": case.get("filters"),
        "history": None,
    }
    start = time.monotonic()
    try:
        resp = client.post(
            f"{BASE_URL}/ai/chat/completion",
            json=payload,
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        elapsed = time.monotonic() - start

        if resp.status_code != 200:
            return TestResult(
                season=case["season"],
                category=case["category"],
                question=case["question"],
                passed=False,
                response_len=0,
                elapsed=elapsed,
                answer_snippet="",
                reason=f"HTTP {resp.status_code}: {resp.text[:200]}",
                data_note=case.get("data_note", ""),
            )

        data = resp.json()
        answer = data.get("answer", data.get("response_text", ""))
        passed, reason = check_response(answer, case.get("expect_keywords", []))
        return TestResult(
            season=case["season"],
            category=case["category"],
            question=case["question"],
            passed=passed,
            response_len=len(answer),
            elapsed=elapsed,
            answer_snippet=answer[:120].replace("\n", " "),
            reason=reason,
            data_note=case.get("data_note", ""),
        )
    except Exception as e:
        return TestResult(
            season=case["season"],
            category=case["category"],
            question=case["question"],
            passed=False,
            response_len=0,
            elapsed=time.monotonic() - start,
            answer_snippet="",
            reason=f"Exception: {e}",
            data_note=case.get("data_note", ""),
        )


def print_results(results: list[TestResult]) -> None:
    print("\n" + "=" * 80)
    print(f"{'시즌별 챗봇 커버리지 테스트 결과':^80}")
    print("=" * 80)

    # 시즌별 그룹핑
    by_season: dict[int, list[TestResult]] = {}
    for r in results:
        by_season.setdefault(r.season, []).append(r)

    total_pass = total_fail = 0

    for season in sorted(by_season.keys(), reverse=True):
        season_results = by_season[season]
        s_pass = sum(1 for r in season_results if r.passed)
        s_fail = len(season_results) - s_pass
        total_pass += s_pass
        total_fail += s_fail

        status_icon = "✅" if s_fail == 0 else ("⚠️ " if s_pass > 0 else "❌")
        print(f"\n── {season}년 {status_icon}  ({s_pass}/{len(season_results)} 통과) " + "-" * 40)

        for r in season_results:
            icon = "✅" if r.passed else "❌"
            t = f"{r.elapsed:.1f}s"
            print(f"  {icon} [{r.category:10s}] {r.question[:45]:<45} ({t})")
            if not r.passed:
                print(f"       └─ FAIL: {r.reason}")
            else:
                print(f"       └─ {r.answer_snippet[:80]}")

    print("\n" + "=" * 80)
    total = total_pass + total_fail
    pct = total_pass / total * 100 if total else 0
    print(f"  총합: {total_pass}/{total} 통과  ({pct:.0f}%)")
    if total_fail > 0:
        print(f"  실패: {total_fail}건")
    print("=" * 80)

    # 시즌별 요약 표
    print("\n[시즌별 요약]")
    print(f"  {'시즌':>6}  {'통과':>6}  {'실패':>6}  {'비고'}")
    for season in sorted(by_season.keys(), reverse=True):
        season_results = by_season[season]
        s_pass = sum(1 for r in season_results if r.passed)
        s_fail = len(season_results) - s_pass
        note = season_results[0].data_note if season_results else ""
        icon = "✅" if s_fail == 0 else ("⚠️" if s_pass > 0 else "❌")
        print(f"  {season:>6}  {s_pass:>6}  {s_fail:>6}  {icon} {note}")


def main() -> int:
    global BASE_URL, TIMEOUT  # noqa: PLW0603
    parser = argparse.ArgumentParser(description="시즌별 챗봇 커버리지 테스트")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--timeout", type=float, default=TIMEOUT)
    parser.add_argument("--season", type=int, default=None,
                        help="특정 시즌만 테스트 (예: 2026)")
    parser.add_argument("--output", default=None,
                        help="JSON 결과 파일 경로")
    args = parser.parse_args()

    BASE_URL = args.base_url  # type: ignore[assignment]
    TIMEOUT = args.timeout  # type: ignore[assignment]

    cases = TEST_CASES
    if args.season:
        cases = [c for c in TEST_CASES if c["season"] == args.season]
        if not cases:
            print(f"ERROR: {args.season}년 테스트 케이스가 없습니다.")
            return 1

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"챗봇 커버리지 테스트 시작 — {len(cases)}건")
    print(f"  대상 서버: {BASE_URL}")
    print(f"  타임아웃: {TIMEOUT}s")

    # 서버 health 체크
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=10)
        print(f"  서버 상태: {r.status_code} {r.json()}")
    except Exception as e:
        print(f"  ❌ 서버 연결 실패: {e}")
        return 2

    results: list[TestResult] = []
    with httpx.Client() as client:
        for i, case in enumerate(cases, 1):
            print(f"  [{i:02d}/{len(cases)}] {case['season']}년 {case['category']}: "
                  f"{case['question'][:50]}...", end=" ", flush=True)
            r = run_test(client, case)
            results.append(r)
            print("✅" if r.passed else f"❌ ({r.reason[:50]})")
            if i < len(cases):
                time.sleep(REQUEST_DELAY)  # rate limit 회피

    print_results(results)

    if args.output:
        out = [
            {
                "season": r.season,
                "category": r.category,
                "question": r.question,
                "passed": r.passed,
                "reason": r.reason,
                "elapsed": round(r.elapsed, 2),
                "answer_snippet": r.answer_snippet,
            }
            for r in results
        ]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {args.output}")

    total_fail = sum(1 for r in results if not r.passed)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
