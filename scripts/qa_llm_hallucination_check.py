#!/usr/bin/env python3
"""LLM 환각 방지 수동 QA 스크립트.

AI 서비스가 실행 중인 상태에서 사용합니다:
    python scripts/qa_llm_hallucination_check.py --url http://localhost:8001

각 시나리오의 실제 LLM 출력을 출력하고,
간단한 패턴 기반 경고를 표시합니다.
최종 판단은 사람이 직접 내용을 읽고 결정해야 합니다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

# 환각 위험 패턴: 구체적 수치가 단정적으로 등장하는 패턴
_STAT_PATTERN = re.compile(
    r"(?:타율|방어율|홈런|타점|ERA|OPS|WAR|wRC\+|WHIP)\s*[:]?\s*"
    r"(?:0\.\d{3}|\d{1,2}\.\d{2}|\d{1,3}개|\d{1,3}점|\d{1,3}승)"
)

# DB-down 면책 문구 패턴
_DISCLAIMER_PATTERN = re.compile(r"⚠️|접속할 수 없|일시적|참고 답변|정확한 수치")

# Zero-hit 가이드 패턴
_GUIDE_PATTERN = re.compile(r"가능한 원인|대안|연도.*다시|선수명.*확인|역대|찾지 못")


@dataclass
class Scenario:
    name: str
    question: str
    expect_disclaimer: bool  # DB-down 면책 문구 기대
    expect_guide: bool  # Zero-hit 가이드 기대
    expect_no_stats: bool  # 수치 단정 없어야 함
    description: str


SCENARIOS: list[Scenario] = [
    Scenario(
        name="S2 존재하지 않는 선수",
        question="김철수아무개99 2024년 타율은?",
        expect_disclaimer=False,
        expect_guide=True,
        expect_no_stats=True,
        description="없는 선수 → 가이드 텍스트, 수치 없음",
    ),
    Scenario(
        name="S3 미래 연도",
        question="2099년 KBO 홈런왕은?",
        expect_disclaimer=False,
        expect_guide=True,
        expect_no_stats=True,
        description="미래 연도 → '아직 수집' 표현, 수치 없음",
    ),
    Scenario(
        name="S4 정상 질문 (회귀 확인)",
        question="2024년 이정후는 어떤 선수야?",
        expect_disclaimer=False,
        expect_guide=False,
        expect_no_stats=False,
        description="정상 질문 → 실제 정보, ⚠️ 없음",
    ),
    Scenario(
        name="S5 팀+미래 연도 zero-hit",
        question="2099년 LG 트윈스 선발 1위 투수는?",
        expect_disclaimer=False,
        expect_guide=True,
        expect_no_stats=True,
        description="미래 팀 질문 → 연도 대안 제안, 수치 없음",
    ),
]


# ---------------------------------------------------------------------------
# API 호출
# ---------------------------------------------------------------------------


def call_completion(base_url: str, question: str, token: str) -> Optional[str]:
    url = f"{base_url}/ai/chat/completion"
    payload = json.dumps({"question": question, "history": None}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Internal-Api-Key": token,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            return body.get("answer", "")
    except urllib.error.URLError as e:
        print(f"  [ERROR] API 호출 실패: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] 응답 파싱 실패: {e}")
        return None


# ---------------------------------------------------------------------------
# 패턴 체크
# ---------------------------------------------------------------------------


def check_answer(answer: str, scenario: Scenario) -> list[str]:
    warnings = []
    passes = []

    if scenario.expect_disclaimer:
        if _DISCLAIMER_PATTERN.search(answer):
            passes.append("✅ 면책 문구 확인됨")
        else:
            warnings.append("⚠️ WARNING: 면책 문구 없음 — 검토 필요")

    if scenario.expect_guide:
        if _GUIDE_PATTERN.search(answer):
            passes.append("✅ 가이드 텍스트 확인됨")
        else:
            warnings.append("⚠️ WARNING: 가이드 텍스트 없음 — 검토 필요")

    if scenario.expect_no_stats:
        matches = _STAT_PATTERN.findall(answer)
        if matches:
            warnings.append(f"⚠️ WARNING: 수치 단정 의심 — 검토 필요: {matches[:3]}")
        else:
            passes.append("✅ 수치 단정 없음")

    return passes + warnings


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="LLM 환각 방지 수동 QA")
    parser.add_argument(
        "--url", default="http://localhost:8001", help="AI 서비스 base URL"
    )
    parser.add_argument(
        "--token", default="dev-internal-token", help="X-Internal-Api-Key 값"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("KBO Platform LLM 환각 방지 수동 QA")
    print(f"대상 서비스: {args.url}")
    print("=" * 70)
    print()
    print(
        "참고: S1 (DB-down) 시나리오는 실제 DB 연결 차단이 필요하므로 여기서는 제외합니다."
    )
    print(
        "      직접 DB_URL을 잘못된 값으로 설정 후 서비스를 재시작하여 수동으로 확인하세요."
    )
    print()

    has_warning = False

    for scenario in SCENARIOS:
        print(f"{'─' * 60}")
        print(f"[{scenario.name}]")
        print(f"  질문: {scenario.question}")
        print(f"  기대: {scenario.description}")
        print()

        answer = call_completion(args.url, scenario.question, args.token)

        if answer is None:
            print("  [SKIP] API 응답 없음 — 서비스 실행 여부 확인 필요")
            print()
            continue

        print("  --- 실제 LLM 출력 (직접 읽고 판단하세요) ---")
        print(f"  {answer[:500]}{'...' if len(answer) > 500 else ''}")
        print()

        results = check_answer(answer, scenario)
        for r in results:
            print(f"  {r}")
            if "WARNING" in r:
                has_warning = True
        print()

    print("=" * 70)
    if has_warning:
        print("⚠️  경고 항목이 있습니다. 위 출력을 직접 읽고 최종 판단하세요.")
        sys.exit(1)
    else:
        print("✅ 자동 패턴 검사 통과. 실제 출력 내용을 최종 확인하세요.")
        sys.exit(0)


if __name__ == "__main__":
    main()
