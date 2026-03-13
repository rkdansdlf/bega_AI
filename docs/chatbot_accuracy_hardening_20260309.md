# 야구 챗봇 정확도 하드닝 정리 (2026-03-09, 최종 갱신 2026-03-10)

## 목표
- `KBO + 야구 일반 상식` 질문에서 답변 가능 범위를 넓히되, 출력 형식보다 정확도를 우선한다.
- 최신성 질문은 `내부 DB -> 문서 KB -> latest search` 순서로 처리한다.
- 설명형 질문은 문서 검색 기반으로 답하되, 문서 덤프와 브리핑형 출력을 막는다.

## 핵심 변경
- 라우팅 보정
  - 팀 지표 질문(`팀 타율`, `팀 OPS`, `팀 평균자책점`, `팀 홈런`, `팀 타점`)을 선수 리더보드와 분리했다.
  - 설명형/규정형/최신성 질문을 각각 `docs`, `latest`, `fast-path`로 더 일관되게 라우팅한다.
- 데이터 조회 보정
  - 리더보드 쿼리를 `중복 제거 -> 전역 정렬 -> limit` 순서로 바꿔 순위 오정렬을 막았다.
  - 팀 지표 질문은 `get_team_advanced_metrics`를 우선 활용하도록 강화했다.
- 답변 합성 보정
  - `질문하신`, `분석 결과`, `## 요약` 같은 브리핑형 출력과 문서 개요 덤프를 후처리에서 제거했다.
  - `BABIP`, `ABS`, `WPA`, `WHIP`, `QS`, `체인지업`, `히트앤런`, `필승조`, `포수 프레이밍`, `수비 시프트`, `병살타`, `보크`, `낫아웃`, `지명타자`, `타이브레이크`, `비디오 판독`, `승리투수 요건`, `사구 vs 고의4구` 등 실패 빈도가 높던 설명형/규정형 질문은 직접 설명 경로를 보강했다.
  - 최신성 선수/맞대결 질문은 약한 웹 결과를 억지로 요약하지 않고 `기준 시점 + 직접 확인 불가`로 안전하게 응답하는 경로를 추가했다.
- 검증 체계 추가
  - style 중심 `quality` 스모크 외에 `required_all`, `required_any`, `forbidden` 규칙을 가진 factual canary를 추가했다.
  - 분석 단계 실패 시에도 팀 지표 질문은 `DB fast-path`로 한 번 더 복구하도록 보강했다.

## 주요 변경 파일
- `/Users/mac/project/KBO_platform/bega_AI/app/ml/intent_router.py`
- `/Users/mac/project/KBO_platform/bega_AI/app/agents/chat_intent_router.py`
- `/Users/mac/project/KBO_platform/bega_AI/app/agents/baseball_agent.py`
- `/Users/mac/project/KBO_platform/bega_AI/app/agents/chat_renderers.py`
- `/Users/mac/project/KBO_platform/bega_AI/app/tools/database_query.py`
- `/Users/mac/project/KBO_platform/bega_AI/app/core/prompts.py`
- `/Users/mac/project/KBO_platform/bega_AI/scripts/smoke_chatbot_factual_canary.py`
- `/Users/mac/project/KBO_platform/bega_AI/scripts/smoke_chatbot_factual_canary_cases.json`

## 검증 결과
- 설명형 canary: `12 / 12`
  - `/Users/mac/project/KBO_platform/reports/explainer_accuracy_canary_20260309_rerun_summary.json`
- 최신성 canary: `6 / 6`
  - `/Users/mac/project/KBO_platform/reports/latest_accuracy_canary_20260309_final_summary.json`
- factual canary v1: `24 / 24`
  - `/Users/mac/project/KBO_platform/reports/factual_canary_20260309_rerun3_summary.json`
- factual canary v2 (`36`문항, `72`체크): `72 / 72`
  - `/Users/mac/project/KBO_platform/reports/factual_canary_20260310_v2_final2_summary.json`
- baseball 180: `360 / 360`
  - `/Users/mac/project/KBO_platform/reports/baseball_180_accuracy_20260309_final2_summary.json`

## 해석
- 현재 품질 스모크 기준으로는 회귀가 닫혔다.
- 특히 `team metric misroute`, `leaderboard misorder`, `explainer intro leakage`, `latest briefing leakage`는 재현되지 않았다.
- 추가 확장한 factual canary에서도 `팀 지표`, `설명형 정의`, `규정형 설명`, `최근 맞대결`, `최근 경기 활약`까지 모두 통과했다.
- 다만 최신성 질문은 네트워크/검색 환경 영향을 받기 때문에, 배포 환경에서도 동일한 latest search 품질을 유지하는지 확인이 필요하다.

## 다음 유지보수 원칙
- 설명형 질문은 LLM 일반 생성보다 `docs + direct explainer override`를 우선한다.
- 팀 질문에 `팀/구단 + 타율/OPS/ERA/홈런/타점`이 들어가면 리더보드보다 팀 지표 경로를 먼저 탄다.
- 새로운 실패가 생기면 먼저 factual canary에 재현 케이스를 추가한 뒤 수정한다.
- latest 질문 회귀는 `quality`보다 `forbidden 브리핑 문구` 기준으로 먼저 본다.

## 참고
- factual canary는 현재 `v1(12문항)`과 `v2(36문항)` 두 기준을 모두 통과한 상태다.
- 최신성 질문은 실제 경기 결과를 찾지 못한 경우에도 추정 대신 `기준 시점 + 직접 확인 불가`로 답하게 설계했다.
