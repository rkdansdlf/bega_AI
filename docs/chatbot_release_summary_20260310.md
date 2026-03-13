# 야구 챗봇 배포 요약 (2026-03-10)

## 목적
- `KBO + 야구 일반 상식` 범위에서 정확도를 우선하는 챗봇 응답 정책을 운영 가능한 수준으로 끌어올린다.
- 설명형, 규정형, 최신성 질문에서 추정성 답변과 브리핑형 출력 누수를 줄인다.

## 이번 변경의 핵심
- 팀 지표 질문을 선수 리더보드 경로와 분리했다.
- 리더보드 정렬 로직을 고쳐 top 순위 오답을 줄였다.
- 설명형/규정형 자주 실패하던 질문에 직접 답변 경로를 추가했다.
- 최신성 질문은 약한 웹 결과를 억지로 요약하지 않고, `기준 시점 + 직접 확인 불가`로 안전하게 응답하도록 보강했다.
- 분석 단계 실패 시 일부 질문은 builtin/direct answer 또는 DB fast-path로 복구하도록 했다.

## 최종 검증 결과
- 설명형 canary: `12 / 12`
- 최신성 canary: `6 / 6`
- factual canary v2: `72 / 72`
- baseball 180: `360 / 360`

## 대표 보고서
- `/Users/mac/project/KBO_platform/reports/explainer_accuracy_canary_20260309_rerun_summary.json`
- `/Users/mac/project/KBO_platform/reports/latest_accuracy_canary_20260309_final_summary.json`
- `/Users/mac/project/KBO_platform/reports/factual_canary_20260310_v2_final2_summary.json`
- `/Users/mac/project/KBO_platform/reports/baseball_180_accuracy_20260309_final2_summary.json`

## 운영 시 특히 볼 항목
- `팀 타율`, `팀 OPS`, `팀 평균자책점`, `팀 홈런`, `팀 타점` 질문이 선수 순위 답변으로 새지 않는지
- `포수 프레이밍`, `비디오 판독`, `승리투수 요건`, `WHIP`, `QS` 같은 설명형/규정형 질문이 바로 정의로 시작하는지
- `최근 맞대결`, `최근 경기 활약`, `5위 싸움` 질문이 기준 시점을 포함하고 불확실할 때 추정하지 않는지
- stream 응답에 `##`, `분석 결과`, `출처:` 라인이 남지 않는지

## 배포 판단
- 현재 스모크 기준으로는 배포 가능한 상태다.
- 다만 latest search 품질은 배포 환경 네트워크와 외부 검색 품질에 영향을 받으므로, 배포 후 실제 질문 5~10건을 운영 환경에서 다시 확인하는 것이 안전하다.
