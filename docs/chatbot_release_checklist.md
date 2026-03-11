# 야구 챗봇 배포 전 체크리스트

## 1. 데이터/검색 계층
- `rag_chunks`에 최신 문서 청크가 적재되어 있는지 확인한다.
- `markdown_docs`, `kbo_definitions`, `kbo_regulations` 검색이 모두 살아 있는지 확인한다.
- latest search가 배포 환경에서 실제로 외부 검색 결과를 받는지 확인한다.

## 2. 서비스 기동
- `kbo-ai` 컨테이너 또는 프로세스가 정상 기동하는지 확인한다.
- `/health`가 `{"status":"ok"}`를 반환하는지 확인한다.
- 내부 토큰(`AI_INTERNAL_TOKEN`)이 smoke/ops 경로에서 일관되게 사용되는지 확인한다.

## 2-1. 현재 검증 완료 기준선
- 설명형 canary: `12 / 12`
  - `/Users/mac/project/KBO_platform/reports/explainer_accuracy_canary_20260309_rerun_summary.json`
- 최신성 canary: `6 / 6`
  - `/Users/mac/project/KBO_platform/reports/latest_accuracy_canary_20260309_final_summary.json`
- factual canary v2 (`36`문항, `72`체크): `72 / 72`
  - `/Users/mac/project/KBO_platform/reports/factual_canary_20260310_v2_final2_summary.json`
- full 180: `360 / 360`
  - `/Users/mac/project/KBO_platform/reports/baseball_180_accuracy_20260309_final2_summary.json`

## 3. 최소 검증 순서
1. 설명형 canary
```bash
./.venv/bin/python bega_AI/scripts/smoke_chatbot_quality.py \
  --chat-batch-size 6 \
  --chat-question-list /tmp/explainer_accuracy_canary.txt \
  --disable-cache \
  --stream-style compact \
  --no-strict
```
2. 최신성 canary
```bash
./.venv/bin/python bega_AI/scripts/smoke_chatbot_quality.py \
  --chat-batch-size 3 \
  --chat-question-list /tmp/latest_accuracy_canary.txt \
  --disable-cache \
  --stream-style compact \
  --no-strict
```
3. factual canary
```bash
./.venv/bin/python bega_AI/scripts/smoke_chatbot_factual_canary.py \
  --disable-cache \
  --stream-style compact \
  --no-strict
```
4. full 180
```bash
./.venv/bin/python bega_AI/scripts/smoke_chatbot_quality.py \
  --chat-batch-size 180 \
  --chat-question-list bega_AI/scripts/smoke_chatbot_quality_baseball_180.txt \
  --disable-cache \
  --stream-style compact \
  --no-strict
```

## 4. 통과 기준
- 설명형 canary: `12 / 12`
- 최신성 canary: `6 / 6`
- factual canary v2: `72 / 72`
- full 180: `360 / 360`
- `quality_no_source_line`, `quality_no_briefing_headers`, `quality_natural_chat` 전부 통과

## 5. 배포 후 확인
- `최근 맞대결`, `최근 경기 활약`, `5위 싸움`, `BABIP`, `WPA`, `보크`, `낫아웃`을 실제로 한 번씩 질의한다.
- `포수 프레이밍`, `비디오 판독`, `승리투수 요건`, `WHIP`, `QS`도 한 번씩 질의한다.
- 팀 지표 질문에서 선수 리더보드 답변이 나오지 않는지 확인한다.
- stream 응답에서 `##`, `분석 결과`, `출처:` 같은 브리핑형 라인이 남지 않는지 확인한다.
- latest 질문이 내부 DB 부족 시 `기준 시점`을 포함해 답하는지 확인한다.

## 6. 롤백 기준
- team metric 질문이 다시 선수 상위권 리더보드로 답하면 즉시 롤백 검토
- latest 질문에서 브리핑체/분석체가 재발하면 롤백보다 우선 후처리 규칙 재적용 검토
- factual canary v2에서 기존 통과 케이스가 1개라도 깨지면 릴리스 보류
