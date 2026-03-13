# 챗봇 Query Helper 정리

## 목적
- 이번 정리 배치에서 도입한 query helper의 책임을 고정한다.
- 새 query tool 작성 시 기존 helper를 우선 사용해 중복 구현을 막는다.
- 이번 문서 범위에서는 추가 base class나 mixin을 만들지 않는다.

## Helper 책임

### `team_display`
- 팀 코드, 약칭, 표시명을 사용자 노출용 문자열로 정리한다.
- agent, query tool, renderer에서 같은 팀명 치환 규칙을 공유할 때 사용한다.
- 팀명을 직접 하드코딩해 다시 포맷하지 말고 이 helper를 우선 호출한다.

### `team_mapping_loader`
- 팀명 매핑 조회, 재시도, snapshot fallback을 한 경로로 묶는다.
- `DatabaseQueryTool`, `GameQueryTool`처럼 DB 기반 팀 매핑이 필요한 tool에서 사용한다.
- 매핑 로드 실패 시 각 tool 안에서 별도 SQL 체인을 다시 만들지 않는다.

### `pooled_connection`
- pooled connection 획득과 `connection is closed` 계열 재시도를 공용화한다.
- query tool이 connection pool을 직접 다루면 이 helper를 우선 사용한다.
- closed connection 복구는 public API마다 따로 구현하지 않고 retry wrapper에 맡긴다.

### `query_result`
- query tool의 result dict 기본 shape와 success/error 반영을 공용화한다.
- 기존 public response key shape는 유지한 채, 내부 조립 로직만 단순화할 때 사용한다.
- 새 helper를 쓰더라도 기존 API contract를 바꾸면 안 된다.

### `query_logging`
- `event=` / `action=` 기반 로그 포맷과 action 상수를 제공한다.
- `query_start`, `query_retry`, `query_success`, `query_empty`, `query_error`, `dependency_missing` 이벤트를 같은 형식으로 남길 때 사용한다.
- query tool마다 다른 free-form 로그 문구를 새로 만들기보다 이 helper의 포맷을 따른다.

## 새 Query Tool 체크리스트
- 팀 표시명이 필요하면 `team_display`부터 검토한다.
- 팀명 매핑을 DB에서 읽어야 하면 `team_mapping_loader`를 우선 사용한다.
- connection pool이나 closed-connection retry가 필요하면 `pooled_connection`을 붙인다.
- 반환값이 dict 기반이면 `query_result`로 success/error shape를 맞춘다.
- 운영 로그를 남기면 `query_logging`의 event/action 포맷을 따른다.
- 기존 public response key와 로그 action 문자열은 바꾸지 않는다.
- helper로 해결되는 영역에 대해 새 mixin, base class, 중복 retry 코드를 추가하지 않는다.

## 이번 정리 묶음
- 팀 매핑/표시 경로
  - `app/agents/baseball_agent.py`
  - `app/tools/database_query.py`
  - `app/tools/game_query.py`
  - `app/core/renderers/baseball.py`
  - `app/tools/team_display.py`
  - `app/tools/team_mapping_loader.py`
- 문서/규정 query 안정화 경로
  - `app/tools/document_query.py`
  - `app/tools/regulation_query.py`
  - `app/tools/pooled_connection.py`
  - `app/tools/query_result.py`
  - `app/tools/query_logging.py`

## 유지 원칙
- helper는 동작을 감추기 위한 추상화보다, 반복 패턴을 고정하기 위한 얇은 공용 계층으로 유지한다.
- 새 회귀가 생기면 helper를 늘리기 전에 먼저 해당 계약을 테스트로 고정한다.
- generated artifact는 `coverage.xml`, `htmlcov/`처럼 `.gitignore`에서 차단한다.
