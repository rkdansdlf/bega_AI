# KBO 야구 기록 지표 설명

이 문서는 **KBO 플랫폼 데이터베이스의 실제 스키마**를 기반으로 한 야구 기록과 지표에 대한 설명을 담고 있습니다. 이 정보는 AI 챗봇의 지식 베이스로 활용되며, **환각 방지**를 위해 실제 데이터베이스에 존재하는 컬럼과 데이터만을 기준으로 작성되었습니다.

## 데이터베이스 구조 개요

### 주요 테이블
- **player_basic**: 선수 기본 정보 (이름, 포지션, 생년월일 등)
- **player_season_batting**: 선수별 시즌 타격 기록 
- **player_season_pitching**: 선수별 시즌 투구 기록
- **teams**: 팀 정보 (팀명, 창단연도, 연고지 등)

### 데이터 범위
- **시즌**: 1982년부터 현재까지의 KBO 데이터
- **리그**: '정규시즌', '포스트시즌', '올스타' 구분
- **레벨**: 'KBO1' (1군), 'KBO2' (2군/퓨처스리그) 구분

---

## 타자 기록 (player_season_batting 테이블)

** 중요**: 아래는 실제 데이터베이스의 `player_season_batting` 테이블에 존재하는 컬럼들입니다. 이 컬럼들만 사용하여 답변해야 합니다.

### 기본 식별 정보
- **player_id**: 선수 고유 ID (player_basic 테이블과 JOIN하여 이름 조회)
- **season**: 시즌 연도 (예: 2024)
- **team_code**: 팀 코드 (teams 테이블의 team_id와 매핑)
- **league**: 리그 구분 ('정규시즌', '포스트시즌', '올스타')
- **level**: 레벨 구분 ('KBO1', 'KBO2')

### 실제 존재하는 타격 지표 (DB 컬럼명 기준)
- **games**: 출전 경기 수
- **plate_appearances**: 타석 (PA)
- **at_bats**: 타수 (AB) 
- **runs**: 득점 (R)
- **hits**: 안타 (H)
- **doubles**: 2루타 (2B)
- **triples**: 3루타 (3B) 
- **home_runs**: 홈런 (HR)
- **rbi**: 타점 (RBI)
- **walks**: 볼넷 (BB)
- **intentional_walks**: 고의사구 (IBB)
- **hbp**: 사구 (HBP)
- **strikeouts**: 삼진 (SO)
- **stolen_bases**: 도루 (SB)
- **caught_stealing**: 도루실패 (CS)
- **sacrifice_hits**: 희생번트 (SAC)
- **sacrifice_flies**: 희생플라이 (SF)
- **gdp**: 병살타 (GDP)

### 세부 공격 지표 (추가 데이터)
- **iso**: 순장타율 (ISO) = slg - avg
- **babip**: 인플레이타구안타율 (BABIP)
- **extra_stats**: JSON 형태의 추가 통계
  - **xr**: 추정 득점 (Extrapolated Runs)

### 최소 샘플 기준 (신뢰도 있는 통계를 위해)
- **정규 타자**: plate_appearances >= 100 (약 100타석 이상)
- **주전 타자**: plate_appearances >= 300 (약 300타석 이상)

---

## 투수 기록 (player_season_pitching 테이블)

**중요**: 아래는 실제 데이터베이스의 `player_season_pitching` 테이블에 존재하는 컬럼들입니다.

### 기본 식별 정보
- **player_id**: 선수 고유 ID (player_basic 테이블과 JOIN하여 이름 조회)
- **season**: 시즌 연도
- **team_code**: 팀 코드
- **league**: 리그 구분
- **level**: 레벨 구분

### 실제 존재하는 투구 지표 (DB 컬럼명 기준)
- **games**: 출전 경기 수 (G)
- **games_started**: 선발 경기 수 (GS)
- **wins**: 승수 (W)
- **losses**: 패수 (L)
- **saves**: 세이브 (SV)
- **holds**: 홀드 (HLD)
- **innings_pitched**: 투구 이닝 (IP) - 소수점 표기 (예: 180.1)
- **hits_allowed**: 피안타 (H)
- **runs_allowed**: 실점 (R)
- **earned_runs**: 자책점 (ER)
- **home_runs_allowed**: 피홈런 (HR)
- **walks_allowed**: 볼넷 허용 (BB)
- **strikeouts**: 삼진 (SO)
- **hit_batters**: 사구 (HBP)

### 계산된 지표 (실제 DB 컬럼)
- **era**: 평균자책점 (ERA) = (earned_runs × 9) / innings_pitched
- **whip**: WHIP = (hits_allowed + walks_allowed) / innings_pitched
- **k_per_nine**: 9이닝당 탈삼진 (K/9)
- **bb_per_nine**: 9이닝당 볼넷 (BB/9)
- **kbb**: 탈삼진/볼넷 비율 (K/BB)

### 추가 통계 정보
- **extra_stats**: JSON 형태의 추가 통계
  - **fip**: 수비 독립 평균자책점 (Fielding Independent Pitching)
  - **war**: 대체 선수 대비 승수 기여도

---

## 💥 WPA & 클러치 상황 (game_events 테이블)

WPA는 각 플레이가 팀의 승리 확률을 얼마나 변화시켰는지를 나타냅니다.

### 주요 개념
- **WPA (Win Probability Added)**: 승리 확률 기여도. 양수(+)는 승리 확률 증가, 음수(-)는 감소를 의미.
- **클러치 상황 (Clutch)**: WPA 절대값이 큰 상황 (0.05 이상). 경기 승부처를 의미.
- **레버리지 인덱스 (LI)**: 상황의 중요도.

### WPA 리더보드
- **타자 WPA**: 승부처에서 안타/홈런 등으로 팀 승리에 기여한 정도.
- **투수 WPA**: 위기 상황을 막아내어 팀 승리 확률을 지킨 정도.

---

## 👥 선수 정보 (player_basic 테이블)

### 실제 존재하는 선수 정보 컬럼
- **player_id**: 선수 고유 ID
- **name**: 선수 이름 (한글)
- **uniform_no**: 등번
- **position**: 포지션 ('내야수', '외야수', '투수', '포수' 등)
- **birth_date**: 생년월일
- **height_cm**: 키 (cm)
- **weight_kg**: 몸무게 (kg)
- **career**: 출신 학교/경력
- **team_id**: 소속 팀 ID

---

## 팀 정보 (teams 테이블)

### 실제 존재하는 팀 정보 컬럼
- **team_id**: 팀 고유 ID (HT, LG, SS 등)
- **team_short_name**: 팀 줄임말 ('기아', 'LG', '삼성' 등)
- **city**: 연고지
- **stadium_name**: 홈구장명
- **team_name**: 정식 팀명 ('기아 타이거즈', 'LG 트윈스' 등)
- **founded_year**: 창단연도

### 팀 코드 매핑 (실제 데이터베이스 기준)
- **HT**: 기아 타이거즈
- **LG**: LG 트윈스  
- **SS**: 삼성 라이온즈
- **LT**: 롯데 자이언츠
- **OB**: 두산 베어스
- **WO**: 키움 히어로즈
- **HH**: 한화 이글스
- **KT**: KT 위즈
- **NC**: NC 다이노스
- **SK**: SSG 랜더스

---

## 절대 사용 금지 컬럼/정보

**이 항목들은 데이터베이스에 없으므로 절대 언급하면 안 됩니다:**

### 타자 관련
- player_name (대신 player_basic.name과 JOIN)
- team_name (대신 teams.team_name과 JOIN)
- season_year (대신 season 사용)
- ops_plus (extra_stats JSON에서 확인 필요)
- wrc_plus (extra_stats JSON에서 확인 필요)
- war (extra_stats JSON에서 확인 필요)

### 투수 관련  
- player_name (대신 player_basic.name과 JOIN)
- team_name (대신 teams.team_name과 JOIN)
- tbf (상대한 타자수 - 컬럼 없음)
- fip (FIP 지표 - extra_stats에서 확인 필요)

---

## 쿼리 작성 가이드

### 올바른 JOIN 패턴
```sql
-- 선수 타격 성적 조회 예시
SELECT 
    pb.name as player_name,
    t.team_name,
    psb.season,
    psb.avg, psb.ops, psb.home_runs
FROM player_season_batting psb
JOIN player_basic pb ON psb.player_id = pb.player_id
LEFT JOIN teams t ON psb.team_code = t.team_id
WHERE pb.name LIKE '%김도영%' 
AND psb.season = 2024 
AND psb.league = '정규시즌';
```

### 필수 필터링 조건
- **league = '정규시즌'**: 정규시즌 데이터만 조회
- **level = 'KBO1'**: 1군 데이터만 조회 (일반적)
- **최소 샘플**: 타자 100타석, 투수 30이닝 이상

이 문서의 정보만을 사용하여 정확한 데이터베이스 조회와 답변을 생성해야 합니다.