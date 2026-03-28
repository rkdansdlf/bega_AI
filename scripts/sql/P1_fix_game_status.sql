-- P1-1: game_status NULL 채우기
-- 규칙:
--   1. 스코어 존재 → COMPLETED
--   2. 미래 경기 → SCHEDULED
--   3. 과거 + 스코어 없음 → UNKNOWN

-- Step 1: 스코어가 있는 경기 → COMPLETED
UPDATE game
SET game_status = 'COMPLETED'
WHERE game_status IS NULL
  AND home_score IS NOT NULL
  AND away_score IS NOT NULL;

-- Step 2: 미래 경기 → SCHEDULED
UPDATE game
SET game_status = 'SCHEDULED'
WHERE game_status IS NULL
  AND game_date > CURRENT_DATE;

-- Step 3: 나머지 (과거 + 스코어 없음) → UNKNOWN
UPDATE game
SET game_status = 'UNKNOWN'
WHERE game_status IS NULL
  AND game_date <= CURRENT_DATE
  AND (home_score IS NULL OR away_score IS NULL);

-- 남은 NULL 처리 (혹시 모를 edge case)
UPDATE game
SET game_status = 'UNKNOWN'
WHERE game_status IS NULL;
