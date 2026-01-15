-- v_team_rank_all 뷰 생성
-- 팀의 시즌별 순위를 계산하여 제공하는 뷰

CREATE OR REPLACE VIEW v_team_rank_all AS
WITH team_stats AS (
    SELECT 
        ks.season_year,
        team,
        SUM(CASE WHEN winning_team = team THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN winning_team IS NOT NULL AND winning_team != team THEN 1 ELSE 0 END) as losses,
        SUM(CASE WHEN winning_team IS NULL AND home_score = away_score THEN 1 ELSE 0 END) as draws
    FROM (
        SELECT season_id, home_team as team, winning_team, home_score, away_score 
        FROM game 
        WHERE game_status = 'COMPLETED'
        
        UNION ALL
        
        SELECT season_id, away_team as team, winning_team, home_score, away_score 
        FROM game 
        WHERE game_status = 'COMPLETED'
    ) all_games
    JOIN kbo_seasons ks ON all_games.season_id = ks.season_id
    WHERE ks.league_type_code = '0' -- 정규시즌만
    GROUP BY ks.season_year, team
),
ranked_stats AS (
    SELECT 
        season_year,
        team as team_id,
        team as team_name, -- team_id와 team_name이 동일하다고 가정 (또는 별도 매핑 필요)
        wins,
        losses,
        draws,
        RANK() OVER (PARTITION BY season_year ORDER BY (wins::float / NULLIF(wins + losses, 0)) DESC) as season_rank
    FROM team_stats
)
SELECT 
    season_year,
    team_id,
    team_name,
    season_rank,
    wins,
    losses,
    draws
FROM ranked_stats;
