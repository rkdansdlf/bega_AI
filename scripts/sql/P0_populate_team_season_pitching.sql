-- P0-2: player_season_pitching → team_season_pitching 집계
-- 멱등성: TRUNCATE + INSERT
-- FK: COALESCE(canonical_team_code, team_code) → teams(team_id)

TRUNCATE team_season_pitching RESTART IDENTITY;

INSERT INTO team_season_pitching (
    team_id, team_name, season, league,
    games, wins, losses, ties, saves, holds,
    innings_pitched, runs_allowed, earned_runs,
    hits_allowed, home_runs_allowed, walks_allowed, strikeouts,
    era, whip, avg_against,
    extra_stats, created_at, updated_at
)
SELECT
    COALESCE(psp.canonical_team_code, psp.team_code)  AS team_id,
    COALESCE(t.team_name, COALESCE(psp.canonical_team_code, psp.team_code)) AS team_name,
    psp.season,
    psp.league,
    MAX(psp.games)                                     AS games,
    SUM(COALESCE(psp.wins, 0))                         AS wins,
    SUM(COALESCE(psp.losses, 0))                       AS losses,
    0                                                  AS ties,
    SUM(COALESCE(psp.saves, 0))                        AS saves,
    SUM(COALESCE(psp.holds, 0))                        AS holds,
    SUM(COALESCE(psp.innings_pitched, 0))              AS innings_pitched,
    SUM(COALESCE(psp.runs_allowed, 0))                 AS runs_allowed,
    SUM(COALESCE(psp.earned_runs, 0))                  AS earned_runs,
    SUM(COALESCE(psp.hits_allowed, 0))                 AS hits_allowed,
    SUM(COALESCE(psp.home_runs_allowed, 0))            AS home_runs_allowed,
    SUM(COALESCE(psp.walks_allowed, 0))                AS walks_allowed,
    SUM(COALESCE(psp.strikeouts, 0))                   AS strikeouts,
    -- ERA = (ER * 9) / IP
    ROUND((CASE WHEN SUM(COALESCE(psp.innings_pitched, 0)) > 0
        THEN (SUM(COALESCE(psp.earned_runs, 0)) * 9.0) / SUM(psp.innings_pitched)
        ELSE 0 END)::numeric, 2)::double precision     AS era,
    -- WHIP = (BB + H) / IP
    ROUND((CASE WHEN SUM(COALESCE(psp.innings_pitched, 0)) > 0
        THEN (SUM(COALESCE(psp.walks_allowed, 0)) + SUM(COALESCE(psp.hits_allowed, 0)))::numeric
             / SUM(psp.innings_pitched)
        ELSE 0 END)::numeric, 2)::double precision     AS whip,
    -- AVG Against = HA / (TBF - BB - HBP)
    ROUND((CASE WHEN (SUM(COALESCE(psp.tbf, 0)) - SUM(COALESCE(psp.walks_allowed, 0))
                   - SUM(COALESCE(psp.hit_batters, 0))) > 0
        THEN SUM(COALESCE(psp.hits_allowed, 0))::numeric
             / (SUM(COALESCE(psp.tbf, 0)) - SUM(COALESCE(psp.walks_allowed, 0))
              - SUM(COALESCE(psp.hit_batters, 0)))
        ELSE NULL END)::numeric, 3)::double precision  AS avg_against,
    NULL                                               AS extra_stats,
    NOW()                                              AS created_at,
    NOW()                                              AS updated_at
FROM player_season_pitching psp
LEFT JOIN teams t ON t.team_id = COALESCE(psp.canonical_team_code, psp.team_code)
WHERE psp.team_code IS NOT NULL
GROUP BY COALESCE(psp.canonical_team_code, psp.team_code), t.team_name, psp.season, psp.league
ORDER BY psp.season DESC, COALESCE(psp.canonical_team_code, psp.team_code);
