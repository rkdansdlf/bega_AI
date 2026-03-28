-- P0-1: player_season_batting → team_season_batting 집계
-- 멱등성: ON CONFLICT (team_id, season, league) DO UPDATE
-- FK: COALESCE(canonical_team_code, team_code) → teams(team_id)

TRUNCATE team_season_batting RESTART IDENTITY;

INSERT INTO team_season_batting (
    team_id, team_name, season, league,
    games, plate_appearances, at_bats, runs, hits,
    doubles, triples, home_runs, rbi,
    stolen_bases, caught_stealing, walks, strikeouts,
    avg, obp, slg, ops,
    extra_stats, created_at, updated_at
)
SELECT
    COALESCE(psb.canonical_team_code, psb.team_code)  AS team_id,
    COALESCE(t.team_name, COALESCE(psb.canonical_team_code, psb.team_code)) AS team_name,
    psb.season,
    psb.league,
    -- games: 팀에서 가장 많이 출장한 선수의 경기 수 ≈ 팀 전체 경기 수
    MAX(psb.games)                                     AS games,
    SUM(COALESCE(psb.plate_appearances, 0))            AS plate_appearances,
    SUM(COALESCE(psb.at_bats, 0))                      AS at_bats,
    SUM(COALESCE(psb.runs, 0))                         AS runs,
    SUM(COALESCE(psb.hits, 0))                         AS hits,
    SUM(COALESCE(psb.doubles, 0))                      AS doubles,
    SUM(COALESCE(psb.triples, 0))                      AS triples,
    SUM(COALESCE(psb.home_runs, 0))                    AS home_runs,
    SUM(COALESCE(psb.rbi, 0))                          AS rbi,
    SUM(COALESCE(psb.stolen_bases, 0))                 AS stolen_bases,
    SUM(COALESCE(psb.caught_stealing, 0))              AS caught_stealing,
    SUM(COALESCE(psb.walks, 0))                        AS walks,
    SUM(COALESCE(psb.strikeouts, 0))                   AS strikeouts,
    -- AVG = H / AB
    ROUND(CASE WHEN SUM(COALESCE(psb.at_bats, 0)) > 0
        THEN SUM(COALESCE(psb.hits, 0))::numeric / SUM(psb.at_bats)
        ELSE 0 END, 3)::double precision               AS avg,
    -- OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
    ROUND(CASE WHEN (SUM(COALESCE(psb.at_bats,0)) + SUM(COALESCE(psb.walks,0))
                   + SUM(COALESCE(psb.hbp,0)) + SUM(COALESCE(psb.sacrifice_flies,0))) > 0
        THEN (SUM(COALESCE(psb.hits,0)) + SUM(COALESCE(psb.walks,0)) + SUM(COALESCE(psb.hbp,0)))::numeric
             / (SUM(COALESCE(psb.at_bats,0)) + SUM(COALESCE(psb.walks,0))
              + SUM(COALESCE(psb.hbp,0)) + SUM(COALESCE(psb.sacrifice_flies,0)))
        ELSE 0 END, 3)::double precision               AS obp,
    -- SLG = (1B + 2*2B + 3*3B + 4*HR) / AB  =  (H + 2B + 2*3B + 3*HR) / AB
    ROUND(CASE WHEN SUM(COALESCE(psb.at_bats, 0)) > 0
        THEN (SUM(COALESCE(psb.hits,0))
            + SUM(COALESCE(psb.doubles,0))
            + 2 * SUM(COALESCE(psb.triples,0))
            + 3 * SUM(COALESCE(psb.home_runs,0)))::numeric
             / SUM(psb.at_bats)
        ELSE 0 END, 3)::double precision               AS slg,
    -- OPS = OBP + SLG (재계산)
    ROUND(
        CASE WHEN (SUM(COALESCE(psb.at_bats,0)) + SUM(COALESCE(psb.walks,0))
                 + SUM(COALESCE(psb.hbp,0)) + SUM(COALESCE(psb.sacrifice_flies,0))) > 0
            THEN (SUM(COALESCE(psb.hits,0)) + SUM(COALESCE(psb.walks,0)) + SUM(COALESCE(psb.hbp,0)))::numeric
                 / (SUM(COALESCE(psb.at_bats,0)) + SUM(COALESCE(psb.walks,0))
                  + SUM(COALESCE(psb.hbp,0)) + SUM(COALESCE(psb.sacrifice_flies,0)))
            ELSE 0 END
        +
        CASE WHEN SUM(COALESCE(psb.at_bats, 0)) > 0
            THEN (SUM(COALESCE(psb.hits,0))
                + SUM(COALESCE(psb.doubles,0))
                + 2 * SUM(COALESCE(psb.triples,0))
                + 3 * SUM(COALESCE(psb.home_runs,0)))::numeric
                 / SUM(psb.at_bats)
            ELSE 0 END
    , 3)::double precision                             AS ops,
    NULL                                               AS extra_stats,
    NOW()                                              AS created_at,
    NOW()                                              AS updated_at
FROM player_season_batting psb
LEFT JOIN teams t ON t.team_id = COALESCE(psb.canonical_team_code, psb.team_code)
WHERE psb.team_code IS NOT NULL
GROUP BY COALESCE(psb.canonical_team_code, psb.team_code), t.team_name, psb.season, psb.league
ORDER BY psb.season DESC, COALESCE(psb.canonical_team_code, psb.team_code);
