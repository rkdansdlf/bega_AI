from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.ingest_from_kbo import ingest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = PROJECT_ROOT / "scripts" / "verify_embedding_coverage.py"
VERIFY_OUTPUT_JSON = PROJECT_ROOT / "logs" / "embedding_coverage.json"
VERIFY_OUTPUT_CSV = PROJECT_ROOT / "logs" / "embedding_coverage.csv"


def run_verifier() -> None:
    cmd = [
        sys.executable,
        str(VERIFY_SCRIPT),
        "--mode",
        "all",
        "--start-year",
        "2018",
        "--end-year",
        "2025",
        "--output",
        str(VERIFY_OUTPUT_JSON),
        "--csv-output",
        str(VERIFY_OUTPUT_CSV),
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run() -> None:
    non_season_tables = [
        "teams",
        "team_franchises",
        "stadiums",
        "player_basic",
        "player_movements",
        "kbo_metrics_explained",
        "kbo_regulations_basic",
        "kbo_regulations_player",
        "kbo_regulations_game",
        "kbo_regulations_technical",
        "kbo_regulations_discipline",
        "kbo_regulations_postseason",
        "kbo_regulations_special",
        "kbo_regulations_terms",
    ]
    print("===== START NON-SEASON TABLES =====", flush=True)
    ingest(
        tables=non_season_tables,
        limit=None,
        embed_batch_size=32,
        read_batch_size=500,
        season_year=None,
        use_legacy_renderer=False,
        since=None,
        skip_embedding=False,
        max_concurrency=2,
        commit_interval=1000,
    )
    print("===== END NON-SEASON TABLES =====", flush=True)

    season_tables = [
        "kbo_seasons",
        "team_history",
        "awards",
        "player_season_batting",
        "player_season_pitching",
        "game",
        "game_metadata",
        "game_inning_scores",
        "game_lineups",
        "game_batting_stats",
        "game_pitching_stats",
        "game_summary",
    ]

    for year in range(2018, 2026):
        print(f"\n===== START YEAR {year} =====", flush=True)
        ingest(
            tables=season_tables,
            limit=None,
            embed_batch_size=32,
            read_batch_size=500,
            season_year=year,
            use_legacy_renderer=False,
            since=None,
            skip_embedding=False,
            max_concurrency=2,
            commit_interval=1000,
        )
        print(f"===== END YEAR {year} =====\n", flush=True)

    print("===== START COVERAGE VERIFY =====", flush=True)
    run_verifier()
    print("===== END COVERAGE VERIFY =====", flush=True)


if __name__ == "__main__":
    run()
