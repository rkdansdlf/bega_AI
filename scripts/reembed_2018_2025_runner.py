from __future__ import annotations

from scripts.run_semantic_chunk_rollout import main as rollout_main

if __name__ == "__main__":
    raise SystemExit(
        rollout_main(
            [
                "--start-year",
                "2018",
                "--end-year",
                "2025",
                "--skip-benchmark",
            ]
        )
    )
