from datetime import date
import importlib.util
from pathlib import Path
import sys


def _resolve_module_path() -> Path:
    test_file = Path(__file__).resolve()
    candidates = (
        test_file.parents[1] / "scripts" / "repair_postseason_season_ids.py",
        test_file.parents[2] / "scripts" / "repair_postseason_season_ids.py",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


MODULE_PATH = _resolve_module_path()
MODULE_SPEC = importlib.util.spec_from_file_location(
    "repair_postseason_season_ids_root",
    MODULE_PATH,
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)

GameRow = MODULE.GameRow
MismatchRow = MODULE.MismatchRow
SeasonStage = MODULE.SeasonStage
collect_mismatches = MODULE.collect_mismatches
mismatch_exit_code = MODULE.mismatch_exit_code


def test_collect_mismatches_flags_postseason_stage_scope_errors() -> None:
    stages_by_year = {
        2025: {
            2: SeasonStage(2025, 2, 261, date(2025, 10, 6)),
            3: SeasonStage(2025, 3, 262, date(2025, 10, 9)),
            4: SeasonStage(2025, 4, 263, date(2025, 10, 18)),
            5: SeasonStage(2025, 5, 264, date(2025, 10, 26)),
        }
    }
    games = [
        GameRow("20251006NCSS0", date(2025, 10, 6), 264, 5, "SS", "NC"),
        GameRow("20251026LGHH0", date(2025, 10, 26), 264, 5, "HH", "LG"),
    ]

    mismatches = collect_mismatches(games, stages_by_year)

    assert len(mismatches) == 1
    assert mismatches[0].game_id == "20251006NCSS0"
    assert mismatches[0].target_season_id == 261
    assert mismatches[0].inferred_code == 2


def test_mismatch_exit_code_fails_only_for_dry_run_mismatches() -> None:
    mismatches = [MismatchRow("ignored", date(2025, 10, 6), "SS", "NC", 264, 5, 2, 261)]

    assert mismatch_exit_code(mismatches, apply=False, fail_on_mismatch=True) == 2
    assert mismatch_exit_code(mismatches, apply=True, fail_on_mismatch=True) == 0
    assert mismatch_exit_code(mismatches, apply=False, fail_on_mismatch=False) == 0
    assert mismatch_exit_code([], apply=False, fail_on_mismatch=True) == 0
