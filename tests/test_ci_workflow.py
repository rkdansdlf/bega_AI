from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
DEV_REQUIREMENTS = (ROOT / "requirements-dev").read_text(encoding="utf-8")
LINT_JOB = WORKFLOW[WORKFLOW.index("  lint:\n") : WORKFLOW.index("\n  test:\n")]
FORMAT_STEP = LINT_JOB[
    LINT_JOB.index("      - name: Check changed Python formatting\n") : LINT_JOB.index(
        "\n      - name: Lint Summary\n"
    )
]


def test_ci_uses_pinned_black_and_full_comparison_history() -> None:
    assert "black==26.5.1" in DEV_REQUIREMENTS
    assert "fetch-depth: 0" in LINT_JOB
    assert "pip install -r requirements-dev" in LINT_JOB
    assert "pip install flake8 black" not in LINT_JOB


def test_ci_maps_events_to_the_correct_git_comparison() -> None:
    assert "github.event.pull_request.base.sha" in FORMAT_STEP
    assert "github.event.pull_request.head.sha" in FORMAT_STEP
    assert "github.event.before" in FORMAT_STEP
    assert (
        "COMPARISON: ${{ github.event_name == 'pull_request' "
        "&& 'merge-base' || 'range' }}"
    ) in FORMAT_STEP


def test_ci_formats_only_nul_safe_changed_python_paths() -> None:
    assert "scripts/list_changed_python_files.py" in FORMAT_STEP
    assert "mapfile -d '' changed_python_files" in FORMAT_STEP
    assert '> "${changed_files_output}"' in FORMAT_STEP
    assert 'black --check -- "${changed_python_files[@]}"' in FORMAT_STEP
    assert "black --check ." not in LINT_JOB
