from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
DEV_REQUIREMENTS = (ROOT / "requirements-dev").read_text(encoding="utf-8")


def test_ci_uses_pinned_black_and_full_comparison_history() -> None:
    assert "black==26.5.1" in DEV_REQUIREMENTS
    assert "fetch-depth: 0" in WORKFLOW
    assert "pip install -r requirements-dev" in WORKFLOW
    assert "pip install flake8 black" not in WORKFLOW


def test_ci_maps_events_to_the_correct_git_comparison() -> None:
    assert "github.event.pull_request.base.sha" in WORKFLOW
    assert "github.event.pull_request.head.sha" in WORKFLOW
    assert "github.event.before" in WORKFLOW
    assert (
        "COMPARISON: ${{ github.event_name == 'pull_request' "
        "&& 'merge-base' || 'range' }}"
    ) in WORKFLOW


def test_ci_formats_only_nul_safe_changed_python_paths() -> None:
    assert "scripts/list_changed_python_files.py" in WORKFLOW
    assert "mapfile -d '' changed_python_files" in WORKFLOW
    assert '> "${changed_files_output}"' in WORKFLOW
    assert 'black --check -- "${changed_python_files[@]}"' in WORKFLOW
    assert "black --check ." not in WORKFLOW
