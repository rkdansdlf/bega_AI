# Incremental Black Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AI CI enforce pinned Black formatting on changed Python files without allowing pre-existing whole-repository format debt to block unrelated changes.

**Architecture:** A standard-library Python CLI asks Git for NUL-delimited added, copied, modified, or renamed `.py` paths. The lint workflow maps push events to a two-dot commit range and pull requests to a three-dot merge-base comparison, then passes the exact path array to the Black version pinned in `requirements-dev`.

**Tech Stack:** Python 3.14, Git CLI, GitHub Actions YAML, Bash, Black 26.5.1, pytest.

## Global Constraints

- Do not reformat unchanged Python files or clean up existing formatting debt.
- Keep `black==26.5.1` in `requirements-dev` as the formatter source of truth.
- Preserve the existing fatal Flake8 check, unit tests, dependency audits, and container scan.
- Pull requests use merge-base comparison; pushes use the exact before-to-head range.
- An all-zero push base compares the head against Git's empty tree.
- Include only added, copied, modified, or renamed `.py` files; exclude deleted and non-Python files.
- Emit NUL-delimited paths and consume them without word splitting.
- Use only trusted repository Git history; do not query an external data source.
- Use `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python` for local Python commands.
- Run each RED test before production code, confirm the expected failure, and implement only enough for GREEN.

---

## File Responsibility Map

- `scripts/list_changed_python_files.py`: validate CLI arguments, resolve a zero base to Git's empty tree, execute the selected Git diff, and forward NUL-delimited changed Python paths.
- `tests/test_list_changed_python_files.py`: exercise the CLI against temporary Git histories and unusual path names.
- `.github/workflows/ci.yml`: fetch comparison history, install pinned lint requirements, map GitHub event SHAs, and run Black on the selected array only.
- `tests/test_ci_workflow.py`: lock the workflow's formatter and comparison contract.

---

### Task 1: Select changed Python files safely

**Files:**
- Create: `scripts/list_changed_python_files.py`
- Create: `tests/test_list_changed_python_files.py`

**Interfaces:**
- Produces: CLI flags `--base: str`, `--head: str`, and `--comparison: Literal["range", "merge-base"]`; successful stdout is a NUL-delimited byte sequence of repository-relative `.py` paths.
- Consumes: the current Git repository and locally available commits only.

- [ ] **Step 1: Write the failing CLI tests**

Create the complete test module with temporary Git helpers and these cases:

```python
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "list_changed_python_files.py"


def run_git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def initialize_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    run_git(repository, "init", "-q", "-b", "main")
    run_git(repository, "config", "user.email", "tests@example.invalid")
    run_git(repository, "config", "user.name", "CI Tests")
    return repository


def commit_all(repository: Path, message: str) -> str:
    run_git(repository, "add", "-A")
    run_git(repository, "commit", "-qm", message)
    return run_git(repository, "rev-parse", "HEAD")


def commit_files(repository: Path, files: dict[str, str]) -> str:
    for name, content in files.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return commit_all(repository, "update files")


def run_selector(
    repository: Path, base: str, head: str, comparison: str
) -> list[bytes]:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base",
            base,
            "--head",
            head,
            "--comparison",
            comparison,
        ],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [path for path in completed.stdout.split(b"\0") if path]


def test_range_lists_only_live_changed_python_paths(tmp_path: Path) -> None:
    repository = initialize_repository(tmp_path)
    base = commit_files(
        repository,
        {
            "keep.py": "x = 1\n",
            "delete.py": "x = 1\n",
            "rename_old.py": "x = 1\n",
        },
    )
    (repository / "keep.py").write_text("x = 2\n", encoding="utf-8")
    (repository / "delete.py").unlink()
    (repository / "notes.md").write_text("notes\n", encoding="utf-8")
    (repository / "space name.py").write_text("x = 3\n", encoding="utf-8")
    run_git(repository, "mv", "rename_old.py", "rename_new.py")
    head = commit_all(repository, "change files")

    assert run_selector(repository, base, head, "range") == [
        b"keep.py",
        b"rename_new.py",
        b"space name.py",
    ]


def test_merge_base_excludes_changes_made_only_on_base_branch(tmp_path: Path) -> None:
    repository = initialize_repository(tmp_path)
    root = commit_files(repository, {"shared.py": "x = 1\n"})
    run_git(repository, "switch", "-c", "feature")
    feature = commit_files(repository, {"feature.py": "x = 1\n"})
    run_git(repository, "switch", "main")
    base = commit_files(repository, {"base_only.py": "x = 1\n"})

    assert run_selector(repository, base, feature, "merge-base") == [b"feature.py"]
    assert root != base


def test_zero_base_lists_python_files_from_head(tmp_path: Path) -> None:
    repository = initialize_repository(tmp_path)
    head = commit_files(repository, {"first.py": "x = 1\n", "README.md": "read\n"})

    assert run_selector(repository, "0" * 40, head, "range") == [b"first.py"]


def test_output_is_nul_safe_for_newline_in_path(tmp_path: Path) -> None:
    repository = initialize_repository(tmp_path)
    base = commit_files(repository, {"base.py": "x = 1\n"})
    head = commit_files(repository, {"line\nbreak.py": "x = 2\n"})

    assert run_selector(repository, base, head, "range") == [b"line\nbreak.py"]


def test_invalid_nonzero_ref_fails_loudly(tmp_path: Path) -> None:
    repository = initialize_repository(tmp_path)
    head = commit_files(repository, {"first.py": "x = 1\n"})

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base",
            "missing-ref",
            "--head",
            head,
            "--comparison",
            "range",
        ],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode != 0
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_list_changed_python_files.py -q
```

Expected: failures because `scripts/list_changed_python_files.py` does not exist.

- [ ] **Step 3: Implement the minimal selector CLI**

Create the complete standard-library script:

```python
"""List changed Python files for CI as NUL-delimited repository paths."""

from __future__ import annotations

import argparse
import subprocess
import sys


ZERO_SHA = "0" * 40


def git_output(arguments: list[str], *, stdin: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *arguments], input=stdin, stdout=subprocess.PIPE, check=True
    ).stdout


def empty_tree() -> str:
    return git_output(["hash-object", "-t", "tree", "--stdin"], stdin=b"").decode().strip()


def changed_python_files(base: str, head: str, comparison: str) -> bytes:
    if base == ZERO_SHA:
        base = empty_tree()
        comparison = "range"
    separator = "..." if comparison == "merge-base" else ".."
    return git_output(
        [
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
            f"{base}{separator}{head}",
            "--",
            "*.py",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument(
        "--comparison", required=True, choices=("range", "merge-base")
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    sys.stdout.buffer.write(
        changed_python_files(
            arguments.base,
            arguments.head,
            arguments.comparison,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2.

Expected: all selector tests pass.

- [ ] **Step 5: Commit the selector and tests**

```bash
git add scripts/list_changed_python_files.py tests/test_list_changed_python_files.py
git commit -m "ci: select changed Python files for formatting"
```

---

### Task 2: Wire incremental Black enforcement into CI

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `tests/test_ci_workflow.py`

**Interfaces:**
- Consumes: NUL-delimited stdout from `scripts/list_changed_python_files.py`; GitHub event fields `event_name`, `event.before`, `pull_request.base.sha`, `pull_request.head.sha`, and `sha`.
- Produces: a lint gate that runs `black --check -- "${changed_python_files[@]}"` only when the array is non-empty.

- [ ] **Step 1: Write the failing workflow contract tests**

Create `tests/test_ci_workflow.py` with:

```python
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
    assert "COMPARISON: ${{ github.event_name == 'pull_request' && 'merge-base' || 'range' }}" in WORKFLOW


def test_ci_formats_only_nul_safe_changed_python_paths() -> None:
    assert "scripts/list_changed_python_files.py" in WORKFLOW
    assert "mapfile -d '' changed_python_files" in WORKFLOW
    assert 'black --check -- "${changed_python_files[@]}"' in WORKFLOW
    assert "black --check ." not in WORKFLOW
```

- [ ] **Step 2: Run the workflow test and verify RED**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_ci_workflow.py -q
```

Expected: failures because checkout is shallow, lint tools are installed ad hoc,
event refs are absent, and Black checks the entire repository.

- [ ] **Step 3: Update the lint workflow**

Set `fetch-depth: 0` on the lint checkout. Replace the lint tool install with
`pip install -r requirements-dev`. Add this environment and formatting step:

```yaml
      - name: Check changed Python formatting
        env:
          BASE_SHA: ${{ github.event_name == 'pull_request' && github.event.pull_request.base.sha || github.event.before }}
          HEAD_SHA: ${{ github.event_name == 'pull_request' && github.event.pull_request.head.sha || github.sha }}
          COMPARISON: ${{ github.event_name == 'pull_request' && 'merge-base' || 'range' }}
        run: |
          changed_files_output="${RUNNER_TEMP}/changed-python-files"
          python scripts/list_changed_python_files.py \
            --base "${BASE_SHA}" \
            --head "${HEAD_SHA}" \
            --comparison "${COMPARISON}" \
            > "${changed_files_output}"
          mapfile -d '' changed_python_files < "${changed_files_output}"
          if (( ${#changed_python_files[@]} == 0 )); then
            echo "No changed Python files to format-check."
          else
            black --check -- "${changed_python_files[@]}"
          fi
```

- [ ] **Step 4: Run both focused test files and verify GREEN**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_list_changed_python_files.py tests/test_ci_workflow.py -q
```

Expected: all selector and workflow contract tests pass.

- [ ] **Step 5: Run Black on the branch's changed Python files**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python \
  scripts/list_changed_python_files.py \
  --base main --head HEAD --comparison range \
  | xargs -0 /Users/mac/project/KBO_platform/bega_AI/.venv/bin/python \
      -m black --check --
```

Expected: both new Python files pass Black 26.5.1.

- [ ] **Step 6: Commit the workflow gate**

```bash
git add .github/workflows/ci.yml tests/test_ci_workflow.py
git commit -m "ci: enforce Black on changed Python files"
```

---

### Task 3: Verify and integrate the branch

**Files:**
- Verify only: all files changed since `main`.

**Interfaces:**
- Consumes: the completed branch and the existing `main` head.
- Produces: a fast-forwarded and pushed `main` whose exact SHA has a successful GitHub Actions run.

- [ ] **Step 1: Run local release verification**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/ -q
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m compileall -q app scripts tests
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python scripts/export_openapi_contract.py --check
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python /Users/mac/project/KBO_platform/scripts/validate_baseball_data_policy.py
git diff --check main...HEAD
```

Expected: 1974 existing tests plus the new tests pass with only documented skips;
compileall, OpenAPI consistency, policy validation, and diff checks exit zero.

- [ ] **Step 2: Review branch scope and main safety**

Run:

```bash
git status --short
git log --oneline main..HEAD
git diff --stat main...HEAD
git -C /Users/mac/project/KBO_platform/bega_AI rev-parse HEAD
git rev-parse main
```

Expected: the implementation worktree is clean, only planned commits/files are
present, and the original checkout still points to the branch base.

- [ ] **Step 3: Fast-forward main and push**

From the original checkout, run:

```bash
git merge --ff-only codex/ci-black-incremental-gate-20260723
git push origin main
```

Expected: main fast-forwards without touching unrelated working-tree files and
the remote accepts the new commits.

- [ ] **Step 4: Require the GitHub Actions result for the pushed SHA**

Run:

```bash
gh run list --repo rkdansdlf/bega_AI --commit "$(git rev-parse HEAD)" --limit 5
gh run watch --repo rkdansdlf/bega_AI <run-id> --exit-status
```

Expected: `AI Service CI Pipeline` completes successfully, including Python
Linting, Unit Tests, Security Scan, and Container Image Scan.

- [ ] **Step 5: Remove the completed worktree and local feature branch**

Run from the original checkout after successful remote verification:

```bash
git worktree remove .worktrees/ci-black-incremental-gate-20260723
git branch -d codex/ci-black-incremental-gate-20260723
git status --short
```

Expected: the temporary worktree and merged local branch are removed while the
user's unrelated dirty files remain exactly present.
