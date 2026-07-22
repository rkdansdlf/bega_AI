from pathlib import Path
import subprocess
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "list_changed_python_files.py"
)


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


def test_merge_base_excludes_changes_made_only_on_base_branch(
    tmp_path: Path,
) -> None:
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
    head = commit_files(
        repository,
        {"first.py": "x = 1\n", "README.md": "read\n"},
    )

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
