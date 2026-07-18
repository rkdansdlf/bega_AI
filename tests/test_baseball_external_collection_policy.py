from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCANNED_ROOTS = (
    PROJECT_ROOT / "app",
    PROJECT_ROOT / "scripts",
)
BASEBALL_PATH_HINTS = (
    "baseball",
    "kbo",
    "game",
    "lineup",
    "operator_data",
    "coach",
)
FORBIDDEN_FILENAME_MARKERS = (
    "crawler",
    "scraper",
    "scrape",
    "crawl",
)
FORBIDDEN_IMPORT_MARKERS = (
    "import selenium",
    "from selenium",
    "import bs4",
    "from bs4",
    "beautifulsoup",
    "import scrapy",
    "from scrapy",
)
FORBIDDEN_SECRET_MARKERS = (
    "134.185.107.178",
    "rkdansdlf",
)


def _is_baseball_path(path: Path) -> bool:
    normalized = path.relative_to(PROJECT_ROOT).as_posix().lower()
    return any(hint in normalized for hint in BASEBALL_PATH_HINTS)


def _is_text_file(path: Path) -> bool:
    return path.suffix in {
        ".py",
        ".sh",
        ".md",
        ".sql",
        ".yml",
        ".yaml",
        ".json",
        ".toml",
    }


def test_ai_baseball_paths_do_not_introduce_external_collection_code() -> None:
    violations: list[str] = []

    for root in SCANNED_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not _is_baseball_path(path):
                continue

            filename = path.name.lower()
            for marker in FORBIDDEN_FILENAME_MARKERS:
                if marker in filename:
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)} has forbidden filename marker `{marker}`"
                    )

            if not _is_text_file(path):
                continue
            content = path.read_text(encoding="utf-8").lower()
            for marker in FORBIDDEN_IMPORT_MARKERS:
                if marker in content:
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)} contains forbidden import marker `{marker}`"
                    )
            for marker in FORBIDDEN_SECRET_MARKERS:
                if marker in content:
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)} contains forbidden hard-coded DB marker `{marker}`"
                    )

    assert violations == []


def test_ai_runtime_scripts_do_not_hard_code_dev_database_credentials() -> None:
    violations: list[str] = []

    for root in SCANNED_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not _is_text_file(path):
                continue

            content = path.read_text(encoding="utf-8").lower()
            for marker in FORBIDDEN_SECRET_MARKERS:
                if marker in content:
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)} contains forbidden hard-coded DB marker `{marker}`"
                    )

    assert violations == []
