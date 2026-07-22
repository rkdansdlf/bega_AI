"""List changed Python files for CI as NUL-delimited repository paths."""

from __future__ import annotations

import argparse
import subprocess
import sys

ZERO_SHA = "0" * 40


def git_output(arguments: list[str], *, stdin: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        input=stdin,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout


def empty_tree() -> str:
    return (
        git_output(
            ["hash-object", "-t", "tree", "--stdin"],
            stdin=b"",
        )
        .decode()
        .strip()
    )


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
        "--comparison",
        required=True,
        choices=("range", "merge-base"),
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
