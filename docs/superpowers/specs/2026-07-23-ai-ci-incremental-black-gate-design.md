# Incremental Black Gate for AI CI

## Summary

The AI CI lint job currently runs `black --check .` over the entire repository.
The repository has pre-existing formatting debt, so an unrelated code change can
fail before unit tests and security jobs start. The workflow also installs
`black` without a version constraint even though `requirements-dev` already pins
`black==26.5.1`.

This change makes the Black gate incremental. CI checks only Python files added,
copied, modified, or renamed by the pushed commit range or pull request branch.
Existing formatting debt remains visible but no longer blocks unrelated work;
every newly changed Python file must conform to the pinned formatter.

## Goals

- Use the exact Black version pinned in `requirements-dev`.
- Check all changed `.py` files and no unchanged Python files.
- Use push range semantics for push events and merge-base semantics for pull
  requests.
- Handle a zero `before` SHA on the first push of a branch by comparing the head
  commit with Git's empty tree.
- Ignore deleted files and non-Python files.
- Preserve file names safely, including whitespace and newline characters.
- Fail loudly when a non-zero comparison ref is invalid or unavailable.
- Keep the existing whole-repository fatal Flake8 check unchanged.

## Non-goals

- Reformatting the repository's existing Black debt.
- Changing application behavior, dependencies, APIs, or baseball data paths.
- Weakening Flake8, unit tests, dependency auditing, or container scanning.
- Adding a third-party changed-files action.

## Architecture

Add `scripts/list_changed_python_files.py`, a standard-library-only CLI that
invokes Git and emits NUL-delimited repository-relative paths. Its interface is:

```text
python scripts/list_changed_python_files.py \
  --base <git-object-or-zero-sha> \
  --head <git-object> \
  --comparison range|merge-base
```

`range` builds `<base>..<head>` and is used for push events. `merge-base` builds
`<base>...<head>` and is used for pull requests. An all-zero base is replaced by
the repository's empty-tree object and always uses range comparison. Git runs
with `--diff-filter=ACMR`, the `*.py` pathspec, and `-z` output.

The lint checkout uses `fetch-depth: 0` so both event SHAs are available. Event
metadata selects the comparison without querying the network:

- Pull request: base is `github.event.pull_request.base.sha`, head is
  `github.event.pull_request.head.sha`, comparison is `merge-base`.
- Push: base is `github.event.before`, head is `github.sha`, comparison is
  `range`.

The workflow loads NUL-delimited paths into a Bash array. With no changed Python
files it records a successful no-op. Otherwise it runs `black --check --` with
the exact array elements. Lint dependencies are installed from
`requirements-dev`, which currently pins `black==26.5.1`.

## Testing

Unit tests create temporary Git repositories and exercise range comparison,
merge-base comparison, deleted and non-Python filtering, zero-SHA handling, and
NUL-safe file names. A workflow contract test verifies full-history checkout,
the pinned requirements install, event-to-comparison mapping, and removal of the
whole-repository Black command.

Before integration, run the focused tests, the incremental Black command for the
branch diff, the complete AI pytest suite, bytecode compilation, OpenAPI export
consistency, and the repository baseball-data policy validator. After a
fast-forward merge and push, require the GitHub Actions run for the pushed SHA to
finish successfully before declaring the task complete.

## Rollout and Rollback

This is an additive script plus a workflow-only policy change. Rollback reverts
the implementation commit and restores the previous whole-repository gate. No
database, API, runtime, or stored-data rollback is required.
