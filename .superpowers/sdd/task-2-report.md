# Task 2 Report: Attempt-Level Provider Observations

## RED

Command:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_llm_model_candidates.py::test_openrouter_generator_uses_model_override_without_fallback tests/test_llm_model_candidates.py::test_openrouter_429_falls_back_to_gpt_oss tests/test_llm_model_candidates.py::test_openrouter_empty_choices_retry_reports_each_attempt -q
```

Output:

```text
FFF                                                                      [100%]
3 failed in 0.80s
```

The failures were the expected `TypeError` for the unsupported `usage_observer` keyword.

## GREEN

Command:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_llm_model_candidates.py -q
```

Output:

```text
.......                                                                  [100%]
7 passed, 1 warning in 1.23s
```

The warning is the installed `google.generativeai` SDK deprecation warning.

## Files

- `app/agents/runtime_factory.py`: added optional `usage_observer` callbacks for OpenRouter model/retry attempts and Gemini generation attempts.
- `tests/test_llm_model_candidates.py`: added OpenRouter success, empty-retry, and fallback observations plus a fake Gemini observation test.

## Commit

`8160dbf feat(ai): observe routed model attempts`

## Self-review

- `git diff --check` passed before commit.
- Provider/model selection, yielded chunks, retry and fallback metrics, and provider exception re-raising remain unchanged.
- Observer payloads contain only the requested provider, model, messages, output text, and outcome fields.
- No external network/model calls, baseball data sources, secrets, or package installs were used.

## Concerns

- The focused suite was run; the full AI-service test suite was not run.
- The Gemini test emits the existing SDK deprecation warning.
- The pre-existing untracked `.superpowers/` directory remains untouched except for this requested report.

## Review fixes

### Files

- `app/agents/runtime_factory.py`: made usage observers best-effort, deep-copied observer messages, and recorded failed observations during unrecorded OpenRouter and Gemini generator finalization.
- `tests/test_llm_model_candidates.py`: added regression coverage for observer exceptions, observer message mutation, early close for both providers, and Gemini execution with a fully mocked SDK import and no warnings.

### Commit

`abf80af fix(ai): harden model usage observers`

### Command and Output

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_llm_model_candidates.py -q
```

```text
..........                                                               [100%]
10 passed in 0.25s
```

### Self-review

- Callback exceptions and snapshot failures are logged but cannot change provider output, fallback progression, or provider exceptions.
- Each started attempt records once; the generator `finally` path reports `failed` only when success or failure has not already been observed.
- Observer message payloads are deep-copied before invocation, so callback mutation cannot alter later provider attempts.
- `git diff --check` passed before commit. No network/model calls, external baseball sources, or package installs were used.

### Remaining Task 2 review gaps

#### Files

- `app/agents/runtime_factory.py`: isolated observer callbacks and defensive snapshots from provider control flow, including `BaseException` subclasses; snapshot failure now delivers a fresh empty-list snapshot to the observer.
- `tests/test_llm_model_candidates.py`: added focused OpenRouter regressions for a cancelling observer and a failing snapshot, each asserting preserved output and exactly one observer call.

#### Commit

`4bf4ba1 fix(ai): harden observer isolation`

#### Command and Output

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_llm_model_candidates.py::test_openrouter_cancelled_error_observer_does_not_change_output tests/test_llm_model_candidates.py::test_openrouter_snapshot_failure_still_observes_once -q
```

```text
..                                                                       [100%]
2 passed in 0.22s
```

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_llm_model_candidates.py -q
```

```text
............                                                             [100%]
12 passed in 0.20s
```

#### Self-review

- Observer callbacks and snapshot creation now catch `BaseException`, so cancellation and other observer failures cannot change output, retries, fallback, or provider exceptions.
- A snapshot failure supplies a new empty list and still invokes the observer once; successful helper return lets the existing attempt guard prevent finalization duplicates.
- The focused suite and complete module ran without warnings. No network/model calls, external baseball sources, package installs, or unrelated worktree changes were used.
