# AI Model Routing Cost Gate Runbook

## Scope

This runbook records comparable, internal-only golden evidence for a proposed
planner model change. It does not change model routing, prompts, answer
behavior, cache-serving configuration, or baseball-data sources. Missing or
inconsistent baseball data remains on the `MANUAL_BASEBALL_DATA_REQUIRED` path.

A passing report is evidence only and does not authorize deployment. Deployment,
target-host release gates, and post-deploy observation require their own
approvals.

## Preconditions

Run the commands from `bega_AI` with the approved internal environment. Do not
print, commit, or include credentials in reports.

`CHAT_MODEL_PRICING_JSON` is optional for ordinary serving. The routing gate
requires exact provider/model catalog entries for every explicit planner and
answer model used by both runs. Its JSON values are decimal strings in USD per
one million tokens, keyed by provider and then exact model name. Do not use
aliases, `default`, or `unknown` labels.

Each configuration must supply explicit model names through
`CHAT_PLANNER_MODEL_NAME` and `CHAT_ANSWER_MODEL_NAME`. The answer model must
remain fixed between baseline and candidate. Only the planner model may be
evaluated as the candidate variable.

CLI planner/answer labels are evidence assertions only and do not configure the
server. They must match the active controlled AI service configuration for the
corresponding run.

## Golden path

The approved input is the immutable operator-provided
`scripts/chat_quality_golden_60.json` at SHA-256
`8899b468d65a87a505083a5f229808a0b9b405819287f00ff707741715bb124d`.
Confirm the path and hash before either run. The runner records the exact input
hash and ordered case IDs in each report; a mismatch is invalid evidence.

The release evidence defaults to 60 cases and requires the full approved
dataset. Cache bypass is required and defaults to enabled. Do not pass
`--no-cache-bypass`; it produces invalid release evidence.

Reports contain sanitized metadata, token counts, model labels, latency, and
cost totals. Reports must not contain answers, prompts, tool payloads,
credentials, headers, or user identifiers.

### Baseline

Before the baseline, configure the controlled AI service with
`CHAT_PLANNER_MODEL_NAME`, `CHAT_ANSWER_MODEL_NAME`, and the matching
`CHAT_MODEL_PRICING_JSON` catalog. Do not restart or redeploy the controlled AI
service on a target host for this experiment. The local procedure below starts
a fresh process on a dedicated local port. Its health check alone does not
prove active model configuration. Instead, lsof binds the current-shell child
PID to the dedicated port; the CLI labels are evidence matched later by the
reports.

## Controlled Local Procedure

Obtain separate live-call approval before running any part of this procedure,
including the local health check. This recreates a controlled local service
only; it is not production deployment approval. Run the entire block in one
shell from `bega_AI`. The operator must provide the shell variables before
starting it. Do not echo their values, commit them, or include them in reports.

```bash
set -u
set -a; source ../.env.prod; set +a

: "${BASELINE_PLANNER_MODEL:?operator-provided baseline planner model is required}"
: "${FIXED_ANSWER_MODEL:?operator-provided fixed answer model is required}"
: "${BASELINE_PRICING_JSON:?operator-provided baseline pricing JSON is required}"
: "${CANDIDATE_PLANNER_MODEL:?operator-provided candidate planner model is required}"
: "${CANDIDATE_PRICING_JSON:?operator-provided candidate pricing JSON is required}"

mkdir -p outputs/model-routing
CONTROLLED_AI_LOG=outputs/model-routing/controlled-ai.log
CONTROLLED_AI_PID=""
CONTROLLED_AI_PORT="${CONTROLLED_AI_PORT:-18001}"
export AI_MODEL_ROUTING_BASE_URL=http://127.0.0.1:$CONTROLLED_AI_PORT

stop_controlled_ai() {
  if [ -z "$CONTROLLED_AI_PID" ]; then
    return
  fi

  if kill -0 "$CONTROLLED_AI_PID" 2>/dev/null; then
    kill -TERM "$CONTROLLED_AI_PID" 2>/dev/null || true
    for _ in $(seq 1 10); do
      if ! kill -0 "$CONTROLLED_AI_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$CONTROLLED_AI_PID" 2>/dev/null; then
      kill -KILL "$CONTROLLED_AI_PID" 2>/dev/null || true
    fi
  fi

  wait "$CONTROLLED_AI_PID" 2>/dev/null || true
  CONTROLLED_AI_PID=""
}

start_controlled_ai() {
  if ! command -v lsof >/dev/null 2>&1; then
    printf '%s\n' 'lsof is required for controlled AI port ownership checks' >&2
    return 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    printf '%s\n' 'curl is required for controlled AI health checks' >&2
    return 1
  fi
  if lsof -nP -iTCP:"$CONTROLLED_AI_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    printf 'controlled AI port %s already has a listener\n' "$CONTROLLED_AI_PORT" >&2
    lsof -nP -iTCP:"$CONTROLLED_AI_PORT" -sTCP:LISTEN >&2
    return 1
  fi

  ./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "$CONTROLLED_AI_PORT" >"$CONTROLLED_AI_LOG" 2>&1 &
  CONTROLLED_AI_PID=$!

  for attempt in $(seq 1 30); do
    if ! kill -0 "$CONTROLLED_AI_PID" 2>/dev/null; then
      printf '%s\n' 'controlled AI failed to start; inspect outputs/model-routing/controlled-ai.log locally under access controls' >&2
      return 1
    fi
    listener_pid="$(lsof -t -a -p "$CONTROLLED_AI_PID" -iTCP:"$CONTROLLED_AI_PORT" -sTCP:LISTEN 2>/dev/null || true)"
    if [ "$listener_pid" = "$CONTROLLED_AI_PID" ] && \
      curl -fsS "$AI_MODEL_ROUTING_BASE_URL/health" >/dev/null; then
      return 0
    fi
    sleep 1
  done

  printf '%s\n' 'controlled AI failed to start; inspect outputs/model-routing/controlled-ai.log locally under access controls' >&2
  return 1
}

trap stop_controlled_ai EXIT

export CHAT_PLANNER_MODEL_NAME="$BASELINE_PLANNER_MODEL"
export CHAT_ANSWER_MODEL_NAME="$FIXED_ANSWER_MODEL"
export CHAT_MODEL_PRICING_JSON="$BASELINE_PRICING_JSON"
if ! start_controlled_ai; then
  exit 1
fi

set +e
AI_MODEL_ROUTING_SAMPLES=scripts/chat_quality_golden_60.json \
AI_MODEL_ROUTING_OUTPUT=outputs/model-routing/baseline.json \
./.venv/bin/python scripts/chat_model_routing_experiment.py \
  --planner-model-label "$CHAT_PLANNER_MODEL_NAME" \
  --answer-model-label "$CHAT_ANSWER_MODEL_NAME"
BASELINE_EXIT_CODE=$?

stop_controlled_ai
printf 'BASELINE_EXIT_CODE=%s\n' "$BASELINE_EXIT_CODE"
if [ "$BASELINE_EXIT_CODE" -eq 2 ]; then
  exit "$BASELINE_EXIT_CODE"
fi

# Change only the planner and matching pricing catalog; keep the answer fixed.
export CHAT_PLANNER_MODEL_NAME="$CANDIDATE_PLANNER_MODEL"
export CHAT_ANSWER_MODEL_NAME="$FIXED_ANSWER_MODEL"
export CHAT_MODEL_PRICING_JSON="$CANDIDATE_PRICING_JSON"
if ! start_controlled_ai; then
  exit 1
fi

set +e
AI_MODEL_ROUTING_SAMPLES=scripts/chat_quality_golden_60.json \
AI_MODEL_ROUTING_OUTPUT=outputs/model-routing/candidate.json \
./.venv/bin/python scripts/chat_model_routing_experiment.py \
  --baseline-report outputs/model-routing/baseline.json \
  --planner-model-label "$CHAT_PLANNER_MODEL_NAME" \
  --answer-model-label "$CHAT_ANSWER_MODEL_NAME"
CANDIDATE_EXIT_CODE=$?

stop_controlled_ai
printf 'CANDIDATE_EXIT_CODE=%s\n' "$CANDIDATE_EXIT_CODE"
trap - EXIT
exit "$CANDIDATE_EXIT_CODE"
```

Preserve `outputs/model-routing/baseline.json` as the comparison input. Do not
edit it after generation. Before every start, the procedure refuses a port with
an existing listener and never reads or kills a stale PID file. The startup loop
requires both a live child PID and an `lsof` listener match for that child on
the dedicated port before calling the health endpoint. A child exit or bind
failure prints only the fixed diagnostic path and instruction to inspect locally
under access controls, then fails the procedure. The cleanup trap
stops only the child spawned by the current shell, with bounded TERM polling,
KILL fallback, and `wait`; do not substitute a broad process kill.

### Candidate

Before the candidate, change only `CHAT_PLANNER_MODEL_NAME` and its catalog
entry as needed; keep `CHAT_ANSWER_MODEL_NAME` fixed. The candidate portion of
the controlled local procedure exports that same fixed answer value, starts a
fresh local process, verifies its PID-to-port listener binding, runs the exact
candidate command, and stops its current-shell child. The CLI planner/answer
labels remain evidence assertions; they do not apply this configuration.

### Run Report

After cleanup, append a sanitized operational report with the live-call approval
reference, golden input hash, baseline and candidate report paths, and the
numeric `BASELINE_EXIT_CODE` and `CANDIDATE_EXIT_CODE` values for every runner
that was attempted. A baseline exit code of `2` is invalid evidence and aborts
before the candidate; a baseline exit code of `1` may continue to the candidate.
The candidate status is the procedure exit code after cleanup. Do not include
`.env.prod` contents, shell-variable values, credentials, headers, prompts,
answers, or service logs in that report.

Do not execute the baseline or candidate commands without separate live-call
approval. They issue model requests even though the input data is internal and
the reports are sanitized.

## Decision Rules

The candidate gate passes only when the reports have the same immutable golden
hash and ordered 60 case IDs, all observed calls are captured and catalog-priced,
and the candidate introduces no quality or routing regression. The answer model
must remain fixed, a planner reduction of at least 20% is required, and
candidate total model cost must not increase relative to baseline.

The command exit codes are stable:

- `0: valid evidence and all gates pass`
- `1: valid evidence but a quality or cost criterion fails`
- `2: invalid or incomplete evidence, configuration, input, or report`

Exit `2` includes absent or unpriced model usage where an LLM planner or answer
call is required, implicit labels, disabled cache bypass, missing/duplicate
cases, a dataset/hash mismatch, or an answer-model mismatch that prevents
comparison. Absent usage is invalid only when the selected planner or answer
path requires an LLM call. A deterministic mode may validly make no model call
and therefore have empty usage. Exit `1` is measured evidence that fails the
quality or cost rules. A zero exit code is not a deployment authorization.

## Approval And CI Boundaries

The live baseline/candidate procedure requires separate live-call approval.
CI runs only deterministic unit and contract tests; it does not execute this
golden runner against a service, invoke a provider, or use model credentials.
Actual deployment remains separately approved; this controlled experiment does
not authorize a production deployment.

Do not use this gate to repair baseball data, search for baseball data, scrape
external sites, or call external baseball APIs. Obtain operator-provided manual
data when the internal database cannot satisfy a case.
