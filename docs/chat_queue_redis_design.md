# Chat Queue Redis Design

## Goal

Provide a shared admission layer for `POST /ai/chat/stream` when the AI service
runs more than one worker, process, or container. The current in-memory queue is
valid only for a single process and must not be treated as a global FIFO queue.

## Non-Goals

- Do not change baseball data sourcing or add any external baseball data lookup.
- Do not replace response generation, cache storage, or OpenRouter fallback logic.
- Do not implement this design until the deployment target actually requires
  multi-worker global admission.

## Proposed Keys

- `ai:chat:queue:waiting`: Redis list of reservation IDs in FIFO order.
- `ai:chat:queue:started`: Redis sorted set of admitted reservation IDs scored by
  admission timestamp.
- `ai:chat:queue:reservation:{id}`: reservation metadata with state, created time,
  heartbeat time, and optional request fingerprint.
- `ai:chat:queue:seq`: monotonically increasing reservation ID counter.

## Admission Flow

1. Generate `reservation_id` with `INCR ai:chat:queue:seq`.
2. Remove expired entries from `ai:chat:queue:started`.
3. If `started` cardinality is below `CHAT_QUEUE_RPM_LIMIT` and `waiting` is
   empty, admit immediately and add the reservation to `started`.
4. Otherwise append the reservation to `waiting` if queue size and estimated wait
   are inside configured bounds.
5. If queue bounds are exceeded, return the existing 429 contract with
   `Retry-After`.

Use a Lua script or Redis transaction for steps 2-5 so admission and queue length
checks are atomic.

## Status Polling

Status checks should:

- prune expired `started` entries;
- promote FIFO waiters while capacity is available;
- return `processing` for admitted reservations;
- return `queued` with 1-based position and estimated wait for waiting
  reservations;
- return a terminal state such as `cancelled` for missing reservations.

## Cleanup

- Client disconnect before processing: remove reservation from `waiting`, delete
  reservation metadata, and emit the existing cancelled metric.
- Normal stream completion: remove reservation from `started` and delete metadata.
- Worker crash: reservation metadata must have a short TTL refreshed by heartbeat.
  A cleanup job or status/admission script should remove stale reservations from
  both `waiting` and `started`.

## Observability

Keep the existing metric names where possible:

- `ai_chat_queue_depth{state="waiting"}`
- `ai_chat_queue_depth{state="admitted"}`
- `ai_chat_queue_events_total{event="queued|admitted|admitted_from_queue|overflow|cancelled|released"}`
- `ai_chat_queue_estimated_wait_seconds`

For multi-worker deployments, publish depth from Redis rather than process-local
memory so dashboards do not double-count per instance.

## Rollout

1. Keep `CHAT_QUEUE_ENABLED=true` with the current in-memory queue for single
   worker deployments.
2. Add a separate flag, for example `CHAT_QUEUE_BACKEND=memory|redis`.
3. Roll out Redis backend dark in staging with focused gate tests plus a
   multi-worker synthetic queue test.
4. Enable Redis backend only after verifying global FIFO, disconnect cleanup,
   crash cleanup, and 429 `Retry-After` behavior.
