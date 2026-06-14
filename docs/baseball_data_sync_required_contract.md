# BASEBALL_DATA_SYNC_REQUIRED Contract

## Purpose

`BASEBALL_DATA_SYNC_REQUIRED` is the handoff contract for baseball data that the AI service can identify but must not collect by itself.

The AI service does not crawl, web search, synthesize, or directly enter missing baseball data. A separate trusted baseball data collection project owns collection and verification. This repo only emits structured missing-data requirements so that external project can sync verified rows back into the internal database.

`MANUAL_BASEBALL_DATA_REQUIRED` remains as a legacy compatibility marker for existing clients, smoke tests, and reports. New consumers should prefer the `dataSyncRequest` object when it is present.

## Coach Stream Meta

When Coach cannot safely analyze because game data is missing or stale, `meta.manual_data_request` keeps the existing shape and adds sync fields:

```json
{
  "code": "MANUAL_BASEBALL_DATA_REQUIRED",
  "dataSyncRequired": true,
  "dataSyncCode": "BASEBALL_DATA_SYNC_REQUIRED",
  "externalSource": "trusted_baseball_data_project",
  "dataSyncRequest": {
    "code": "BASEBALL_DATA_SYNC_REQUIRED",
    "requestId": "coach:20260405LGKT0:game_review",
    "consumer": "ai_coach",
    "scope": "coach.analyze",
    "analysisType": "game_review",
    "targetSource": "trusted_baseball_data_project",
    "handoff": "external_trusted_baseball_data_sync",
    "blocking": true,
    "entity": {
      "gameId": "20260405LGKT0",
      "gameDate": "2026-04-05",
      "seasonYear": 2026,
      "homeTeamId": "LG",
      "awayTeamId": "KT",
      "stage": "REGULAR"
    },
    "missingItems": [
      {
        "key": "final_score",
        "label": "최종 점수",
        "reason": "과거 경기의 최종 점수가 비어 있습니다.",
        "expectedFormat": "home_score, away_score",
        "requiredFields": ["game.home_score", "game.away_score"]
      }
    ]
  }
}
```

## Report Outputs

Coach backfill audit now writes both compatibility and sync-oriented CSVs:

- `coach_manual_baseball_data_required_*.csv`: legacy compatibility output.
- `coach_baseball_data_sync_required_*.csv`: external trusted sync handoff output.

Operator-data recovery gate now writes:

- `manual_baseball_data_required_rows.csv`: legacy compatibility output.
- `baseball_data_sync_required_rows.csv`: external trusted sync handoff output.

The sync CSVs include `data_sync_code`, `data_sync_request_id`, `external_source`, `handoff_target`, `missing_code`, and `required_fields`. External collection tooling should use those fields to decide what to collect and verify. It should not require this project to provide raw baseball values by direct input.

## Operational Rule

If a required row is missing, stale, unverified, or inconsistent:

1. Keep answer generation blocked or on the existing manual fallback path.
2. Emit `BASEBALL_DATA_SYNC_REQUIRED` details in stream meta or reports.
3. Wait for the external trusted baseball data project to sync verified data.
4. Re-run Coach/backfill/smoke after the trusted sync is reflected in the internal DB.

