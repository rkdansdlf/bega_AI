# KBO AI Service Schemas

> This file is generated. Do not edit directly.
> Source: `contracts/openapi.json`
> Regenerate with: `python scripts/export_openapi_contract.py`

Version: `0.1.0`
Schemas: **37**

<a id="body-analyze-ticket-image-ai-vision-ticket-post"></a>
## Body_analyze_ticket_image_ai_vision_ticket_post
- Type: `object`
- Required properties: `file`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `file` | yes | `string` |  |  |

Unsupported property file metadata:

```json
{
  "contentMediaType": "application/octet-stream",
  "title": "File"
}
```

<a id="body-analyze-ticket-image-vision-ticket-post"></a>
## Body_analyze_ticket_image_vision_ticket_post
- Type: `object`
- Required properties: `file`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `file` | yes | `string` |  |  |

Unsupported property file metadata:

```json
{
  "contentMediaType": "application/octet-stream",
  "title": "File"
}
```

<a id="body-classify-seat-view-image-ai-vision-seat-view-classify-post"></a>
## Body_classify_seat_view_image_ai_vision_seat_view_classify_post
- Type: `object`
- Required properties: `file`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `file` | yes | `string` |  |  |

Unsupported property file metadata:

```json
{
  "contentMediaType": "application/octet-stream",
  "title": "File"
}
```

<a id="body-classify-seat-view-image-vision-seat-view-classify-post"></a>
## Body_classify_seat_view_image_vision_seat_view_classify_post
- Type: `object`
- Required properties: `file`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `file` | yes | `string` |  |  |

Unsupported property file metadata:

```json
{
  "contentMediaType": "application/octet-stream",
  "title": "File"
}
```

<a id="body-transcribe-audio-ai-chat-voice-post"></a>
## Body_transcribe_audio_ai_chat_voice_post
- Type: `object`
- Required properties: `file`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `file` | yes | `string` |  |  |

Unsupported property file metadata:

```json
{
  "contentMediaType": "application/octet-stream",
  "title": "File"
}
```

<a id="coachanalyzerequest"></a>
## CoachAnalyzeRequest

POST `/ai/coach/analyze` body, including existing compatibility aliases.
- Type: `object`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `analysis_type` | no | `{   "anyOf": [     {       "enum": [         "game_review",         "game_preview"       ],       "type": "string"     },     {       "type": "null"     }   ],   "title": "Analysis Type" }` |  |  |

Unsupported property analysis_type metadata:

```json
{
  "anyOf": [
    {
      "enum": [
        "game_review",
        "game_preview"
      ],
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Analysis Type"
}
```
| `away_team_id` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Away Team Id" }` |  |  |

Unsupported property away_team_id metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Away Team Id"
}
```
| `expected_cache_key` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Expected Cache Key" }` |  |  |

Unsupported property expected_cache_key metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Expected Cache Key"
}
```
| `focus` | no | `array` |  |  |
- `focus`: Items: `string`

Unsupported property focus metadata:

```json
{
  "title": "Focus"
}
```
| `game_id` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Game Id" }` |  |  |

Unsupported property game_id metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Game Id"
}
```
| `home_team_id` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Home Team Id" }` |  |  |

Unsupported property home_team_id metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Home Team Id"
}
```
| `league_context` | no | `{   "anyOf": [     {       "additionalProperties": {         "$ref": "#/components/schemas/JsonValue"       },       "type": "object"     },     {       "type": "null"     }   ],   "title": "League Context" }` |  |  |

Unsupported property league_context metadata:

```json
{
  "anyOf": [
    {
      "additionalProperties": {
        "$ref": "#/components/schemas/JsonValue"
      },
      "type": "object"
    },
    {
      "type": "null"
    }
  ],
  "title": "League Context"
}
```
| `lineup_signature` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Lineup Signature" }` |  |  |

Unsupported property lineup_signature metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Lineup Signature"
}
```
| `question_override` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Question Override" }` |  |  |

Unsupported property question_override metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Question Override"
}
```
| `request_mode` | no | `string` |  |  |
- `request_mode`: Default: `"manual_detail"`; Enum: `auto_brief`, `manual_detail`

Unsupported property request_mode metadata:

```json
{
  "title": "Request Mode"
}
```
| `starter_signature` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Starter Signature" }` |  |  |

Unsupported property starter_signature metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Starter Signature"
}
```
| `team_id` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Team Id" }` |  |  |

Unsupported property team_id metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Team Id"
}
```

<a id="coachautobriefopsgate"></a>
## CoachAutoBriefOpsGate
- Type: `object`
- Required properties: `failed_locked_count`, `insufficient_count`, `insufficient_ratio`, `pending_wait_count`, `thresholds`, `verdict`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `checks` | no | [CoachAutoBriefOpsGateChecks](api-schemas.md#coachautobriefopsgatechecks) |  |  |
| `failed_locked_count` | yes | `integer` |  |  |

Unsupported property failed_locked_count metadata:

```json
{
  "title": "Failed Locked Count"
}
```
| `insufficient_count` | yes | `integer` |  |  |

Unsupported property insufficient_count metadata:

```json
{
  "title": "Insufficient Count"
}
```
| `insufficient_ratio` | yes | `number` |  |  |

Unsupported property insufficient_ratio metadata:

```json
{
  "title": "Insufficient Ratio"
}
```
| `pending_wait_count` | yes | `integer` |  |  |

Unsupported property pending_wait_count metadata:

```json
{
  "title": "Pending Wait Count"
}
```
| `thresholds` | yes | [CoachAutoBriefOpsGateThresholds](api-schemas.md#coachautobriefopsgatethresholds) |  |  |
| `verdict` | yes | `string` |  |  |
- `verdict`: Enum: `PASS`, `WARN`, `FAIL`

Unsupported property verdict metadata:

```json
{
  "title": "Verdict"
}
```

<a id="coachautobriefopsgatechecks"></a>
## CoachAutoBriefOpsGateChecks
- Type: `object`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `failed` | no | `array` |  |  |
- `failed`: Items: `string`

Unsupported property failed metadata:

```json
{
  "title": "Failed"
}
```
| `warnings` | no | `array` |  |  |
- `warnings`: Items: `string`

Unsupported property warnings metadata:

```json
{
  "title": "Warnings"
}
```

<a id="coachautobriefopsgatethresholds"></a>
## CoachAutoBriefOpsGateThresholds
- Type: `object`
- Required properties: `fail_on_missing_report`, `max_failed_locked`, `max_unresolved`, `min_selected_targets`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `fail_on_missing_report` | yes | `boolean` |  |  |

Unsupported property fail_on_missing_report metadata:

```json
{
  "title": "Fail On Missing Report"
}
```
| `max_failed_locked` | yes | `integer` |  |  |

Unsupported property max_failed_locked metadata:

```json
{
  "title": "Max Failed Locked"
}
```
| `max_insufficient_ratio` | no | `{   "anyOf": [     {       "type": "number"     },     {       "type": "null"     }   ],   "title": "Max Insufficient Ratio" }` |  |  |

Unsupported property max_insufficient_ratio metadata:

```json
{
  "anyOf": [
    {
      "type": "number"
    },
    {
      "type": "null"
    }
  ],
  "title": "Max Insufficient Ratio"
}
```
| `max_pending_wait` | no | `{   "anyOf": [     {       "type": "integer"     },     {       "type": "null"     }   ],   "title": "Max Pending Wait" }` |  |  |

Unsupported property max_pending_wait metadata:

```json
{
  "anyOf": [
    {
      "type": "integer"
    },
    {
      "type": "null"
    }
  ],
  "title": "Max Pending Wait"
}
```
| `max_unresolved` | yes | `integer` |  |  |

Unsupported property max_unresolved metadata:

```json
{
  "title": "Max Unresolved"
}
```
| `min_selected_targets` | yes | `integer` |  |  |

Unsupported property min_selected_targets metadata:

```json
{
  "title": "Min Selected Targets"
}
```

<a id="coachautobriefopshealthresponse"></a>
## CoachAutoBriefOpsHealthResponse
- Type: `object`
- Required properties: `date_window`, `gate`, `generated_at_utc`, `recommended_command`, `runbook_path`, `summary`, `window`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `date_window` | yes | `string` |  |  |

Unsupported property date_window metadata:

```json
{
  "title": "Date Window"
}
```
| `gate` | yes | [CoachAutoBriefOpsGate](api-schemas.md#coachautobriefopsgate) |  |  |
| `generated_at_utc` | yes | `string (date-time)` |  |  |

Unsupported property generated_at_utc metadata:

```json
{
  "title": "Generated At Utc"
}
```
| `latest_report` | no | `{   "anyOf": [     {       "$ref": "#/components/schemas/CoachAutoBriefOpsLatestReport"     },     {       "type": "null"     }   ] }` |  |  |

Unsupported property latest_report metadata:

```json
{
  "anyOf": [
    {
      "$ref": "#/components/schemas/CoachAutoBriefOpsLatestReport"
    },
    {
      "type": "null"
    }
  ]
}
```
| `recommended_command` | yes | `string` |  |  |

Unsupported property recommended_command metadata:

```json
{
  "title": "Recommended Command"
}
```
| `runbook_path` | yes | `string` |  |  |

Unsupported property runbook_path metadata:

```json
{
  "title": "Runbook Path"
}
```
| `summary` | yes | [CoachAutoBriefOpsSummary](api-schemas.md#coachautobriefopssummary) |  |  |
| `unresolved_targets` | no | `array` |  |  |
- `unresolved_targets`: Items: [CoachAutoBriefOpsTargetSample](api-schemas.md#coachautobriefopstargetsample)

Unsupported property unresolved_targets metadata:

```json
{
  "title": "Unresolved Targets"
}
```
| `window` | yes | `string` |  |  |
- `window`: Enum: `today`, `tomorrow`, `custom`

Unsupported property window metadata:

```json
{
  "title": "Window"
}
```

<a id="coachautobriefopslatestreport"></a>
## CoachAutoBriefOpsLatestReport
- Type: `object`
- Required properties: `path`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `cache_state_breakdown` | no | `object` |  |  |

Unsupported property cache_state_breakdown metadata:

```json
{
  "additionalProperties": {
    "type": "integer"
  },
  "title": "Cache State Breakdown"
}
```
| `completed_count` | no | `integer` |  |  |
- `completed_count`: Default: `0`

Unsupported property completed_count metadata:

```json
{
  "title": "Completed Count"
}
```
| `data_quality_breakdown` | no | `object` |  |  |

Unsupported property data_quality_breakdown metadata:

```json
{
  "additionalProperties": {
    "type": "integer"
  },
  "title": "Data Quality Breakdown"
}
```
| `date_window` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Date Window" }` |  |  |

Unsupported property date_window metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Date Window"
}
```
| `path` | yes | `string` |  |  |

Unsupported property path metadata:

```json
{
  "title": "Path"
}
```
| `run_finished_at` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Run Finished At" }` |  |  |

Unsupported property run_finished_at metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Run Finished At"
}
```
| `run_started_at` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Run Started At" }` |  |  |

Unsupported property run_started_at metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Run Started At"
}
```
| `unresolved_count` | no | `integer` |  |  |
- `unresolved_count`: Default: `0`

Unsupported property unresolved_count metadata:

```json
{
  "title": "Unresolved Count"
}
```

<a id="coachautobriefopssummary"></a>
## CoachAutoBriefOpsSummary
- Type: `object`
- Required properties: `cache_hit_count`, `completed_count`, `failed_count`, `generated_success_count`, `in_progress_count`, `loaded_target_count`, `selected_target_count`, `unresolved_count`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `cache_hit_count` | yes | `integer` |  |  |

Unsupported property cache_hit_count metadata:

```json
{
  "title": "Cache Hit Count"
}
```
| `cache_state_breakdown` | no | `object` |  |  |

Unsupported property cache_state_breakdown metadata:

```json
{
  "additionalProperties": {
    "type": "integer"
  },
  "title": "Cache State Breakdown"
}
```
| `completed_count` | yes | `integer` |  |  |

Unsupported property completed_count metadata:

```json
{
  "title": "Completed Count"
}
```
| `data_quality_breakdown` | no | `object` |  |  |

Unsupported property data_quality_breakdown metadata:

```json
{
  "additionalProperties": {
    "type": "integer"
  },
  "title": "Data Quality Breakdown"
}
```
| `failed_count` | yes | `integer` |  |  |

Unsupported property failed_count metadata:

```json
{
  "title": "Failed Count"
}
```
| `generated_success_count` | yes | `integer` |  |  |

Unsupported property generated_success_count metadata:

```json
{
  "title": "Generated Success Count"
}
```
| `in_progress_count` | yes | `integer` |  |  |

Unsupported property in_progress_count metadata:

```json
{
  "title": "In Progress Count"
}
```
| `loaded_target_count` | yes | `integer` |  |  |

Unsupported property loaded_target_count metadata:

```json
{
  "title": "Loaded Target Count"
}
```
| `selected_target_count` | yes | `integer` |  |  |

Unsupported property selected_target_count metadata:

```json
{
  "title": "Selected Target Count"
}
```
| `unresolved_count` | yes | `integer` |  |  |

Unsupported property unresolved_count metadata:

```json
{
  "title": "Unresolved Count"
}
```

<a id="coachautobriefopstargetsample"></a>
## CoachAutoBriefOpsTargetSample
- Type: `object`
- Required properties: `away_team_id`, `cache_key`, `cache_state`, `data_quality`, `game_date`, `game_id`, `game_status_bucket`, `home_team_id`, `stage_label`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `away_team_id` | yes | `string` |  |  |

Unsupported property away_team_id metadata:

```json
{
  "title": "Away Team Id"
}
```
| `cache_key` | yes | `string` |  |  |

Unsupported property cache_key metadata:

```json
{
  "title": "Cache Key"
}
```
| `cache_state` | yes | `string` |  |  |

Unsupported property cache_state metadata:

```json
{
  "title": "Cache State"
}
```
| `data_quality` | yes | `string` |  |  |

Unsupported property data_quality metadata:

```json
{
  "title": "Data Quality"
}
```
| `game_date` | yes | `string` |  |  |

Unsupported property game_date metadata:

```json
{
  "title": "Game Date"
}
```
| `game_id` | yes | `string` |  |  |

Unsupported property game_id metadata:

```json
{
  "title": "Game Id"
}
```
| `game_status_bucket` | yes | `string` |  |  |

Unsupported property game_status_bucket metadata:

```json
{
  "title": "Game Status Bucket"
}
```
| `headline` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Headline" }` |  |  |

Unsupported property headline metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Headline"
}
```
| `home_team_id` | yes | `string` |  |  |

Unsupported property home_team_id metadata:

```json
{
  "title": "Home Team Id"
}
```
| `reason` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Reason" }` |  |  |

Unsupported property reason metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Reason"
}
```
| `stage_label` | yes | `string` |  |  |

Unsupported property stage_label metadata:

```json
{
  "title": "Stage Label"
}
```

<a id="coachcacheresetrequest"></a>
## CoachCacheResetRequest

``POST /coach/cache/reset`` 입력. cache_key 단건 또는 team_id+year 일괄.
- Type: `object`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `cache_key` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Cache Key" }` |  |  |

Unsupported property cache_key metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Cache Key"
}
```
| `include_stale_pending` | no | `boolean` |  |  |
- `include_stale_pending`: Default: `true`

Unsupported property include_stale_pending metadata:

```json
{
  "title": "Include Stale Pending"
}
```
| `retryable_only` | no | `boolean` |  |  |
- `retryable_only`: Default: `false`

Unsupported property retryable_only metadata:

```json
{
  "title": "Retryable Only"
}
```
| `team_id` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Team Id" }` |  |  |

Unsupported property team_id metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Team Id"
}
```
| `year` | no | `{   "anyOf": [     {       "type": "integer"     },     {       "type": "null"     }   ],   "title": "Year" }` |  |  |

Unsupported property year metadata:

```json
{
  "anyOf": [
    {
      "type": "integer"
    },
    {
      "type": "null"
    }
  ],
  "title": "Year"
}
```

<a id="evidenceitem"></a>
## EvidenceItem
- Type: `object`
- Required properties: `claim`, `excerpt`, `source`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `claim` | yes | `string` |  |  |

Unsupported property claim metadata:

```json
{
  "title": "Claim"
}
```
| `excerpt` | yes | `string` |  |  |

Unsupported property excerpt metadata:

```json
{
  "title": "Excerpt"
}
```
| `source` | yes | `string` |  |  |

Unsupported property source metadata:

```json
{
  "title": "Source"
}
```

<a id="httpvalidationerror"></a>
## HTTPValidationError
- Type: `object`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `detail` | no | `array` |  |  |
- `detail`: Items: [ValidationError](api-schemas.md#validationerror)

Unsupported property detail metadata:

```json
{
  "title": "Detail"
}
```

<a id="ingestpayload"></a>
## IngestPayload
- Type: `object`
- Required properties: `content`, `source_row_id`, `source_table`, `title`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `content` | yes | `string` |  |  |

Unsupported property content metadata:

```json
{
  "title": "Content"
}
```
| `player_id` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Player Id" }` |  |  |

Unsupported property player_id metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Player Id"
}
```
| `season_year` | no | `{   "anyOf": [     {       "type": "integer"     },     {       "type": "null"     }   ],   "title": "Season Year" }` |  |  |

Unsupported property season_year metadata:

```json
{
  "anyOf": [
    {
      "type": "integer"
    },
    {
      "type": "null"
    }
  ],
  "title": "Season Year"
}
```
| `source_row_id` | yes | `string` |  |  |

Unsupported property source_row_id metadata:

```json
{
  "title": "Source Row Id"
}
```
| `source_table` | yes | `string` |  |  |

Unsupported property source_table metadata:

```json
{
  "title": "Source Table"
}
```
| `source_type` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Source Type" }` |  |  |

Unsupported property source_type metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Source Type"
}
```
| `source_uri` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Source Uri" }` |  |  |

Unsupported property source_uri metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Source Uri"
}
```
| `team_id` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Team Id" }` |  |  |

Unsupported property team_id metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Team Id"
}
```
| `title` | yes | `string` |  |  |

Unsupported property title metadata:

```json
{
  "title": "Title"
}
```

<a id="jsonvalue"></a>
## JsonValue
- Type: `{}`

<a id="moderationresult"></a>
## ModerationResult
- Type: `object`
- Required properties: `action`, `category`, `decisionSource`, `reason`, `riskLevel`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `action` | yes | `string` |  |  |
- `action`: Enum: `ALLOW`, `BLOCK`

Unsupported property action metadata:

```json
{
  "title": "Action"
}
```
| `category` | yes | `string` |  |  |

Unsupported property category metadata:

```json
{
  "title": "Category"
}
```
| `decisionSource` | yes | `string` |  |  |
- `decisionSource`: Enum: `RULE`, `MODEL`, `FALLBACK`

Unsupported property decisionSource metadata:

```json
{
  "title": "Decisionsource"
}
```
| `reason` | yes | `string` |  |  |

Unsupported property reason metadata:

```json
{
  "title": "Reason"
}
```
| `riskLevel` | yes | `string` |  |  |
- `riskLevel`: Enum: `LOW`, `MEDIUM`, `HIGH`

Unsupported property riskLevel metadata:

```json
{
  "title": "Risklevel"
}
```

<a id="releasedecisionartifactrecord"></a>
## ReleaseDecisionArtifactRecord
- Type: `object`
- Required properties: `artifact_id`, `draft_response`, `markdown`, `saved_at_utc`, `scenario`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `allowed_roots` | no | `array` |  |  |
- `allowed_roots`: Items: `string`

Unsupported property allowed_roots metadata:

```json
{
  "title": "Allowed Roots"
}
```
| `artifact_id` | yes | `string` |  |  |

Unsupported property artifact_id metadata:

```json
{
  "title": "Artifact Id"
}
```
| `draft_response` | yes | [ReleaseDecisionRunResult](api-schemas.md#releasedecisionrunresult) |  |  |
| `evaluation` | no | `{   "anyOf": [     {       "$ref": "#/components/schemas/ReleaseDecisionEvaluateResponse"     },     {       "type": "null"     }   ] }` |  |  |

Unsupported property evaluation metadata:

```json
{
  "anyOf": [
    {
      "$ref": "#/components/schemas/ReleaseDecisionEvaluateResponse"
    },
    {
      "type": "null"
    }
  ]
}
```
| `markdown` | yes | `string` |  |  |

Unsupported property markdown metadata:

```json
{
  "title": "Markdown"
}
```
| `saved_at_utc` | yes | `string` |  |  |

Unsupported property saved_at_utc metadata:

```json
{
  "title": "Saved At Utc"
}
```
| `scenario` | yes | `string` |  |  |

Unsupported property scenario metadata:

```json
{
  "title": "Scenario"
}
```
| `seed_paths` | no | `array` |  |  |
- `seed_paths`: Items: `string`

Unsupported property seed_paths metadata:

```json
{
  "title": "Seed Paths"
}
```
| `task_prompt` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Task Prompt" }` |  |  |

Unsupported property task_prompt metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Task Prompt"
}
```

<a id="releasedecisionartifactsummary"></a>
## ReleaseDecisionArtifactSummary
- Type: `object`
- Required properties: `artifact_id`, `decision`, `json_filename`, `markdown_filename`, `saved_at_utc`, `scenario`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `artifact_id` | yes | `string` |  |  |

Unsupported property artifact_id metadata:

```json
{
  "title": "Artifact Id"
}
```
| `decision` | yes | `string` |  |  |
- `decision`: Enum: `GO`, `NO_GO`, `PENDING`

Unsupported property decision metadata:

```json
{
  "title": "Decision"
}
```
| `eval_status` | no | `{   "anyOf": [     {       "enum": [         "PASS",         "FAIL"       ],       "type": "string"     },     {       "type": "null"     }   ],   "title": "Eval Status" }` |  |  |

Unsupported property eval_status metadata:

```json
{
  "anyOf": [
    {
      "enum": [
        "PASS",
        "FAIL"
      ],
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Eval Status"
}
```
| `json_filename` | yes | `string` |  |  |

Unsupported property json_filename metadata:

```json
{
  "title": "Json Filename"
}
```
| `markdown_filename` | yes | `string` |  |  |

Unsupported property markdown_filename metadata:

```json
{
  "title": "Markdown Filename"
}
```
| `saved_at_utc` | yes | `string` |  |  |

Unsupported property saved_at_utc metadata:

```json
{
  "title": "Saved At Utc"
}
```
| `scenario` | yes | `string` |  |  |

Unsupported property scenario metadata:

```json
{
  "title": "Scenario"
}
```

<a id="releasedecisiondraft"></a>
## ReleaseDecisionDraft
- Type: `object`
- Required properties: `confidence`, `decision`, `summary`, `title`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `blockers` | no | `array` |  |  |
- `blockers`: Items: `string`

Unsupported property blockers metadata:

```json
{
  "title": "Blockers"
}
```
| `confidence` | yes | `string` |  |  |
- `confidence`: Enum: `low`, `medium`, `high`

Unsupported property confidence metadata:

```json
{
  "title": "Confidence"
}
```
| `decision` | yes | `string` |  |  |
- `decision`: Enum: `GO`, `NO_GO`, `PENDING`

Unsupported property decision metadata:

```json
{
  "title": "Decision"
}
```
| `evidence` | no | `array` |  | minItems=2 |
- `evidence`: Items: [EvidenceItem](api-schemas.md#evidenceitem)

Unsupported property evidence metadata:

```json
{
  "title": "Evidence"
}
```
| `next_actions` | no | `array` |  |  |
- `next_actions`: Items: `string`

Unsupported property next_actions metadata:

```json
{
  "title": "Next Actions"
}
```
| `risks` | no | `array` |  |  |
- `risks`: Items: `string`

Unsupported property risks metadata:

```json
{
  "title": "Risks"
}
```
| `summary` | yes | `string` |  |  |

Unsupported property summary metadata:

```json
{
  "title": "Summary"
}
```
| `title` | yes | `string` |  |  |

Unsupported property title metadata:

```json
{
  "title": "Title"
}
```

<a id="releasedecisiondraftrequest"></a>
## ReleaseDecisionDraftRequest
- Type: `object`
- Required properties: `scenario`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `allowed_roots` | no | `array` | Additional repo-relative allowed roots inside the workspace. |  |
- `allowed_roots`: Items: `string`

Unsupported property allowed_roots metadata:

```json
{
  "title": "Allowed Roots"
}
```
| `max_output_tokens` | no | `integer` |  | minimum=400.0, maximum=8000.0 |
- `max_output_tokens`: Default: `2200`

Unsupported property max_output_tokens metadata:

```json
{
  "title": "Max Output Tokens"
}
```
| `max_tool_rounds` | no | `integer` |  | minimum=1.0, maximum=10.0 |
- `max_tool_rounds`: Default: `6`

Unsupported property max_tool_rounds metadata:

```json
{
  "title": "Max Tool Rounds"
}
```
| `model` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "description": "Optional Responses API model.",   "title": "Model" }` | Optional Responses API model. |  |

Unsupported property model metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Model"
}
```
| `scenario` | yes | `string` | Built-in scenario preset name. |  |

Unsupported property scenario metadata:

```json
{
  "title": "Scenario"
}
```
| `seed_paths` | no | `array` | Additional repo-relative seed paths. |  |
- `seed_paths`: Items: `string`

Unsupported property seed_paths metadata:

```json
{
  "title": "Seed Paths"
}
```
| `task_prompt` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "description": "Optional override prompt for the selected scenario.",   "title": "Task Prompt" }` | Optional override prompt for the selected scenario. |  |

Unsupported property task_prompt metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Task Prompt"
}
```

<a id="releasedecisiondraftresponse"></a>
## ReleaseDecisionDraftResponse
- Type: `object`
- Required properties: `markdown`, `result`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `markdown` | yes | `string` |  |  |

Unsupported property markdown metadata:

```json
{
  "title": "Markdown"
}
```
| `result` | yes | [ReleaseDecisionRunResult](api-schemas.md#releasedecisionrunresult) |  |  |

<a id="releasedecisionevalcasesummary"></a>
## ReleaseDecisionEvalCaseSummary
- Type: `object`
- Required properties: `case_id`, `expected_decision`, `scenario`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `case_id` | yes | `string` |  |  |

Unsupported property case_id metadata:

```json
{
  "title": "Case Id"
}
```
| `expected_decision` | yes | `string` |  |  |

Unsupported property expected_decision metadata:

```json
{
  "title": "Expected Decision"
}
```
| `required_keywords` | no | `array` |  |  |
- `required_keywords`: Items: `string`

Unsupported property required_keywords metadata:

```json
{
  "title": "Required Keywords"
}
```
| `required_sources` | no | `array` |  |  |
- `required_sources`: Items: `string`

Unsupported property required_sources metadata:

```json
{
  "title": "Required Sources"
}
```
| `scenario` | yes | `string` |  |  |

Unsupported property scenario metadata:

```json
{
  "title": "Scenario"
}
```

<a id="releasedecisionevalresult"></a>
## ReleaseDecisionEvalResult
- Type: `object`
- Required properties: `case_id`, `decision_ok`, `keyword_hits`, `missing_keywords`, `missing_sources`, `source_hits`, `status`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `case_id` | yes | `string` |  |  |

Unsupported property case_id metadata:

```json
{
  "title": "Case Id"
}
```
| `decision_ok` | yes | `boolean` |  |  |

Unsupported property decision_ok metadata:

```json
{
  "title": "Decision Ok"
}
```
| `keyword_hits` | yes | `object` |  |  |

Unsupported property keyword_hits metadata:

```json
{
  "additionalProperties": {
    "type": "boolean"
  },
  "title": "Keyword Hits"
}
```
| `missing_keywords` | yes | `array` |  |  |
- `missing_keywords`: Items: `string`

Unsupported property missing_keywords metadata:

```json
{
  "title": "Missing Keywords"
}
```
| `missing_sources` | yes | `array` |  |  |
- `missing_sources`: Items: `string`

Unsupported property missing_sources metadata:

```json
{
  "title": "Missing Sources"
}
```
| `source_hits` | yes | `object` |  |  |

Unsupported property source_hits metadata:

```json
{
  "additionalProperties": {
    "type": "boolean"
  },
  "title": "Source Hits"
}
```
| `status` | yes | `string` |  |  |
- `status`: Enum: `PASS`, `FAIL`

Unsupported property status metadata:

```json
{
  "title": "Status"
}
```

<a id="releasedecisionevaluaterequest"></a>
## ReleaseDecisionEvaluateRequest
- Type: `object`
- Required properties: `case_id`, `draft`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `case_id` | yes | `string` |  |  |

Unsupported property case_id metadata:

```json
{
  "title": "Case Id"
}
```
| `draft` | yes | [ReleaseDecisionDraft](api-schemas.md#releasedecisiondraft) |  |  |

<a id="releasedecisionevaluateresponse"></a>
## ReleaseDecisionEvaluateResponse
- Type: `object`
- Required properties: `case`, `evaluation`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `case` | yes | [ReleaseDecisionEvalCaseSummary](api-schemas.md#releasedecisionevalcasesummary) |  |  |
| `evaluation` | yes | [ReleaseDecisionEvalResult](api-schemas.md#releasedecisionevalresult) |  |  |

<a id="releasedecisionpresetresponse"></a>
## ReleaseDecisionPresetResponse
- Type: `object`
- Required properties: `allowed_roots`, `scenario`, `seed_paths`, `task_prompt`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `allowed_roots` | yes | `array` |  |  |
- `allowed_roots`: Items: `string`

Unsupported property allowed_roots metadata:

```json
{
  "title": "Allowed Roots"
}
```
| `scenario` | yes | `string` |  |  |

Unsupported property scenario metadata:

```json
{
  "title": "Scenario"
}
```
| `seed_paths` | yes | `array` |  |  |
- `seed_paths`: Items: `string`

Unsupported property seed_paths metadata:

```json
{
  "title": "Seed Paths"
}
```
| `task_prompt` | yes | `string` |  |  |

Unsupported property task_prompt metadata:

```json
{
  "title": "Task Prompt"
}
```

<a id="releasedecisionrunresult"></a>
## ReleaseDecisionRunResult
- Type: `object`
- Required properties: `draft`, `generated_at_utc`, `model`, `raw_response_text`, `scenario`, `seed_paths`, `task_prompt`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `draft` | yes | [ReleaseDecisionDraft](api-schemas.md#releasedecisiondraft) |  |  |
| `generated_at_utc` | yes | `string` |  |  |

Unsupported property generated_at_utc metadata:

```json
{
  "title": "Generated At Utc"
}
```
| `model` | yes | `string` |  |  |

Unsupported property model metadata:

```json
{
  "title": "Model"
}
```
| `raw_response_text` | yes | `string` |  |  |

Unsupported property raw_response_text metadata:

```json
{
  "title": "Raw Response Text"
}
```
| `response_id` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Response Id" }` |  |  |

Unsupported property response_id metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Response Id"
}
```
| `scenario` | yes | `string` |  |  |

Unsupported property scenario metadata:

```json
{
  "title": "Scenario"
}
```
| `seed_paths` | yes | `array` |  |  |
- `seed_paths`: Items: `string`

Unsupported property seed_paths metadata:

```json
{
  "title": "Seed Paths"
}
```
| `task_prompt` | yes | `string` |  |  |

Unsupported property task_prompt metadata:

```json
{
  "title": "Task Prompt"
}
```
| `tool_trace` | no | `array` |  |  |
- `tool_trace`: Items: [ToolTraceItem](api-schemas.md#tooltraceitem)

Unsupported property tool_trace metadata:

```json
{
  "title": "Tool Trace"
}
```

<a id="releasedecisionsaverequest"></a>
## ReleaseDecisionSaveRequest
- Type: `object`
- Required properties: `draft_response`, `markdown`, `scenario`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `allowed_roots` | no | `array` |  |  |
- `allowed_roots`: Items: `string`

Unsupported property allowed_roots metadata:

```json
{
  "title": "Allowed Roots"
}
```
| `draft_response` | yes | [ReleaseDecisionRunResult](api-schemas.md#releasedecisionrunresult) |  |  |
| `evaluation` | no | `{   "anyOf": [     {       "$ref": "#/components/schemas/ReleaseDecisionEvaluateResponse"     },     {       "type": "null"     }   ] }` |  |  |

Unsupported property evaluation metadata:

```json
{
  "anyOf": [
    {
      "$ref": "#/components/schemas/ReleaseDecisionEvaluateResponse"
    },
    {
      "type": "null"
    }
  ]
}
```
| `markdown` | yes | `string` |  | minLength=1 |

Unsupported property markdown metadata:

```json
{
  "title": "Markdown"
}
```
| `scenario` | yes | `string` |  |  |

Unsupported property scenario metadata:

```json
{
  "title": "Scenario"
}
```
| `seed_paths` | no | `array` |  |  |
- `seed_paths`: Items: `string`

Unsupported property seed_paths metadata:

```json
{
  "title": "Seed Paths"
}
```
| `task_prompt` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Task Prompt" }` |  |  |

Unsupported property task_prompt metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Task Prompt"
}
```

<a id="runingestpayload"></a>
## RunIngestPayload
- Type: `object`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `mode` | no | `string` |  |  |
- `mode`: Default: `"INCREMENTAL"`

Unsupported property mode metadata:

```json
{
  "title": "Mode"
}
```
| `season_year` | no | `{   "anyOf": [     {       "type": "integer"     },     {       "type": "null"     }   ],   "title": "Season Year" }` |  |  |

Unsupported property season_year metadata:

```json
{
  "anyOf": [
    {
      "type": "integer"
    },
    {
      "type": "null"
    }
  ],
  "title": "Season Year"
}
```
| `since` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Since" }` |  |  |

Unsupported property since metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Since"
}
```
| `tables` | no | `{   "anyOf": [     {       "items": {         "type": "string"       },       "type": "array"     },     {       "type": "null"     }   ],   "title": "Tables" }` |  |  |

Unsupported property tables metadata:

```json
{
  "anyOf": [
    {
      "items": {
        "type": "string"
      },
      "type": "array"
    },
    {
      "type": "null"
    }
  ],
  "title": "Tables"
}
```
| `trigger_source` | no | `string` |  |  |
- `trigger_source`: Default: `"MANUAL_API"`

Unsupported property trigger_source metadata:

```json
{
  "title": "Trigger Source"
}
```

<a id="searchanalysisresponse"></a>
## SearchAnalysisResponse

검색 분석 결과를 담는 응답 모델
- Type: `object`
- Required properties: `entity_analysis`, `execution_time_ms`, `performance_metrics`, `query`, `results`, `search_strategy`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `entity_analysis` | yes | `object` |  |  |

Unsupported property entity_analysis metadata:

```json
{
  "additionalProperties": true,
  "title": "Entity Analysis"
}
```
| `execution_time_ms` | yes | `number` |  |  |

Unsupported property execution_time_ms metadata:

```json
{
  "title": "Execution Time Ms"
}
```
| `performance_metrics` | yes | `object` |  |  |

Unsupported property performance_metrics metadata:

```json
{
  "additionalProperties": true,
  "title": "Performance Metrics"
}
```
| `query` | yes | `string` |  |  |

Unsupported property query metadata:

```json
{
  "title": "Query"
}
```
| `results` | yes | `array` |  |  |
- `results`: Items: `object`

Unsupported property results metadata:

```json
{
  "title": "Results"
}
```
| `search_strategy` | yes | `object` |  |  |

Unsupported property search_strategy metadata:

```json
{
  "additionalProperties": true,
  "title": "Search Strategy"
}
```

<a id="seatviewclassification"></a>
## SeatViewClassification
- Type: `object`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `confidence` | no | `{   "anyOf": [     {       "type": "number"     },     {       "type": "null"     }   ],   "title": "Confidence" }` |  |  |

Unsupported property confidence metadata:

```json
{
  "anyOf": [
    {
      "type": "number"
    },
    {
      "type": "null"
    }
  ],
  "title": "Confidence"
}
```
| `label` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Label" }` |  |  |

Unsupported property label metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Label"
}
```
| `reason` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Reason" }` |  |  |

Unsupported property reason metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Reason"
}
```

<a id="ticketinfo"></a>
## TicketInfo
- Type: `object`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `awayTeam` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Awayteam" }` |  |  |

Unsupported property awayTeam metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Awayteam"
}
```
| `date` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Date" }` |  |  |

Unsupported property date metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Date"
}
```
| `homeTeam` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Hometeam" }` |  |  |

Unsupported property homeTeam metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Hometeam"
}
```
| `peopleCount` | no | `{   "anyOf": [     {       "type": "integer"     },     {       "type": "null"     }   ],   "title": "Peoplecount" }` |  |  |

Unsupported property peopleCount metadata:

```json
{
  "anyOf": [
    {
      "type": "integer"
    },
    {
      "type": "null"
    }
  ],
  "title": "Peoplecount"
}
```
| `price` | no | `{   "anyOf": [     {       "type": "integer"     },     {       "type": "null"     }   ],   "title": "Price" }` |  |  |

Unsupported property price metadata:

```json
{
  "anyOf": [
    {
      "type": "integer"
    },
    {
      "type": "null"
    }
  ],
  "title": "Price"
}
```
| `reservationNumber` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Reservationnumber" }` |  |  |

Unsupported property reservationNumber metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Reservationnumber"
}
```
| `row` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Row" }` |  |  |

Unsupported property row metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Row"
}
```
| `seat` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Seat" }` |  |  |

Unsupported property seat metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Seat"
}
```
| `section` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Section" }` |  |  |

Unsupported property section metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Section"
}
```
| `stadium` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Stadium" }` |  |  |

Unsupported property stadium metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Stadium"
}
```
| `time` | no | `{   "anyOf": [     {       "type": "string"     },     {       "type": "null"     }   ],   "title": "Time" }` |  |  |

Unsupported property time metadata:

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "null"
    }
  ],
  "title": "Time"
}
```

<a id="tooltraceitem"></a>
## ToolTraceItem
- Type: `object`
- Required properties: `arguments`, `result_preview`, `tool_name`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `arguments` | yes | `object` |  |  |

Unsupported property arguments metadata:

```json
{
  "additionalProperties": true,
  "title": "Arguments"
}
```
| `result_preview` | yes | `string` |  |  |

Unsupported property result_preview metadata:

```json
{
  "title": "Result Preview"
}
```
| `tool_name` | yes | `string` |  |  |

Unsupported property tool_name metadata:

```json
{
  "title": "Tool Name"
}
```

<a id="validationerror"></a>
## ValidationError
- Type: `object`
- Required properties: `loc`, `msg`, `type`

### Properties

| Name | Required | Schema | Description | Constraints |
| --- | --- | --- | --- | --- |
| `ctx` | no | `object` |  |  |

Unsupported property ctx metadata:

```json
{
  "title": "Context"
}
```
| `input` | no | `{   "title": "Input" }` |  |  |

Unsupported property input metadata:

```json
{
  "title": "Input"
}
```
| `loc` | yes | `array` |  |  |
- `loc`: Items: `{   "anyOf": [     {       "type": "string"     },     {       "type": "integer"     }   ] }`

Unsupported property loc metadata:

```json
{
  "title": "Location"
}
```
| `msg` | yes | `string` |  |  |

Unsupported property msg metadata:

```json
{
  "title": "Message"
}
```
| `type` | yes | `string` |  |  |

Unsupported property type metadata:

```json
{
  "title": "Error Type"
}
```
