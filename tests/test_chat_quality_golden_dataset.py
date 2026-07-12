import hashlib
import json
from collections import Counter
from pathlib import Path


GOLDEN_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "chat_quality_golden_60.json"
)
EXPECTED_SHA256 = (
    "8899b468d65a87a505083a5f229808a0b9b405819287f00ff707741715bb124d"
)
APPROVED_ANSWERABILITY = {
    "answerable",
    "operator_data_required",
    "clarification_required",
    "future_event_pending",
}


def _load_dataset() -> dict:
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


def test_golden_dataset_schema_and_provenance() -> None:
    dataset = _load_dataset()
    cases = dataset["cases"]

    assert dataset["schema_version"] == 1
    assert dataset["baseball_data_policy"] == "internal_only"
    assert len(cases) == 60
    assert Counter(case["category"] for case in cases) == {
        "multi_player_narrative": 20,
        "regulation": 20,
        "team_db": 14,
        "recovered_regression": 6,
    }
    assert hashlib.sha256(GOLDEN_DATASET_PATH.read_bytes()).hexdigest() == (
        EXPECTED_SHA256
    )


def test_golden_dataset_cases_are_contract_safe() -> None:
    cases = _load_dataset()["cases"]

    assert len({case["id"] for case in cases}) == len(cases)
    assert len({case["question"] for case in cases}) == len(cases)
    assert all("http://" not in case["question"] for case in cases)
    assert all("https://" not in case["question"] for case in cases)
    assert all(case["allowed_planner_modes"] for case in cases)
    assert all(
        case["expected_answerability"] in APPROVED_ANSWERABILITY for case in cases
    )
