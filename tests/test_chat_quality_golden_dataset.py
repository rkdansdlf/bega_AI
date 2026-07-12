import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest


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
APPROVED_CATEGORIES = {
    "multi_player_narrative",
    "regulation",
    "team_db",
    "recovered_regression",
}
APPROVED_PLANNER_MODES = {
    "default_llm_planner",
    "fast_path",
    "fast_path_bundle",
    "player_fast_path",
    "player_llm_planner",
    "predefined",
    "team_llm_planner",
}


def _load_dataset() -> dict:
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


def _assert_case_contract(case: object) -> None:
    assert isinstance(case, dict)
    assert {
        "id",
        "question",
        "category",
        "expected_answerability",
        "allowed_planner_modes",
    }.issubset(case)

    case_id = case["id"]
    question = case["question"]
    assert isinstance(case_id, str) and case_id.strip()
    assert isinstance(question, str) and question.strip()
    category = case["category"]
    expected_answerability = case["expected_answerability"]
    assert isinstance(category, str) and category in APPROVED_CATEGORIES
    assert (
        isinstance(expected_answerability, str)
        and expected_answerability in APPROVED_ANSWERABILITY
    )

    allowed_planner_modes = case["allowed_planner_modes"]
    assert isinstance(allowed_planner_modes, list) and allowed_planner_modes
    assert all(
        isinstance(mode, str)
        and mode.strip()
        and mode in APPROVED_PLANNER_MODES
        for mode in allowed_planner_modes
    )
    assert len(allowed_planner_modes) == len(set(allowed_planner_modes))


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

    for case in cases:
        _assert_case_contract(case)

    assert len({case["id"] for case in cases}) == len(cases)
    assert len({case["question"] for case in cases}) == len(cases)
    assert all("http://" not in case["question"] for case in cases)
    assert all("https://" not in case["question"] for case in cases)
    assert all(case["allowed_planner_modes"] for case in cases)
    assert all(
        case["expected_answerability"] in APPROVED_ANSWERABILITY for case in cases
    )


def test_case_contract_rejects_non_list_planner_modes() -> None:
    malformed_case = {
        "id": "malformed-case",
        "question": "A valid question",
        "category": "regulation",
        "expected_answerability": "answerable",
        "allowed_planner_modes": "fast_path",
    }

    with pytest.raises(AssertionError):
        _assert_case_contract(malformed_case)
