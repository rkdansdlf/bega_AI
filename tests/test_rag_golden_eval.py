import json
from pathlib import Path

import pytest

from app.eval import grounding
from app.eval import retrieval_metrics as rm
from scripts import eval_rag_golden as gate

ROOT = Path(__file__).resolve().parents[1]


def _doc(table, row):
    return {"source_table": table, "source_row_id": row}


def test_metrics_perfect_and_partial_ranking():
    rel = frozenset({"a:1"})
    assert rm.recall_at_k(["a:1", "b:2"], rel, 5) == 1.0
    assert rm.reciprocal_rank(["b:2", "a:1"], rel) == 0.5
    assert rm.ndcg_at_k(["x:0", "a:1"], rel, 10) == pytest.approx(0.6309, abs=1e-3)
    assert rm.recall_at_k(["b:2"], rel, 5) == 0.0
    assert rm.recall_at_k(["b:2"], frozenset(), 5) is None


def test_wrong_source_detects_forbidden_and_missing_required():
    case = rm.RetrievalCase.from_dict(
        {
            "id": "c",
            "question": "q",
            "relevant_doc_keys": ["t:1"],
            "required_source_tables": ["t"],
            "must_not_sources": ["t:BAD", "game_inning_scores"],
        }
    )
    assert rm.is_wrong_source([_doc("t", "BAD")], case)
    assert rm.is_wrong_source([_doc("game_inning_scores", "9")], case)
    assert rm.is_wrong_source([_doc("other", "1")], case)
    assert not rm.is_wrong_source([_doc("t", "1")], case)


def test_no_answer_case_flags_violation_when_docs_returned():
    case = rm.RetrievalCase.from_dict(
        {"id": "n", "question": "q", "relevant_doc_keys": [], "expect_no_answer": True}
    )
    assert rm.evaluate_case(case, [_doc("t", "1")])["no_answer_violation"]
    assert not rm.evaluate_case(case, [])["no_answer_violation"]


def test_baseline_comparison_flags_regressions_only():
    base = {"recall@5": 0.9, "wrong_source_rate": 0.0}
    assert (
        rm.compare_to_baseline({"recall@5": 0.91, "wrong_source_rate": 0.0}, base) == []
    )
    problems = rm.compare_to_baseline({"recall@5": 0.5, "wrong_source_rate": 0.2}, base)
    assert len(problems) == 2


SRC = {"s1": "LG 트윈스 2025 시즌 팀 성적: 85승 57패 2무, 승률 0.599"}


def _case(**over):
    base = {
        "id": "g",
        "sources": SRC,
        "required_sources": ["s1"],
        "required_numbers": ["85"],
        "known_entities": ["KIA 타이거즈"],
        "non_answer_markers": ["확인할 수 없습니다"],
    }
    base.update(over)
    return base


def test_grounded_answer_passes_and_maps_claims_to_sources():
    out = grounding.evaluate_generation(
        _case(),
        "LG 트윈스는 2025 시즌 85승 57패를 기록했습니다. 승률은 0.599입니다.",
        ["s1"],
    )
    assert out["passed"], out
    assert out["unsupported_claims"] == []
    assert all(g["sources"] == ["s1"] for g in out["grounding"])


def test_numeric_hallucination_is_caught():
    out = grounding.evaluate_generation(
        _case(), "LG 트윈스는 2025 시즌 90승을 기록했습니다.", ["s1"]
    )
    assert not out["passed"]
    assert "90" in out["numeric_hallucinations"]
    assert out["unsupported_claims"]


def test_entity_hallucination_and_phantom_citation_are_caught():
    out = grounding.evaluate_generation(
        _case(),
        "KIA 타이거즈는 85승 57패를 기록했습니다.",
        ["s1", "does-not-exist"],
    )
    assert "KIA 타이거즈" in out["entity_hallucinations"]
    assert out["phantom_citations"] == ["does-not-exist"]
    assert out["citation_precision"] == 0.5
    assert not out["passed"]


def test_expected_non_answer_must_actually_refuse():
    case = _case(
        sources={}, required_sources=[], required_numbers=[], expect_non_answer=True
    )
    assert grounding.evaluate_generation(case, "확인할 수 없습니다.", [])["passed"]
    assert not grounding.evaluate_generation(case, "LG는 85승입니다.", [])["passed"]


def test_number_normalisation():
    assert grounding.extract_numbers("1,234명 0.300 12.50") == {"1234", "0.3", "12.5"}


def test_committed_datasets_pass_their_baselines():
    for kind in ("retrieval", "generation"):
        rc = gate.main(
            [kind, "--baseline", str(ROOT / "evals/baselines" / f"{kind}_v1.json")]
        )
        assert rc == 0


def test_gate_fails_when_recorded_results_regress(tmp_path):
    rec = json.loads((ROOT / "evals/rag_retrieval_v1.recorded.json").read_text())
    rec["ret-team-season-001"] = [_doc("team_summary", "team_id=KIA|season_year=2025")]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(rec), encoding="utf-8")
    rc = gate.main(
        [
            "retrieval",
            "--recorded",
            str(bad),
            "--baseline",
            str(ROOT / "evals/baselines/retrieval_v1.json"),
        ]
    )
    assert rc == 1


def test_runtime_grounding_maps_claims_to_chunk_ids_and_flags_unsupported():
    docs = [
        {"id": 101, "content": "LG 트윈스 2025 시즌 85승 57패 2무"},
        {"id": 102, "content": "KIA 타이거즈 2025 시즌 70승"},
    ]
    out = grounding.build_runtime_grounding(
        "LG 트윈스는 2025 시즌 85승 57패를 기록했습니다. LG 트윈스는 90승을 기록했습니다.",
        docs,
    )
    assert out["total_claims"] == 2
    assert out["claims"][0]["source_ids"] == ["101"]
    assert out["claims"][1]["supported"] is False
    assert out["unsupported_claims"] == 1
    assert "90" in out["unsupported_numbers"]
    assert out["coverage"] == 0.5


def test_runtime_grounding_without_docs_reports_everything_unsupported():
    out = grounding.build_runtime_grounding("LG는 85승입니다.", [])
    assert out["unsupported_claims"] == out["total_claims"] == 1
