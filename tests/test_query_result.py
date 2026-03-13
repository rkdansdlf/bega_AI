from app.tools.query_result import apply_error, apply_list_results, build_search_result


def test_build_search_result_sets_default_flags() -> None:
    result = build_search_result(query="질문", documents=[], source="verified_docs")

    assert result == {
        "found": False,
        "error": None,
        "query": "질문",
        "documents": [],
        "source": "verified_docs",
    }


def test_apply_list_results_sets_found_total_and_extra_updates() -> None:
    result = build_search_result(regulations=[], total_found=0, categories=[])

    apply_list_results(
        result,
        field="regulations",
        rows=[{"title": "A"}, {"title": "B"}],
        total_field="total_found",
        extra_updates={"categories": ["player"], "matched_query": "FA"},
    )

    assert result["found"] is True
    assert result["total_found"] == 2
    assert result["categories"] == ["player"]
    assert result["matched_query"] == "FA"
    assert result["regulations"] == [{"title": "A"}, {"title": "B"}]


def test_apply_error_updates_existing_result() -> None:
    result = build_search_result(query="질문")

    apply_error(result, "오류")

    assert result["error"] == "오류"
    assert result["found"] is False
