import pytest
from app.core.entity_extractor import standardize_position, extract_position_code


def test_standardize_position_full_names():
    assert standardize_position("1루수") == "1B"
    assert standardize_position("2루수") == "2B"
    assert standardize_position("3루수") == "3B"
    assert standardize_position("유격수") == "SS"
    assert standardize_position("좌익수") == "LF"
    assert standardize_position("중견수") == "CF"
    assert standardize_position("우익수") == "RF"
    assert standardize_position("지명타자") == "DH"
    assert standardize_position("포수") == "C"
    assert standardize_position("투수") == "P"


def test_standardize_position_hanja():
    assert standardize_position("一") == "1B"
    assert standardize_position("二") == "2B"
    assert standardize_position("三") == "3B"


def test_standardize_position_short_hangul():
    assert standardize_position("유") == "SS"
    assert standardize_position("좌") == "LF"
    assert standardize_position("우") == "RF"
    assert standardize_position("중") == "CF"
    assert standardize_position("포") == "C"
    assert standardize_position("투") == "P"
    assert standardize_position("타") == "PH"
    assert standardize_position("주") == "PR"
    assert standardize_position("지") == "DH"


def test_standardize_position_combined():
    assert standardize_position("타一") == "PH"
    assert standardize_position("주二") == "PR"
    assert standardize_position("지타") == "DH"


def test_extract_position_code_from_query():
    assert extract_position_code("오늘 1루수 누구야?") == "1B"
    assert extract_position_code("최정의 3루수 성적") == "3B"
    assert extract_position_code("유격수 포지션 기록") == "SS"
    assert extract_position_code("투수 방어율 순위") == "P"
    assert extract_position_code("일루수") == "1B"  # partial match or keyword
    assert extract_position_code("대타 성공률") == "PH"


def test_standardize_position_edge_cases():
    assert standardize_position("") is None
    assert standardize_position(None) is None
    assert standardize_position("Unknown") is None
    assert standardize_position("1B") == "1B"
