from unittest.mock import patch

from app.core.rag import MetaWrapper, _process_stat_doc_cached


class TestRagCaching:
    def test_meta_wrapper_hash_equality(self):
        """source_row_id가 같으면 다른 객체라도 같은 해시를 가져야 함"""
        meta1 = {"source_row_id": "123", "player_name": "Kim"}
        meta2 = {"source_row_id": "123", "player_name": "Kim", "extra": "ignored"}

        wrapper1 = MetaWrapper(meta1)
        wrapper2 = MetaWrapper(meta2)

        assert wrapper1 == wrapper2
        assert hash(wrapper1) == hash(wrapper2)

    def test_meta_wrapper_hash_fallback(self):
        """source_row_id가 없으면 전체 딕셔너리 내용을 기반으로 해시 생성"""
        meta1 = {"player_name": "Lee", "stat": 1}
        meta2 = {"player_name": "Lee", "stat": 1}
        meta3 = {"player_name": "Park", "stat": 1}

        wrapper1 = MetaWrapper(meta1)
        wrapper2 = MetaWrapper(meta2)
        wrapper3 = MetaWrapper(meta3)

        assert wrapper1 == wrapper2
        assert hash(wrapper1) == hash(wrapper2)

        assert wrapper1 != wrapper3
        assert hash(wrapper1) != hash(wrapper3)

    def test_caching_mechanism(self):
        """LRU 캐시가 동작하는지 호출 횟수로 확인"""
        if hasattr(_process_stat_doc_cached, "cache_clear"):
            _process_stat_doc_cached.cache_clear()

        meta = {
            "source_row_id": "cache_test_1",
            "player_name": "Choi",
            "innings_pitched": 100,
            "era": 3.50,
            "whip": 1.20,
            "strikeouts": 80,
            "walks_allowed": 20,
            "home_runs_allowed": 10,
            "hit_batters": 2,
            "tbf": 400,
        }
        wrapper = MetaWrapper(meta)

        res1 = _process_stat_doc_cached("player_season_pitching", wrapper)
        res2 = _process_stat_doc_cached("player_season_pitching", wrapper)
        wrapper_clone = MetaWrapper({"source_row_id": "cache_test_1", "other": "val"})
        res3 = _process_stat_doc_cached("player_season_pitching", wrapper_clone)

        assert res1 == res2
        assert res1 == res3

        if hasattr(_process_stat_doc_cached, "cache_info"):
            info = _process_stat_doc_cached.cache_info()
            assert info.hits >= 2

    def test_precalc_usage(self):
        """Meta에 이미 score가 있으면 계산 로직을 건너뛰고 바로 반환해야 함"""
        precalc_meta = {"source_row_id": "precalc_1", "score": 95.5, "wrc_plus": 150}
        wrapper = MetaWrapper(precalc_meta)

        with patch("app.core.rag.kbo_metrics") as mock_metrics:
            result, warning = _process_stat_doc_cached("player_season_batting", wrapper)

            assert result == precalc_meta
            assert warning is None
            mock_metrics.wrc_plus.assert_not_called()
