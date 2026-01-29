import pytest
from unittest.mock import MagicMock, patch
from app.core.rag import MetaWrapper, _process_stat_doc_cached, _meta_cache_key


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
        # _process_stat_doc_cached는 LRU 캐시 데코레이터가 있음

        # 캐시 초기화를 위해 내부 cache_clear 호출 (있는 경우)
        if hasattr(_process_stat_doc_cached, "cache_clear"):
            _process_stat_doc_cached.cache_clear()

        meta = {
            "source_row_id": "cache_test_1",
            "player_name": "Choi",
            "innings_pitched": 100,
            # Add stats required for pitcher_rank_score calc
            "era": 3.50,
            "whip": 1.20,
            "strikeouts": 80,
            "walks_allowed": 20,
            "home_runs_allowed": 10,
            "hit_batters": 2,
            "tbf": 400,
        }
        wrapper = MetaWrapper(meta)

        # 첫 번째 호출
        res1 = _process_stat_doc_cached("player_season_pitching", wrapper)

        # 두 번째 호출 (같은 wrapper 객체)
        res2 = _process_stat_doc_cached("player_season_pitching", wrapper)

        # 세 번째 호출 (새로운 wrapper 객체지만 내용은 같음 -> 해시 같음)
        wrapper_clone = MetaWrapper({"source_row_id": "cache_test_1", "other": "val"})
        res3 = _process_stat_doc_cached("player_season_pitching", wrapper_clone)

        assert res1 == res2
        assert res1 == res3

        # LRU 캐시 정보 확인 (가능한 경우)
        if hasattr(_process_stat_doc_cached, "cache_info"):
            info = _process_stat_doc_cached.cache_info()
            # 첫 호출은 miss, 나머지 2번은 hit여야 함
            assert info.hits >= 2

    def test_precalc_usage(self):
        """Meta에 이미 score가 있으면 계산 로직을 건너뛰고 바로 반환해야 함"""
        precalc_meta = {"source_row_id": "precalc_1", "score": 95.5, "wrc_plus": 150}
        wrapper = MetaWrapper(precalc_meta)

        # Mocking kbo_metrics to ensure it's NOT called
        with patch("app.core.rag.kbo_metrics") as mock_metrics:
            result, warning = _process_stat_doc_cached("player_season_batting", wrapper)

            # 반환된 dict가 입력 meta와 같은지 확인 (pre-calc된 값 사용 시 그대로 반환하므로)
            assert result == precalc_meta
            assert warning is None

            # kbo_metrics 모듈의 함수들이 호출되지 않았따면 성공
            mock_metrics.wrc_plus.assert_not_called()
