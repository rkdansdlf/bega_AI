"""
질문 의도를 분류하기 위한 규칙/모델 기반 라우터 모듈.
챗봇의 '기억력'을 담당하는 핵심역할


"""

import re
from pathlib import Path
from typing import Optional

import joblib

MODEL_PATH = Path(__file__).with_name("intent_router.joblib")


def rule_intent(question: str) -> Optional[str]:
    game_lookup_pattern = r"(박스스코어|box score|이닝별 득점|이닝별|몇 점|7회|8회|9회|연장|경기 흐름|흐름 요약|승부처|언제 갈렸|역전|동점 흐름|초중후반 득점|득점 양상)"
    team_metric_pattern = (
        r"((팀|구단).*(타율|OPS|ops|평균자책|평균 자책|평균자책점|ERA|era|방어율|홈런|타점)|"
        r"(타율|OPS|ops|평균자책|평균 자책|평균자책점|ERA|era|방어율|홈런|타점).*(팀|구단))"
    )
    stats_pattern = r"(순위|상위|평균|합계|ERA|OPS|승률|다음 경기|라인업|박스스코어|일정|결과|기록|플레이오프|포스트시즌|세이브|MVP|mvp|신인왕|골든글러브|수상)"
    explanatory_pattern = (
        r"(역사|구단 변천|구장|파크팩터|특징|규정|룰|규칙|뜻|의미|왜|설명|원리|해설|"
        r"wrc\+|war|ops|whip|fip|babip|qs|abs|세이버|지표|전술|전략|포지션|플래툰|"
        r"번트|히트앤런|수비 시프트|체인지업|커브|슬라이더|피치클락|태그업|인필드 플라이|"
        r"필승조|포수 리드|체크 스윙)"
    )
    long_tail_pattern = (
        r"(마스코트|응원가|별명|엠블럼|유니폼|팬 문화|응원 문화|라이벌|"
        r"홈구장|구장 분위기|구단 역사|전통|사직|잠실|대구|광주|인천|수원|창원)"
    )
    latest_temporal_pattern = r"(오늘|지금|현재|실시간|방금|최신|속보|금일|최근|요즘)"
    latest_subject_pattern = (
        r"(선발|라인업|엔트리|등록|말소|부상|트레이드|루머|이슈|소식|"
        r"현황|상황|점수|중계|속보|맞대결|활약|경기|일정)"
    )

    analysis_pattern = (
        r"(승리 확률|승리확률|기대 승률|해결사|클러치|위기 상황|승부처|역전|WPA|득점권)"
    )
    prediction_pattern = (
        r"(누가 이길까|승부 예측|예상|매치업|상성|대결|승자|이길 확률|vs|VS)"
    )
    bullpen_pattern = r"(불펜|구원 투수|가용|나올 수|등판 가능|투구 수|혹사|계투)"
    strategy_pattern = r"(교체|누구 올|바꿔야|추천|작전|다음 투수|올려야|마무리)"

    if re.search(analysis_pattern, question):
        return "game_analysis"
    if re.search(prediction_pattern, question):
        return "match_prediction"
    if re.search(game_lookup_pattern, question):
        return "game_lookup"
    if re.search(team_metric_pattern, question):
        return "team_analysis"
    if re.search(bullpen_pattern, question):
        return "bullpen_check"
    if re.search(strategy_pattern, question):
        return "strategy_recommendation"
    if re.search(latest_temporal_pattern, question) and re.search(
        latest_subject_pattern, question
    ):
        return "latest_info"
    if re.search(long_tail_pattern, question):
        return "long_tail_entity"
    if re.search(explanatory_pattern, question):
        return "baseball_explainer"
    if re.search(stats_pattern, question):
        return "stats_lookup"
    return None


_clf = None


def load_clf() -> None:
    global _clf
    if MODEL_PATH.exists():
        _clf = joblib.load(MODEL_PATH)


def predict_intent(question: str) -> str:
    intent = rule_intent(question)
    if intent:
        return intent
    if _clf:
        return _clf.predict([question])[0]
    return "freeform"
