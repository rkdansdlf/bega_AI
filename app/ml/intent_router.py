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
    stats_pattern = r"(순위|상위|평균|합계|ERA|OPS|승률|다음 경기|라인업|박스스코어|일정|결과|기록|플레이오프|포스트시즌|세이브)"
    explanatory_pattern = r"(역사|구단 변천|구장|파크팩터|특징|규정|룰|규칙)"

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
    if re.search(bullpen_pattern, question):
        return "bullpen_check"
    if re.search(strategy_pattern, question):
        return "strategy_recommendation"
    if re.search(stats_pattern, question):
        return "stats_lookup"

    if re.search(explanatory_pattern, question):
        return "explanatory"
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
