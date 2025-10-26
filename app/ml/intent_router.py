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
    stats_pattern = r"(순위|상위|평균|합계|ERA|OPS|승률|다음 경기|라인업|박스스코어)"
    explanatory_pattern = r"(역사|구단 변천|구장|파크팩터|특징)"

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
