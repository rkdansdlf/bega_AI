"""
'The Coach' 기능과 관련된 API 엔드포인트를 정의합니다.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

from ..deps import get_agent
from ..agents.baseball_agent import BaseballStatisticsAgent
from ..core.prompts import COACH_PROMPT
from ..core.ratelimit import rate_limit_dependency

logger = logging.getLogger(__name__)

# 빈 응답 시 재시도 횟수
MAX_RETRY_ON_EMPTY = 2

router = APIRouter(prefix="/coach", tags=["coach"])


class AnalyzeRequest(BaseModel):
    team_id: str
    focus: List[str] = []  # 예: ["bullpen", "recent_form", "matchup"]
    game_id: Optional[str] = None
    question_override: Optional[str] = None


@router.post("/analyze")
async def analyze_team(
    payload: AnalyzeRequest,
    agent: BaseballStatisticsAgent = Depends(get_agent),
    _: None = Depends(rate_limit_dependency),
):
    """
    특정 팀에 대한 심층 분석을 요청합니다. 'The Coach' 페르소나가 적용됩니다.
    """
    try:
        team_name = agent._convert_team_id_to_name(payload.team_id)

        # 질문 구성
        if payload.question_override:
            query = payload.question_override
        else:
            focus_text = ", ".join(payload.focus) if payload.focus else "종합적인 전력"

            # Use centralized prompt from prompts.py
            system_prompt = COACH_PROMPT

            query = (
                f"{team_name}의 {focus_text}에 대해 냉철하고 다각적인 분석을 수행해줘."
            )

            # 다각도 분석을 위해 기본적으로 포함될 수 있는 항목들 확장
            if "batting" in payload.focus or not payload.focus:
                query += " 팀의 타격 생산성(OPS, wRC+)과 주요 타자들의 최근 클러치 능력을 진단해줘."

            if "bullpen" in payload.focus:
                query += (
                    " 불펜진의 하이 레버리지 상황 처리 능력과 과부하 지표를 분석해줘."
                )

            if "recent_form" in payload.focus or not payload.focus:
                query += " 최근 5~10경기 승패 패턴과 득실점 효율성(Pythagorean Win %)을 포함해줘."

            if "starter" in payload.focus:
                query += " 선발 로테이션의 이닝 소화력과 QS 비율, 구속 변화를 분석해줘."

            if "matchup" in payload.focus:
                if payload.game_id:
                    query += (
                        " 특정 상대 팀과의 상성 및 전술적 우위/열세 포인트를 짚어줘."
                    )
                else:
                    query += " 리그 내 특정 라이벌 팀들과의 상성 패턴을 분석해줘."

        logger.info(f"[Coach Router] Analyzing for {team_name}: {query}")

        # 에이전트 호출 (Coach 페르소나 적용)
        context_data = {"persona": "coach", "team_id": payload.team_id}
        if "system_prompt" in locals():  # Only add if defined in the else block
            context_data["system_message"] = system_prompt

        final_answer = ""
        tool_calls = []
        verified = False
        data_sources = []

        # 빈 응답에 대한 재시도 로직
        for attempt in range(MAX_RETRY_ON_EMPTY + 1):
            result = await agent.process_query(query, context=context_data)

            # 스트리밍 응답(async_generator)일 경우 텍스트로 변환
            answer = result.get("answer")
            if hasattr(answer, "__aiter__"):
                full_answer = ""
                async for chunk in answer:
                    if chunk:
                        full_answer += chunk
                result["answer"] = full_answer

            final_answer = result.get("answer", "")
            tool_calls = result.get("tool_calls", [])
            verified = result.get("verified", False)
            data_sources = result.get("data_sources", [])

            # 빈 응답 체크
            if final_answer.strip():
                if attempt > 0:
                    logger.info(
                        f"[Coach Router] Retry {attempt} succeeded with {len(final_answer)} chars"
                    )
                break
            else:
                if attempt < MAX_RETRY_ON_EMPTY:
                    logger.warning(
                        f"[Coach Router] Empty response on attempt {attempt + 1}, retrying..."
                    )
                else:
                    logger.error(
                        f"[Coach Router] All {MAX_RETRY_ON_EMPTY + 1} attempts returned empty response"
                    )

        # 필수 섹션 검증 및 Preamble 제거: "## 🔍 AI 시즌 요약"으로 강제 시작
        if "## 🔍 AI 시즌 요약" in final_answer:
            final_answer = (
                "## 🔍 AI 시즌 요약" + final_answer.split("## 🔍 AI 시즌 요약", 1)[1]
            )
        elif "AI 시즌 요약" in final_answer:
            # ## 가 빠진 경우 보정
            header_part = final_answer.split("AI 시즌 요약", 1)[1]
            final_answer = "## 🔍 AI 시즌 요약" + header_part
        elif not final_answer.strip():
            # 모든 재시도 후에도 빈 응답인 경우 기본 오류 메시지 반환
            logger.error(
                "[Coach Router] AI response is completely EMPTY after all retries."
            )
            final_answer = """## 🔍 AI 시즌 요약
### 분석 일시 불가
AI 분석 서버가 일시적으로 응답하지 않습니다. 잠시 후 다시 시도해 주세요.

| 상태 | 설명 |
| :--- | :--- |
| 오류 | 응답 생성 실패 |
"""
        else:
            logger.warning(
                f"[Coach Router] Missing required header. Length: {len(final_answer)}. Content start: {final_answer[:500]!r}"
            )

        return {
            "answer": final_answer,
            "tool_calls": tool_calls,
            "verified": verified,
            "data_sources": data_sources,
        }

    except Exception as e:
        logger.error(f"[Coach Router] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
