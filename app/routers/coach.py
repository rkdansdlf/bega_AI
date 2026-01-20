"""
'The Coach' 기능과 관련된 API 엔드포인트를 정의합니다.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

from ..deps import get_agent
from ..agents.baseball_agent import BaseballStatisticsAgent
from ..core.ratelimit import rate_limit_dependency

logger = logging.getLogger(__name__)

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
            query = f"{team_name}의 {focus_text}에 대해 냉철하게 분석해줘."
            
            if "bullpen" in payload.focus:
                query += " 특히 불펜의 최근 성적과 과부하 여부를 중점적으로 봐줘."
            if "recent_form" in payload.focus:
                query += " 최근 5경기 승패와 득실점 마진을 포함해줘."
            if "matchup" in payload.focus and payload.game_id:
                # game_id가 있으면 좋겠지만, 없으면 "오늘 경기" 또는 "다음 경기"라고 가정
                query += " 상대 팀과의 전적 열세나 상성도 분석해줘."

        logger.info(f"[Coach Router] Analyzing for {team_name}: {query}")

        # 에이전트 호출 (Coach 페르소나 적용)
        result = await agent.process_query(
            query,
            context={
                "persona": "coach",
                "team_id": payload.team_id
            }
        )

        # 스트리밍 응답(async_generator)일 경우 텍스트로 변환
        answer = result.get("answer")
        if hasattr(answer, '__aiter__'):
            full_answer = ""
            async for chunk in answer:
                if chunk:
                    full_answer += chunk
            result["answer"] = full_answer

        return {
            "answer": result.get("answer"),
            "tool_calls": result.get("tool_calls", []),
            "verified": result.get("verified", False),
            "data_sources": result.get("data_sources", [])
        }

    except Exception as e:
        logger.error(f"[Coach Router] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
