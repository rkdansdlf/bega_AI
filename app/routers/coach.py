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
    스트리밍(SSE) 응답을 지원합니다.
    """
    try:
        team_name = agent._convert_team_id_to_name(payload.team_id)

        # 질문 구성
        if payload.question_override:
            query = payload.question_override
        else:
            focus_text = ", ".join(payload.focus) if payload.focus else "종합적인 전력"
            
            # Use centralized prompt from prompts.py (COACH_PROMPT is handled inside agent based on persona)
            system_prompt = COACH_PROMPT

            query = (
                f"{team_name}의 {focus_text}에 대해 냉철하고 다각적인 분석을 수행해줘."
            )

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

        context_data = {"persona": "coach", "team_id": payload.team_id}
        # Note: system_prompt is passed implicitly via persona logic in _generate_verified_answer
        # inside the agent, but we set it here if we want to force it in context
        if "system_prompt" in locals():
             context_data["system_message"] = system_prompt

        async def event_generator():
            # Use process_query_stream to get real-time events, including tool execution
            async for event in agent.process_query_stream(query, context=context_data):
                if event["type"] == "status":
                    yield {
                        "event": "status",
                        "data": json.dumps({"message": event["message"]}, ensure_ascii=False)
                    }
                elif event["type"] == "tool_start":
                    yield {
                        "event": "tool_start",
                        "data": json.dumps({"tool": event["tool"]}, ensure_ascii=False)
                    }
                elif event["type"] == "answer_chunk":
                    yield {
                        "event": "message",
                        "data": json.dumps({"delta": event["content"]}, ensure_ascii=False)
                    }
                elif event["type"] == "metadata":
                     # Send metadata event
                     def safe_serialize(obj):
                        if hasattr(obj, "to_dict"):
                            return obj.to_dict()
                        if hasattr(obj, "__dict__"):
                             return str(obj)
                        return str(obj)

                     # Sanitize metadata (tool_calls, etc)
                     # We need to serialize recursively potentially
                     # For now, just pass what we can
                     from ..routers.chat_stream import ChatPayload # Import helper if possible or just rely on json dump
                     # Manually serializing for safety as in original code
                     meta_payload = {
                        "tool_calls": [tc.to_dict() for tc in event["data"]["tool_calls"]],
                        "verified": event["data"]["verified"],
                        "data_sources": event["data"]["data_sources"]
                     }
                     yield {
                        "event": "meta",
                        "data": json.dumps(meta_payload, ensure_ascii=False)
                     }
            
            yield {"event": "done", "data": "[DONE]"}

        from sse_starlette.sse import EventSourceResponse
        return EventSourceResponse(
            event_generator(),
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        logger.error(f"[Coach Router] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
