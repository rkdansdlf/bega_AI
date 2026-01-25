"""
'The Coach' 기능과 관련된 API 엔드포인트를 정의합니다.
"""

import json
import logging
import re
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


def _create_fallback_response(team_name: str, error_message: str = "") -> Dict[str, Any]:
    """JSON 파싱 실패 시 폴백 응답 생성"""
    return {
        "dashboard": {
            "headline": "분석 일시 불가",
            "context": f"{team_name} 팀 분석 중 오류가 발생했습니다. {error_message}",
            "sentiment": "neutral",
            "stats": []
        },
        "metrics": [],
        "detailed_analysis": "분석 데이터를 불러오는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        "coach_note": "현재 분석 시스템이 일시적으로 응답하지 않습니다."
    }


def _parse_llm_json_response(raw_answer: str) -> Optional[Dict[str, Any]]:
    """LLM의 응답에서 JSON을 파싱합니다. 코드 블록 제거 및 정제 포함."""
    if not raw_answer or not raw_answer.strip():
        return None
    
    text = raw_answer.strip()
    
    # 코드 블록 제거 (```json ... ```)
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if code_block_match:
        text = code_block_match.group(1).strip()
    
    # JSON 객체 추출 시도 ({ ... })
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        text = json_match.group(0)
    
    try:
        parsed = json.loads(text)
        # 필수 필드 검증
        if "dashboard" in parsed:
            return parsed
        else:
            logger.warning("[Coach Router] JSON parsed but missing 'dashboard' field")
            return None
    except json.JSONDecodeError as e:
        logger.warning(f"[Coach Router] JSON parsing failed: {e}. Raw: {text[:300]}...")
        return None


@router.post("/analyze")
async def analyze_team(
    payload: AnalyzeRequest,
    agent: BaseballStatisticsAgent = Depends(get_agent),
    _: None = Depends(rate_limit_dependency),
):
    """
    특정 팀에 대한 심층 분석을 요청합니다. 'The Coach' 페르소나가 적용됩니다.
    Returns structured JSON instead of Markdown.
    """
    try:
        team_name = agent._convert_team_id_to_name(payload.team_id)
        
        # Focus mapping (ID -> Korean label)
        FOCUS_LABELS = {
            "recent_form": "최근 전력",
            "bullpen": "불펜 상태",
            "matchup": "상대 전적",
            "starter": "선발 투수",
            "batting": "타격 생산성"
        }
        
        # 질문 구성
        if payload.question_override:
            query = payload.question_override
        else:
            # Convert focus IDs to Korean labels
            selected_focus = payload.focus if payload.focus else ["recent_form"]
            focus_labels = [FOCUS_LABELS.get(f, f) for f in selected_focus]
            focus_text = ", ".join(focus_labels)
            
            # Use centralized prompt from prompts.py
            system_prompt = COACH_PROMPT
            
            # 선택된 포인트만 명시적으로 요청
            query = f"{team_name}의 분석을 수행해줘. **오직 다음 항목만** 분석하세요: {focus_text}."
            
            # 선택된 포인트에 대해서만 상세 지시 추가
            if "bullpen" in selected_focus:
                query += " [불펜] 하이 레버리지 상황 처리 능력과 과부하 지표를 분석해줘."
                
            if "recent_form" in selected_focus:
                query += " [최근 전력] 최근 5~10경기 승패 패턴과 득실점 효율성을 포함해줘."
                
            if "starter" in selected_focus:
                query += " [선발 투수] 선발 로테이션의 이닝 소화력과 QS 비율을 분석해줘."
                
            if "matchup" in selected_focus:
                query += " [상대 전적] 리그 내 라이벌 팀들과의 상성 패턴을 분석해줘."

        logger.info(f"[Coach Router] Analyzing for {team_name}: {query}")

        # 에이전트 호출 (Coach 페르소나 적용)
        context_data = {
            "persona": "coach",
            "team_id": payload.team_id
        }
        if 'system_prompt' in locals():
            context_data["system_message"] = system_prompt

        raw_answer = ""
        tool_calls = []
        verified = False
        data_sources = []
        parsed_response = None

        # 빈 응답에 대한 재시도 로직
        for attempt in range(MAX_RETRY_ON_EMPTY + 1):
            result = await agent.process_query(
                query,
                context=context_data
            )

            # 스트리밍 응답(async_generator)일 경우 텍스트로 변환
            answer = result.get("answer")
            if hasattr(answer, '__aiter__'):
                full_answer = ""
                async for chunk in answer:
                    if chunk:
                        full_answer += chunk
                result["answer"] = full_answer

            raw_answer = result.get("answer", "")
            tool_calls = result.get("tool_calls", [])
            verified = result.get("verified", False)
            data_sources = result.get("data_sources", [])

            # JSON 파싱 시도
            if raw_answer.strip():
                parsed_response = _parse_llm_json_response(raw_answer)
                if parsed_response:
                    if attempt > 0:
                        logger.info(f"[Coach Router] Retry {attempt} succeeded with valid JSON")
                    break
                else:
                    logger.warning(f"[Coach Router] Attempt {attempt + 1}: Got response but failed to parse JSON")
            
            if attempt < MAX_RETRY_ON_EMPTY:
                logger.warning(f"[Coach Router] Attempt {attempt + 1} failed, retrying...")
            else:
                logger.error(f"[Coach Router] All {MAX_RETRY_ON_EMPTY + 1} attempts failed to produce valid JSON")

        # 파싱 실패 시 폴백 응답 반환
        if not parsed_response:
            logger.error(f"[Coach Router] Using fallback response. Raw answer length: {len(raw_answer)}")
            parsed_response = _create_fallback_response(team_name, "JSON 파싱 실패")

        return {
            "data": parsed_response,
            "raw_answer": raw_answer,  # 디버깅용 (개발 중에만 사용)
            "tool_calls": tool_calls,
            "verified": verified,
            "data_sources": data_sources
        }

    except Exception as e:
        logger.error(f"[Coach Router] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

