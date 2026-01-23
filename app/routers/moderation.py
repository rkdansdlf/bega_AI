from fastapi import APIRouter, Body, HTTPException
import google.generativeai as genai
import json
import logging
from typing import Dict, Any

from ..config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/moderation", tags=["moderation"])

@router.post("/safety-check")
async def safety_check(payload: Dict[str, Any] = Body(...)):
    """
    게시글이나 댓글의 내용을 AI가 검사하여 부적절한 콘텐츠를 필터링합니다.
    """
    content = payload.get("content", "")
    if not content:
        raise HTTPException(status_code=400, detail="Content is required")

    settings = get_settings()
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY가 설정되지 않았습니다. 모든 콘텐츠를 허용합니다.")
        return {"category": "SAFE", "reason": "API Key not configured", "action": "ALLOW"}

    try:
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)

        prompt = f"""
        당신은 KBO(한국 프로야구) 커뮤니티의 콘텐츠 관리자입니다.
        다음 텍스트를 분석하여 카테고리를 분류하고 적절한 조치를 제안하세요.

        분류 카테고리:
        - SAFE: 정상적인 게시글 또는 댓글.
        - INAPPROPRIATE: 비속어, 욕설, 혐오 표현, 특정 팀이나 팬에 대한 과도한 비하.
        - SPAM: 상업적 광고, 무의미한 도배, 게시판 주제와 무관한 내용.
        - SPOILER: 실시간 경기 결과나 결정적 상황을 미리 유출하여 다른 팬의 즐거움을 해치는 내용.

        텍스트: "{content}"

        결과는 반드시 아래의 JSON 형식으로만 응답하세요:
        {{
          "category": "SAFE" | "INAPPROPRIATE" | "SPAM" | "SPOILER",
          "reason": "한국어로 된 짧은 이유",
          "action": "ALLOW" | "BLOCK"
        }}
        """

        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        result = json.loads(response.text)
        logger.info(f"Moderation result for content: {result}")
        return result

    except Exception as e:
        logger.error(f"Moderation error: {str(e)}")
        # 에러 발생 시 시스템 중단을 방지하기 위해 SAFE로 반환 (Fail-open 전략)
        return {
            "category": "SAFE", 
            "reason": f"System error: {str(e)}", 
            "action": "ALLOW"
        }
