"""
RAG (Retrieval-Augmented Generation) 파이프라인의 핵심 로직을 구현한 모듈입니다.

이 모듈은 사용자 쿼리에 대해 관련성 높은 정보를 검색하고, 
LLM(Large Language Model)을 사용하여 자연스러운 답변을 생성하는 과정을 담당합니다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple
from psycopg2.extensions import connection as PgConnection
import math

import httpx

from ..config import Settings
from .embeddings import async_embed_query
from .prompts import FOLLOWUP_PROMPT, SYSTEM_PROMPT, HYDE_PROMPT
from .retrieval import similarity_search
from . import kbo_metrics
from .entity_extractor import enhance_search_strategy
from .query_transformer import QueryTransformer, multi_query_retrieval
from .context_formatter import ContextFormatter
from ..agents.baseball_agent import BaseballStatisticsAgent

logger = logging.getLogger(__name__)


# --- Constants and Helpers ---

TEAM_MAP = {
    "KIA": "KIA 타이거즈", "기아": "KIA 타이거즈",
    "LG": "LG 트윈스",
    "두산": "두산 베어스",
    "롯데": "롯데 자이언츠",
    "삼성": "삼성 라이온즈",
    "키움": "키움 히어로즈",
    "한화": "한화 이글스",
    "KT": "KT 위즈",
    "NC": "NC 다이노스",
    "SSG": "SSG 랜더스",
}
MIN_IP_SP = 70
MIN_IP_RP = 30
MIN_PA_BATTER = 100


def _to_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0

def _get_safe_stat(meta: Dict, key: str, default: Any = None) -> Optional[float]:
    """메타 데이터에서 안전하게 실수 값을 가져옵니다."""
    val = meta.get(key)
    if val in (None, ""):
        return default
    try:
        if isinstance(val, (int, float)):
            return float(val)
        return float(str(val).strip())
    except (ValueError, TypeError):
        return default

def _get_team_name(raw_name: str) -> str:
    """팀 이름을 표준화합니다."""
    return TEAM_MAP.get(raw_name, raw_name)

def batter_rank_score(wrc_plus, war):
    """wRC+와 WAR을 가중 평균하여 타자 점수를 계산합니다."""
    wrc_plus_score = wrc_plus if wrc_plus is not None else 80
    war_score = (war * 20) if war is not None else 0 # WAR 1승당 wRC+ 20점 가치로 환산
    return 0.7 * wrc_plus_score + 0.3 * war_score


HISTORY_CONTEXT_LIMIT = 6
_RAG_CACHE_MAX = 10000
_LEAGUE_CONTEXT = kbo_metrics.LeagueContext()


def _meta_cache_key(meta: Dict[str, Any]) -> str:
    try:
        return json.dumps(meta, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(meta)


@lru_cache(maxsize=_RAG_CACHE_MAX)
def _process_stat_doc_cached(source_table: str, meta_json: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    meta = json.loads(meta_json)

    if source_table == "player_season_pitching":
        ip = _get_safe_stat(meta, "innings_pitched", 0.0)
        gs = _get_safe_stat(meta, "games_started", 0)

        role = "SP" if ip >= MIN_IP_SP or gs >= 10 else "RP"
        min_ip_threshold = MIN_IP_SP if role == "SP" else MIN_IP_RP

        if ip < min_ip_threshold:
            return None, (
                f"'{meta.get('player_name', 'N/A')}' 선수는 표본 부족(IP < {min_ip_threshold})으로 제외되었습니다."
            )

        era = _get_safe_stat(meta, "era", 99.0)
        whip = _get_safe_stat(meta, "whip", 99.0)
        k = _get_safe_stat(meta, "strikeouts")
        bb = _get_safe_stat(meta, "walks_allowed")
        hbp = _get_safe_stat(meta, "hit_batters")
        hr = _get_safe_stat(meta, "home_runs_allowed")
        pa = _get_safe_stat(meta, "tbf", 0)

        fip_val = kbo_metrics.fip(hr, bb, hbp, k, ip, _LEAGUE_CONTEXT) or 99.0
        era_minus_val = kbo_metrics.era_minus(era, _LEAGUE_CONTEXT) or 999.0
        fip_minus_val = kbo_metrics.fip_minus(fip_val, _LEAGUE_CONTEXT) or 999.0
        kbb_pct = kbo_metrics.k_minus_bb_pct(k, bb, pa) or -99.0

        score = kbo_metrics.pitcher_rank_score(era_minus_val, fip_minus_val, kbb_pct, whip, ip)

        return {
            "name": meta.get("player_name", "N/A"),
            "team": _get_team_name(meta.get("team_name", "N/A")),
            "role": role,
            "ip": ip,
            "era": era,
            "whip": whip,
            "kbb_pct": kbb_pct,
            "era_minus": era_minus_val,
            "fip_minus": fip_minus_val,
            "score": score,
        }, None

    if source_table == "player_season_batting":
        pa = int(_get_safe_stat(meta, "plate_appearances", 0) or 0)
        if pa < MIN_PA_BATTER:
            return None, (
                f"'{meta.get('player_name', 'N/A')}' 선수는 표본 부족(PA < {MIN_PA_BATTER})으로 제외되었습니다."
            )

        wrc_plus = _get_safe_stat(meta, "wrc_plus")
        ops_plus = _get_safe_stat(meta, "ops_plus")
        war = _get_safe_stat(meta, "war")
        ops_val = _get_safe_stat(meta, "ops")
        obp = _get_safe_stat(meta, "obp")
        slg = _get_safe_stat(meta, "slg")
        avg = _get_safe_stat(meta, "avg")

        hits = _to_int(meta.get("hits"))
        doubles = _to_int(meta.get("doubles"))
        triples = _to_int(meta.get("triples"))
        home_runs = _to_int(meta.get("home_runs") or meta.get("hr"))
        walks = _to_int(meta.get("walks"))
        ibb = _to_int(meta.get("intentional_walks"))
        hbp = _to_int(meta.get("hbp"))
        sf = _to_int(meta.get("sacrifice_flies"))
        ab = _to_int(meta.get("at_bats"))
        rbi = _to_int(meta.get("rbi"))
        steals = _to_int(meta.get("stolen_bases"))

        if ops_val is None:
            ops_val = kbo_metrics.ops(
                hits,
                walks,
                hbp,
                ab,
                sf,
                doubles,
                triples,
                home_runs,
            )

        league_ops = (_LEAGUE_CONTEXT.lg_OBP + _LEAGUE_CONTEXT.lg_SLG) if _LEAGUE_CONTEXT.lg_OBP and _LEAGUE_CONTEXT.lg_SLG else None
        if ops_plus is None and ops_val and league_ops:
            ops_plus = (ops_val / league_ops) * 100

        woba_val = kbo_metrics.woba(
            walks,
            ibb,
            hbp,
            hits,
            doubles,
            triples,
            home_runs,
            ab,
            sf,
            _LEAGUE_CONTEXT,
        )
        if wrc_plus is None and woba_val is not None and pa > 0:
            wrc_plus = kbo_metrics.wrc_plus(woba_val, pa, _LEAGUE_CONTEXT)

        if war is None and woba_val is not None:
            war = kbo_metrics.war_batter(
                woba_val,
                pa,
                baserunning_runs=0.0,
                fielding_runs=0.0,
                positional_runs=0.0,
                league_adj_runs=0.0,
                ctx=_LEAGUE_CONTEXT,
            )

        score = batter_rank_score(
            wrc_plus if wrc_plus is not None else 90,
            war if war is not None else 0,
        )

        return {
            "name": meta.get("player_name", "N/A"),
            "team": _get_team_name(meta.get("team_name", "N/A")),
            "pa": pa,
            "wrc_plus": wrc_plus,
            "ops_plus": ops_plus,
            "war": war,
            "ops": ops_val,
            "obp": obp,
            "slg": slg,
            "avg": avg,
            "home_runs": home_runs,
            "rbi": rbi,
            "steals": steals,
            "score": score,
            "steals": steals,
            "score": score,
        }, None

    return None, None


def _history_for_messages(history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    if not history:
        return []
    trimmed = history[-HISTORY_CONTEXT_LIMIT:]
    messages: List[Dict[str, str]] = []
    for item in trimmed:
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not content:
            continue
        text = content.strip()
        if not text:
            continue
        messages.append({"role": role, "content": text})
    return messages


def _history_context_block(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return ""
    lines: List[str] = []
    for item in history[-HISTORY_CONTEXT_LIMIT:]:
        role = "사용자" if item.get("role") == "user" else "BEGA"
        content = (item.get("content") or "").strip()
        if not content:
            continue
        snippet = content if len(content) < 400 else content[:400] + "…"
        lines.append(f"- {role}: {snippet}")
    if not lines:
        return ""
    return "### 이전 대화 맥락\n" + "\n".join(lines)


class RAGPipeline:
    """
    검색(Retrieval)과 생성(Generation)을 결합하여 답변을 생성하는 RAG 파이프라인을 관리합니다.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        connection: PgConnection,
    ) -> None:
        self.settings = settings
        self.connection = connection
        self.query_transformer = QueryTransformer(self._generate)
        self.context_formatter = ContextFormatter()
        # 야구 통계 전용 에이전트 초기화
        self.baseball_agent = BaseballStatisticsAgent(connection, self._generate)

    async def _process_and_enrich_docs(
        self, docs: List[Dict[str, Any]], year: int
    ) -> Dict[str, Any]:
        """
        검색된 문서를 필터링, 분류, 계산, 랭킹 매겨 LLM에 전달할 최종 컨텍스트를 생성합니다.
        """
        logger.info(f"[RAG] Processing {len(docs)} retrieved documents for year {year}")
        processed_pitchers = []
        processed_batters = []
        processed_games = []
        processed_awards = []
        processed_movements = []
        raw_docs = []
        warnings = set()
        filtered_playoff_count = 0

        for doc in docs:
            meta = doc.get("meta", {})
            if not meta:
                continue

            # Always keep raw doc reference
            raw_docs.append(doc)

            # Filter out non-regular season data (playoffs, etc.)
            # league = meta.get("league", "")
            # if league and league != "정규시즌":
            #     filtered_playoff_count += 1
            #     continue

            # --- Pitcher / Batter Processing (cached) ---
            if doc.get("source_table") in {"player_season_pitching", "player_season_batting"}:
                league = meta.get("league", "N/A")
                if doc.get("source_table") == "player_season_pitching":
                    logger.info(f"[RAG] Found pitcher: {meta.get('player_name')} - IP: {meta.get('innings_pitched')}, League: {league}")
                meta_key = _meta_cache_key(meta)
                processed, warning = _process_stat_doc_cached(doc.get("source_table"), meta_key)
                if warning:
                    warnings.add(warning)
                    continue
                if processed:
                    if doc.get("source_table") == "player_season_pitching":
                        processed_pitchers.append(processed)
                    else:
                        processed_batters.append(processed)
            elif doc.get("source_table") in ["game", "game_metadata", "game_inning_scores", "game_batting_stats", "game_pitching_stats"]:
                processed_games.append(doc)
            elif doc.get("source_table") == "awards":
                processed_awards.append(doc)
            elif doc.get("source_table") == "player_movements":
                processed_movements.append(doc)

        # Sort by rank score (lower is better)
        processed_pitchers.sort(key=lambda p: p["score"])
        processed_batters.sort(key=lambda b: b["score"], reverse=True)

        # Check for ambiguous players (same name in batters and pitchers)
        batter_names = {p["name"] for p in processed_batters}
        pitcher_names = {p["name"] for p in processed_pitchers}
        ambiguous_names = batter_names.intersection(pitcher_names)

        if ambiguous_names:
            for name in ambiguous_names:
                warnings.add(
                    f"주의: '{name}'은(는) 투수와 타자 모두 존재합니다. "
                    "답변 시 이 모호성을 반드시 언급하고, 가능하다면 포지션(투수/타자)을 명시하여 혼동을 피하세요."
                )

        logger.info(f"[RAG] Filtered {filtered_playoff_count} playoff records")
        logger.info(f"[RAG] Final processed: {len(processed_pitchers)} pitchers, {len(processed_batters)} batters")

        return {
            "pitchers": processed_pitchers,
            "batters": processed_batters,
            "games": processed_games,
            "awards": processed_awards,
            "movements": processed_movements,
            "raw_docs": raw_docs,
            "warnings": list(warnings),
            "context": _LEAGUE_CONTEXT,
        }

    async def retrieve(
        self,
        query: str,
        *,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        hyde_prompt = HYDE_PROMPT.format(question=query)
        hyde_messages = [{"role": "user", "content": hyde_prompt}]

        try:
            hypothetical_document = await self._generate(hyde_messages)
            search_query = hypothetical_document
        except Exception:
            search_query = query

        limit = limit or self.settings.default_search_limit
        embedding = await async_embed_query(search_query, self.settings)
        if not embedding:
            return []

        keyword = query
        from fastapi.concurrency import run_in_threadpool

        docs = await run_in_threadpool(
            similarity_search,
            self.connection,
            embedding,
            limit=limit,
            filters=filters,
            keyword=keyword,
        )
        return docs

    async def retrieve_with_multi_query(
        self,
        query: str,
        entity_filter,
        *,
        filters: Optional[Dict[str, Any]] = None,
        use_llm_expansion: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Multi-query retrieval을 사용하여 검색 품질을 향상시킵니다.
        여러 쿼리 변형으로 검색하고 결과를 결합합니다.
        """
        logger.info(f"[RAG] Multi-query retrieval for: {query}")
        
        # 규칙 기반 쿼리 확장
        query_variations = self.query_transformer.expand_query_with_rules(query, entity_filter)
        
        # LLM 기반 쿼리 확장 (선택적)
        if use_llm_expansion and len(query_variations) < 3:
            try:
                llm_variations = await self.query_transformer.llm_expand_query(query)
                query_variations.extend(llm_variations)
            except Exception as e:
                logger.warning(f"[RAG] LLM query expansion failed: {e}")
        
        # Multi-query retrieval 수행
        docs = await multi_query_retrieval(
            query_variations,
            self.retrieve,
            filters or {},
            limit_per_query=8
        )
        
        logger.info(f"[RAG] Multi-query retrieval returned {len(docs)} documents")
        return docs

    async def _generate(self, messages: Sequence[Dict[str, str]]) -> str:
        provider = self.settings.llm_provider
        
        try:
            if provider == "gemini":
                return await self._generate_with_gemini(messages)
            elif provider == "openrouter":
                return await self._generate_with_openrouter(messages)
            else:
                raise RuntimeError(f"지원되지 않는 LLM 공급자: {provider}")
        except Exception as e:
            logger.error(f"[RAG] Primary LLM provider '{provider}' failed: {e}")
            
            # Try fallback provider
            fallback_provider = "gemini" if provider == "openrouter" else "openrouter"
            
            # Check if fallback is available
            if fallback_provider == "gemini" and self.settings.gemini_api_key:
                logger.info(f"[RAG] Attempting fallback to Gemini")
                try:
                    return await self._generate_with_gemini(messages)
                except Exception as fallback_e:
                    logger.error(f"[RAG] Fallback to Gemini also failed: {fallback_e}")
            elif fallback_provider == "openrouter" and self.settings.openrouter_api_key:
                logger.info(f"[RAG] Attempting fallback to OpenRouter")
                try:
                    return await self._generate_with_openrouter(messages)
                except Exception as fallback_e:
                    logger.error(f"[RAG] Fallback to OpenRouter also failed: {fallback_e}")
            
            # All providers failed
            raise RuntimeError(f"모든 LLM 제공자가 실패했습니다. 주 제공자({provider}): {e}")

    async def _generate_with_openrouter(self, messages: Sequence[Dict[str, str]], max_retries: int = 3) -> str:
        if not self.settings.openrouter_api_key:
            raise RuntimeError("OpenRouter를 사용하려면 OPENROUTER_API_KEY가 필요합니다.")
        
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.settings.openrouter_referer or "",
            "X-Title": self.settings.openrouter_app_title or "",
        }
        payload = {
            "model": self.settings.openrouter_model,
            "messages": list(messages),
            "max_tokens": self.settings.max_output_tokens,
        }

        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Exponential backoff with jitter
                if attempt > 0:
                    base_delay = 2 ** (attempt - 1)  # 1, 2, 4 seconds
                    jitter = random.uniform(0.1, 0.5)  # Add randomness to prevent thundering herd
                    delay = base_delay + jitter
                    logger.info(f"[OpenRouter] Retry attempt {attempt + 1}/{max_retries}, waiting {delay:.2f}s")
                    await asyncio.sleep(delay)
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{self.settings.openrouter_base_url.rstrip('/')}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                
                response.raise_for_status()
                data = response.json()
                
                # 디버깅을 위한 응답 로깅 (첫 번째 시도에서만)
                if attempt == 0:
                    logger.info(f"[OpenRouter] Response status: {response.status_code}")
                    logger.debug(f"[OpenRouter] Response data keys: {list(data.keys())}")
                
                choices = data.get("choices", [])
                if not choices:
                    error_msg = f"OpenRouter 응답에 choices가 없습니다. Keys: {list(data.keys())}"
                    logger.error(f"[OpenRouter] {error_msg}")
                    raise RuntimeError(error_msg)
                
                message = choices[0].get("message", {})
                content = message.get("content", "")
                
                if not content:
                    error_msg = f"OpenRouter 응답이 비어 있습니다. Message keys: {list(message.keys())}"
                    logger.error(f"[OpenRouter] {error_msg}")
                    raise RuntimeError(error_msg)
                
                logger.info(f"[OpenRouter] Successfully generated response on attempt {attempt + 1}")
                return content
                
            except (httpx.RequestError, httpx.HTTPStatusError, RuntimeError) as e:
                last_exception = e
                logger.warning(f"[OpenRouter] Attempt {attempt + 1}/{max_retries} failed: {e}")
                
                # If it's the last attempt, don't wait
                if attempt == max_retries - 1:
                    break
        
        # All retries failed
        logger.error(f"[OpenRouter] All {max_retries} attempts failed. Last error: {last_exception}")
        raise RuntimeError(f"OpenRouter API 호출이 {max_retries}번 모두 실패했습니다. 마지막 오류: {last_exception}")

    async def _generate_with_gemini(self, messages: Sequence[Dict[str, str]], max_retries: int = 3) -> str:
        """Google Gemini API를 사용하여 응답을 생성합니다."""
        if not self.settings.gemini_api_key:
            raise RuntimeError("Gemini를 사용하려면 GEMINI_API_KEY가 필요합니다.")
        
        # Convert OpenAI format messages to Gemini format
        gemini_contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # Gemini doesn't have system role, prepend to first user message
                if not gemini_contents:
                    gemini_contents.append({
                        "role": "user", 
                        "parts": [{"text": f"System: {content}\n\nUser: "}]
                    })
                else:
                    # Prepend to existing user message
                    if gemini_contents[-1]["role"] == "user":
                        gemini_contents[-1]["parts"][0]["text"] = f"System: {content}\n\n" + gemini_contents[-1]["parts"][0]["text"]
            elif role == "user":
                gemini_contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                gemini_contents.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
        
        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "maxOutputTokens": self.settings.max_output_tokens,
                "temperature": 0.7,
            }
        }
        
        model = self.settings.gemini_model or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
        params = {"key": self.settings.gemini_api_key}
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Exponential backoff with jitter
                if attempt > 0:
                    base_delay = 2 ** (attempt - 1)
                    jitter = random.uniform(0.1, 0.5)
                    delay = base_delay + jitter
                    logger.info(f"[Gemini] Retry attempt {attempt + 1}/{max_retries}, waiting {delay:.2f}s")
                    await asyncio.sleep(delay)
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, json=payload, params=params)
                
                response.raise_for_status()
                data = response.json()
                
                # 디버깅을 위한 응답 로깅 (첫 번째 시도에서만)
                if attempt == 0:
                    logger.info(f"[Gemini] Response status: {response.status_code}")
                    logger.debug(f"[Gemini] Response data keys: {list(data.keys())}")
                
                candidates = data.get("candidates", [])
                if not candidates:
                    error_msg = f"Gemini 응답에 candidates가 없습니다. Keys: {list(data.keys())}"
                    logger.error(f"[Gemini] {error_msg}")
                    raise RuntimeError(error_msg)
                
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                
                if not parts or not parts[0].get("text"):
                    error_msg = f"Gemini 응답이 비어 있습니다. Content: {content}"
                    logger.error(f"[Gemini] {error_msg}")
                    raise RuntimeError(error_msg)
                
                result = parts[0]["text"]
                logger.info(f"[Gemini] Successfully generated response on attempt {attempt + 1}")
                return result
                
            except (httpx.RequestError, httpx.HTTPStatusError, RuntimeError) as e:
                last_exception = e
                logger.warning(f"[Gemini] Attempt {attempt + 1}/{max_retries} failed: {e}")
                
                if attempt == max_retries - 1:
                    break
        
        # All retries failed
        logger.error(f"[Gemini] All {max_retries} attempts failed. Last error: {last_exception}")
        raise RuntimeError(f"Gemini API 호출이 {max_retries}번 모두 실패했습니다. 마지막 오류: {last_exception}")

    def _is_statistical_query(self, query: str, entity_filter) -> bool:
        """
        질문이 구체적인 통계 조회인지 판단합니다.
        통계 질문인 경우 야구 에이전트를 우선 사용해야 합니다.
        """
        # 일반 대화 키워드 (이런 질문들은 통계 질문이 아님)
        chitchat_keywords = [
            "안녕", "누구", "좋아해", "응원", "날씨", "어때", "뭐해", "어디",
            "언제", "왜", "어떻게", "고마워", "미안", "반가워", "잘가",
            "소개", "설명", "도움", "기능", "사용법"
        ]
        
        # 통계 관련 키워드
        statistical_keywords = [
            "타율", "홈런", "타점", "득점", "ops", "era", "방어율", "whip", 
            "승", "패", "세이브", "홀드", "삼진", "볼넷", "출루율", "장타율",
            "wrc+", "war", "fip", "babip", "몇위", "순위", "1위", "최고", 
            "상위", "리더", "기록", "통계", "성적", "몇개", "몇점", "얼마나",
            "vs", "대", "비교", "누가", "더", "뛰어난", "우수한", "좋은", "맞대결"
        ]
        
        query_lower = query.lower()
        
        # 1단계: 일반 대화인지 확인 (우선순위 높음)
        is_chitchat = any(keyword in query_lower for keyword in chitchat_keywords)
        if is_chitchat and not any(keyword in query_lower for keyword in statistical_keywords):
            return False  # 일반 대화이므로 통계 질문 아님
        
        # 2단계: 통계 키워드 확인
        has_stat_keywords = any(keyword in query_lower for keyword in statistical_keywords)
        
        # 3단계: 구체적인 데이터 요청인지 확인
        has_specific_request = (
            entity_filter.player_name or 
            entity_filter.stat_type or 
            "년" in query or
            any(word in query_lower for word in ["알려줘", "궁금", "얼마", "몇", "는", "은", "의"]) or
            # 질문 형태나 통계 지표가 있으면 통계 질문으로 간주
            "?" in query or "몇" in query or "어떻게" in query or
            any(stat in query_lower for stat in ["타율", "홈런", "ops", "era", "방어율"])
        )
        
        # 디버깅을 위한 로그 출력
        logger.info(f"[RAG] _is_statistical_query debug:")
        logger.info(f"  query: {query}")
        logger.info(f"  has_stat_keywords: {has_stat_keywords}")
        logger.info(f"  has_specific_request: {has_specific_request}")
        logger.info(f"  entity_filter.player_name: {entity_filter.player_name}")
        logger.info(f"  entity_filter.stat_type: {entity_filter.stat_type}")
        
        # 통계 키워드가 있고 구체적인 요청이면 통계 질문
        result = has_stat_keywords and has_specific_request
        logger.info(f"  RESULT: {result}")
        return result
    
    def _is_regulation_query(self, query: str) -> bool:
        """
        질문이 KBO 규정 관련인지 판단합니다.
        """
        regulation_keywords = [
            "규정", "규칙", "룰", "조항", "가능해", "허용", "금지",
            "벌칙", "징계", "반칙", "파울", "아웃", "세이프",
            "스트라이크", "볼", "홈런", "인플레이", "타이브레이크",
            "지명타자", "연장전", "콜드게임", "더블헤더", "비디오판독",
            "FA", "자유계약", "외국인선수", "몇명까지", "드래프트", "트레이드",
            "도박", "폭력", "약물", "심판", "모독", "퇴장",
            "플레이오프", "포스트시즌", "와일드카드", "한국시리즈", "몇팀", 
            "보크", "방해", "인필드플라이", "그라운드룰", "몸에맞는공",
            "용어", "뜻", "의미", "정의", "설명", "조건", "언제적용",
            "세이브조건", "승리투수", "왕", "홈런왕", "득점왕", "순위",
            "wrc+", "ops", "era", "whip", "babip", "war", "fip"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in regulation_keywords)

    def _is_game_query(self, query: str) -> bool:
        """
        질문이 경기 데이터 관련인지 판단합니다.
        """
        game_keywords = [
            "경기", "게임", "박스스코어", "스코어", "결과", "이닝별",
            "오늘", "어제", "내일", "날짜", "언제", "몇일", "며칠",
            "vs", "대", "맞대결", "직접대결", "상대전적", "시리즈",
            "승부", "이겼", "졌", "무승부", "점수", "승리", "패배",
            "홈", "원정", "away", "home", "구장에서", "에서",
            "몇점", "득점", "실점", "타점", "안타", "홈런친",
            "투구", "선발", "등판", "세이브", "홀드", "승", "패"
        ]
        
        # 날짜 패턴 확인 (YYYY-MM-DD, MM/DD, 월일 등)
        date_patterns = [
            r"\d{4}-\d{1,2}-\d{1,2}",  # 2025-10-15
            r"\d{1,2}/\d{1,2}",        # 10/15
            r"\d{1,2}월\s*\d{1,2}일",   # 10월 15일
            r"오늘|어제|내일|모레|그저께"
        ]
        
        query_lower = query.lower()
        
        # 키워드 매칭
        has_game_keywords = any(keyword in query_lower for keyword in game_keywords)
        
        # 날짜 패턴 매칭
        import re
        has_date_pattern = any(re.search(pattern, query_lower) for pattern in date_patterns)
        
        # 팀 vs 팀 패턴
        team_vs_pattern = r"(KIA|기아|LG|두산|롯데|삼성|키움|한화|KT|NC|SSG).*(vs|대|vs\.|대전|맞대결).*(KIA|기아|LG|두산|롯데|삼성|키움|한화|KT|NC|SSG)"
        has_team_vs_pattern = bool(re.search(team_vs_pattern, query, re.IGNORECASE))
        
        return has_game_keywords or has_date_pattern or has_team_vs_pattern

    def _is_general_conversation(self, query: str) -> bool:
        """
        일반 대화인지 판단합니다.
        """
        # 야구 지식/용어 관련 질문들 + 통계 질문 키워드들
        baseball_keywords = [
            "ops", "wrc+", "war", "era", "whip", "babip", "fip", "골든글러브",
            "fa", "신인왕", "mvp", "타율", "방어율", "출루율", "장타율",
            "자책점", "세이브", "홀드", "승리투수", "뜻", "의미", "정의",
            "계산", "어떻게", "무엇", "기준",
            # 통계 질문 키워드 추가
            "홈런", "타점", "득점", "승", "패", "삼진", "볼넷", "몇위", "순위",
            "1위", "최고", "상위", "리더", "기록", "통계", "성적", "몇개", "몇점",
            "얼마나", "얼마", "vs", "대", "비교", "누가", "더", "뛰어난", "우수한",
            "좋은", "맞대결", "시즌", "년", "연도"
        ]
        
        # 일반 대화 키워드 (야구와 무관한 것들)
        general_keywords = [
            "안녕", "누구", "좋아해", "응원", "날씨", "어때", "뭐해", 
            "고마워", "미안", "반가워", "잘가", "소개", "도움",
            "기능", "사용법"
        ]
        
        query_lower = query.lower()
        
        # 야구 관련 질문이면 일반 대화가 아님  
        if any(keyword in query_lower for keyword in baseball_keywords):
            return False  # 야구 관련 질문이므로 일반 대화가 아님
        
        # 야구/통계와 무관한 일반적인 대화만 일반 대화로 분류
        return any(keyword in query_lower for keyword in general_keywords)

    async def _try_agent_first(
        self,
        query: str,
        *,
        intent: str = "freeform", 
        filters: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        야구 에이전트를 우선 사용하여 통계 질문을 처리합니다.
        에이전트가 실패하면 기존 RAG 방식으로 폴백합니다.
        """
        logger.info(f"[RAG] Attempting agent-first approach for: {query}")
        
        try:
            # 야구 에이전트를 통한 처리 시도
            agent_result = await self.baseball_agent.process_query(query, {
                "intent": intent,
                "filters": filters,
                "history": history
            })
            
            if agent_result["verified"] and not agent_result.get("error"):
                logger.info(f"[RAG] Agent successfully handled query with verified data")
                return {
                    "answer": agent_result["answer"],
                    "citations": [],  # 에이전트는 DB 직접 조회하므로 citations 불필요
                    "intent": intent,
                    "retrieved": agent_result.get("tool_results", []),
                    "strategy": "verified_agent",
                    "verified": True,
                    "tool_calls": agent_result.get("tool_calls", []),
                    "data_sources": agent_result.get("data_sources", [])
                }
            else:
                logger.warning(f"[RAG] Agent failed or returned unverified data: {agent_result.get('error')}")
                return None  # 폴백 신호
                
        except Exception as e:
            logger.error(f"[RAG] Agent processing error: {e}")
            return None  # 폴백 신호

    async def _handle_general_conversation(self, query: str) -> Dict[str, Any]:
        """
        일반 대화를 처리합니다.
        """
        logger.info(f"[RAG] Handling general conversation: {query}")
        
        # 야구 지식 관련 질문 처리
        knowledge_responses = {
            "ops": "OPS는 출루율(OBP)과 장타율(SLG)을 더한 값입니다.\n- 계산법: OPS = 출루율 + 장타율\n- 좋은 OPS: 0.800 이상\n- 뛰어난 OPS: 0.900 이상\n- 최고 수준 OPS: 1.000 이상\n\nOPS는 타자의 종합적인 공격력을 나타내는 대표적인 지표입니다.",
            "wrc+": "wRC+는 가중출루율을 기반으로 한 공격력 지표입니다.\n- 100이 평균 (리그 평균 대비 100%)\n- 120이면 리그 평균보다 20% 우수\n- 80이면 리그 평균보다 20% 부족\n\nwRC+는 볼파크와 리그 환경을 보정한 정확한 공격력 지표입니다.",
            "war": "WAR(Wins Above Replacement)는 대체 선수 대비 승수 기여도입니다.\n- WAR 2: 평균 주전급\n- WAR 5: 올스타급\n- WAR 8+: MVP급\n\nWAR은 선수의 종합적인 가치를 하나의 숫자로 나타내는 가장 포괄적인 지표입니다.",
            "era": "ERA(자책점평균)는 투수가 9이닝당 내주는 자책점 수입니다.\n- 계산법: (자책점 × 9) ÷ 투구이닝\n- 좋은 ERA: 4.00 미만\n- 뛰어난 ERA: 3.00 미만\n- 최고 수준 ERA: 2.50 미만",
            "whip": "WHIP는 투수가 이닝당 내주는 안타와 볼넷의 합계입니다.\n- 계산법: (피안타 + 볼넷) ÷ 투구이닝\n- 좋은 WHIP: 1.30 미만\n- 뛰어난 WHIP: 1.20 미만\n- 최고 수준 WHIP: 1.10 미만",
            "골든글러브": "KBO 골든글러브는 각 포지션별 최고의 수비수에게 수여되는 상입니다.\n- 선정 방식: 기자단 투표\n- 대상: 각 포지션별 1명 (포수, 1루수, 2루수, 3루수, 유격수, 외야수 3명, 지명타자)\n- 기준: 수비율, 범위, 송구력 등 종합적인 수비 능력\n- 최소 출전: 규정 이닝의 2/3 이상",
            "fa": "FA(자유계약선수)는 팀을 자유롭게 선택할 수 있는 선수입니다.\n- 자격 조건: 프로 경력 9년 이상 (2015년부터 8년으로 단축)\n- 권리: 어떤 팀과도 자유롭게 계약 가능\n- 보상: FA 영입팀은 원소속팀에게 보상선수 제공"
        }
        
        # 간단한 대화 응답 패턴
        conversation_responses = {
            "안녕": "안녕하세요! 저는 KBO 리그 데이터 분석가 BEGA입니다. KBO 야구 통계에 대해 궁금한 것이 있으시면 언제든 물어보세요!",
            "누구": "저는 KBO 리그 전문 데이터 분석가 'BEGA'입니다. 한국 프로야구의 각종 통계와 기록을 정확하게 분석해드립니다.",
            "좋아": "네, 저는 야구를 정말 좋아합니다! 특히 KBO 리그의 흥미진진한 경기와 선수들의 기록을 분석하는 것이 제 전문 분야입니다.",
            "응원": "저는 모든 KBO 팀을 공정하게 분석합니다! 어떤 팀을 응원하시든 정확하고 객관적인 데이터를 제공해드릴게요.",
            "날씨": "죄송하지만 날씨 정보는 제공하지 않습니다. 저는 KBO 야구 통계 전문 분석가입니다. 야구 관련 질문이 있으시면 언제든 물어보세요!",
            "도움": "저는 다음과 같은 도움을 드릴 수 있습니다:\n- 선수 개인 통계 조회\n- 팀별 순위 및 기록 분석\n- 야구 지표 설명\n- KBO 리그 역사적 기록 비교\n\n궁금한 야구 통계가 있으시면 언제든 말씀해주세요!",
            "기능": "제 주요 기능은 다음과 같습니다:\n1. 선수 개인 성적 분석 (타율, 홈런, ERA 등)\n2. 팀 순위 및 리더보드 조회\n3. 고급 야구 지표 계산 및 설명\n4. 시즌별, 연도별 기록 비교\n\n구체적인 야구 통계 질문을 해보세요!"
        }
        
        query_lower = query.lower()
        
        # 야구 지식 질문 먼저 확인 (우선순위 높음)
        for keyword, response in knowledge_responses.items():
            if keyword in query_lower:
                return {
                    "answer": response,
                    "citations": [],
                    "intent": "knowledge_explanation",
                    "retrieved": [],
                    "strategy": "knowledge_handler",
                    "verified": True
                }
        
        # 일반 대화 키워드 확인
        for keyword, response in conversation_responses.items():
            if keyword in query_lower:
                return {
                    "answer": response,
                    "citations": [],
                    "intent": "general_conversation",
                    "retrieved": [],
                    "strategy": "conversation_handler",
                    "verified": True
                }
        
        # 기본 응답 (아무 키워드도 매칭되지 않을 때만)
        logger.info(f"[RAG] _handle_general_conversation fallback for query: {query}")
        default_response = """안녕하세요! 저는 KBO 리그 데이터 분석가 'BEGA'입니다. 

KBO 야구와 관련된 다음과 같은 질문들을 도와드릴 수 있습니다:
- "2025년 김도영 타율은?" 
- "홈런왕 TOP 5는?"
- "LG 트윈스 주요 선수는?"
- "OPS가 뭐야?"

야구 통계에 대해 궁금한 것이 있으시면 언제든 물어보세요!"""
        
        return {
            "answer": default_response,
            "citations": [],
            "intent": "general_conversation", 
            "retrieved": [],
            "strategy": "conversation_handler",
            "verified": True
        }

    async def run(
        self,
        query: str,
        *,
        intent: str = "freeform",
        filters: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        # 1. Enhanced Entity Extraction and Search Strategy
        logger.info(f"[RAG] Processing query: {query}")
        
        # Extract entities and enhance search strategy
        search_strategy = enhance_search_strategy(query)
        entity_filter = search_strategy["entity_filter"]
        extracted_filters = search_strategy["db_filters"]
        
        # 2. 통계 질문인지 먼저 확인 (최우선)
        is_statistical = self._is_statistical_query(query, entity_filter)
        logger.info(f"[RAG] Is statistical query: {is_statistical}")
        logger.info(f"[RAG] Entity filter: {entity_filter}")
        if is_statistical:
            logger.info(f"[RAG] Statistical query detected, using traditional RAG directly")
            # 통계 질문이면 바로 RAG로 처리 (에이전트 건너뛰기)
            pass  # 6단계로 진행
        
        # 3. 일반 대화인지 확인
        elif self._is_general_conversation(query):
            logger.info(f"[RAG] General conversation detected")
            return await self._handle_general_conversation(query)
        
        # 4. 규정 질문인지 확인
        elif self._is_regulation_query(query):
            logger.info(f"[RAG] Regulation query detected, trying agent first")
            agent_result = await self._try_agent_first(query, intent=intent, filters=filters, history=history)
            if agent_result is not None:
                return agent_result
            else:
                logger.info(f"[RAG] Regulation agent failed, falling back to traditional RAG")
        
        # 5. 경기 데이터 질문인지 확인
        elif self._is_game_query(query):
            logger.info(f"[RAG] Game query detected, trying agent first")
            agent_result = await self._try_agent_first(query, intent=intent, filters=filters, history=history)
            if agent_result is not None:
                return agent_result
            else:
                logger.info(f"[RAG] Game agent failed, falling back to traditional RAG")
        
        # 6. 기존 RAG 방식으로 폴백 또는 일반 질문 처리
        
        # Merge user-provided filters with extracted filters
        # User-provided filters take precedence
        final_filters = {**extracted_filters, **(filters or {})}
        
        # Determine year for analysis
        year = final_filters.get("season_year") or entity_filter.season_year or 2025
        logger.info(f"[RAG] Analysis year: {year}")
        logger.info(f"[RAG] Final filters: {final_filters}")
        
        # 2. Intelligent Multi-Strategy Retrieval
        docs = []
        
        if search_strategy["is_ranking_query"]:
            logger.info("[RAG] Ranking query detected - using multi-query retrieval")
            
            # For ranking queries, use multi-query retrieval for better coverage
            if not entity_filter.position_type:
                # Search both pitchers and batters with multi-query
                pitcher_filters = dict(final_filters)
                pitcher_filters["source_table"] = "player_season_pitching"
                docs_pitchers = await self.retrieve_with_multi_query(
                    query, entity_filter, filters=pitcher_filters
                )
                
                batter_filters = dict(final_filters)
                batter_filters["source_table"] = "player_season_batting"
                docs_batters = await self.retrieve_with_multi_query(
                    query, entity_filter, filters=batter_filters
                )
                
                docs = docs_pitchers + docs_batters
                logger.info(f"[RAG] Multi-query ranking search: {len(docs_pitchers)} pitchers + {len(docs_batters)} batters")
            else:
                # Position-specific multi-query search
                docs = await self.retrieve_with_multi_query(
                    query, entity_filter, filters=final_filters
                )
        
        elif entity_filter.player_name:
            logger.info(f"[RAG] Player-specific query: {entity_filter.player_name}")
            # For specific player queries, use multi-query with relaxed filters
            player_filters = dict(final_filters)
            player_filters.pop("source_table", None)  # Remove source_table filter
            docs = await self.retrieve_with_multi_query(
                query, entity_filter, filters=player_filters, use_llm_expansion=True
            )
            
        else:
            logger.info("[RAG] General search strategy with multi-query")
            # Use multi-query for general searches to improve coverage
            docs = await self.retrieve_with_multi_query(
                query, entity_filter, filters=final_filters
            )
        
        # 3. Fallback Strategy
        if not docs and final_filters:
            logger.info("[RAG] No results with filters, attempting fallback search")
            # Remove restrictive filters one by one
            fallback_filters = dict(final_filters)
            
            # Try removing source_table first
            if "source_table" in fallback_filters:
                fallback_filters.pop("source_table")
                docs = await self.retrieve(query, filters=fallback_filters, limit=20)
                logger.info(f"[RAG] Fallback without source_table: {len(docs)} docs")
            
            # If still no results, try without team filter
            if not docs and "team_id" in fallback_filters:
                fallback_filters.pop("team_id")
                docs = await self.retrieve(query, filters=fallback_filters, limit=20)
                logger.info(f"[RAG] Fallback without team filter: {len(docs)} docs")
            
            # Final fallback: only keep year and league filters
            if not docs:
                minimal_filters = {}
                if "season_year" in final_filters:
                    minimal_filters["season_year"] = final_filters["season_year"]
                if "meta.league" in final_filters:
                    minimal_filters["meta.league"] = final_filters["meta.league"]
                docs = await self.retrieve(query, filters=minimal_filters, limit=25)
                logger.info(f"[RAG] Minimal fallback: {len(docs)} docs")
        
        logger.info(f"[RAG] Final retrieval result: {len(docs)} documents")

        # 2. 데이터 처리 및 보강
        processed_data = await self._process_and_enrich_docs(docs, year)
        
        # 3. 의도별 컨텍스트 생성 (새로운 컨텍스트 포맷터 사용)
        # TEMP: 디버깅을 위해 raw 검색 결과도 컨텍스트에 포함
        raw_context_parts = []
        for doc in docs[:10]:  # 상위 10개 결과만 사용
            title = doc.get("title", "제목 없음")
            content = doc.get("content", "")[:200]  # 내용 200자 제한
            raw_context_parts.append(f"- {title}: {content}")
        
        raw_context = "\n### 검색된 원본 데이터:\n" + "\n".join(raw_context_parts)
        
        formatted_context = self.context_formatter.format_context(
            processed_data, intent, query, entity_filter, year
        )
        
        # 원본 데이터도 포함
        formatted_context = formatted_context + "\n\n" + raw_context
        
        # 대화 기록 컨텍스트 추가
        history_block = _history_context_block(history)
        if history_block:
            formatted_context = history_block + "\n\n" + formatted_context

        # 4. LLM 프롬프트 구성
        prompt = FOLLOWUP_PROMPT.format(question=query, context=formatted_context)
        
        # DEBUG: 컨텍스트 로깅
        logger.info(f"[RAG_DEBUG] Question: {query}")
        logger.info(f"[RAG_DEBUG] Formatted context length: {len(formatted_context)}")
        logger.info(f"[RAG_DEBUG] Formatted context preview: {formatted_context[:500]}...")
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(_history_for_messages(history))
        messages.append({"role": "user", "content": prompt})
        
        # 5. LLM을 호출하여 답변 생성
        answer = await self._generate(messages)

        # 6. 최종 결과 구성
        return {
            "answer": answer,
            "citations": [{"id": doc["id"], "title": doc.get("title", "")} for doc in docs],
            "intent": intent,
            "retrieved": docs,
            "strategy": "rag_v3_enhanced",  # 업데이트된 버전 명시
            "entity_filter": {
                "season_year": entity_filter.season_year,
                "team_id": entity_filter.team_id,
                "player_name": entity_filter.player_name,
                "stat_type": entity_filter.stat_type,
                "position_type": entity_filter.position_type,
            }
        }
