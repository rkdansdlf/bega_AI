"""
RAG (Retrieval-Augmented Generation) 파이프라인의 핵심 로직을 구현한 모듈입니다.

이 모듈은 사용자 쿼리에 대해 관련성 높은 정보를 검색하고, 
LLM(Large Language Model)을 사용하여 자연스러운 답변을 생성하는 과정을 담당합니다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence
from psycopg2.extensions import connection as PgConnection
import math

import httpx

from ..config import Settings
from .embeddings import async_embed_texts
from .prompts import FOLLOWUP_PROMPT, SYSTEM_PROMPT, HYDE_PROMPT
from .retrieval import similarity_search
from .tools import try_answer_with_sql
from . import kbo_metrics

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

    async def _process_and_enrich_docs(
        self, docs: List[Dict[str, Any]], year: int
    ) -> Dict[str, Any]:
        """
        검색된 문서를 필터링, 분류, 계산, 랭킹 매겨 LLM에 전달할 최종 컨텍스트를 생성합니다.
        """
        logger.info(f"[RAG] Processing {len(docs)} retrieved documents for year {year}")
        ctx = kbo_metrics.LeagueContext()
        processed_pitchers = []
        processed_batters = []
        warnings = set()
        filtered_playoff_count = 0

        for doc in docs:
            meta = doc.get("meta", {})
            if not meta:
                continue

            # Filter out non-regular season data (playoffs, etc.)
            league = meta.get("league", "")
            if league and league != "정규시즌":
                filtered_playoff_count += 1
                continue

            # --- Pitcher Processing ---
            if doc.get("source_table") == "player_season_pitching":
                ip = _get_safe_stat(meta, "innings_pitched", 0.0)
                gs = _get_safe_stat(meta, "games_started", 0)
                logger.info(f"[RAG] Found pitcher: {meta.get('player_name')} - IP: {ip}, League: {league}")
                
                role = "SP" if ip >= MIN_IP_SP or gs >= 10 else "RP"
                min_ip_threshold = MIN_IP_SP if role == "SP" else MIN_IP_RP

                if ip < min_ip_threshold:
                    warnings.add(f"'{meta.get('player_name', 'N/A')}' 선수는 표본 부족(IP < {min_ip_threshold})으로 제외되었습니다.")
                    continue

                # Calculate metrics
                era = _get_safe_stat(meta, "era", 99.0)
                whip = _get_safe_stat(meta, "whip", 99.0)
                k = _get_safe_stat(meta, "strikeouts")
                bb = _get_safe_stat(meta, "walks_allowed")
                hbp = _get_safe_stat(meta, "hit_batters")
                hr = _get_safe_stat(meta, "home_runs_allowed")
                pa = _get_safe_stat(meta, "tbf", 0)

                fip_val = kbo_metrics.fip(hr, bb, hbp, k, ip, ctx) or 99.0
                era_minus_val = kbo_metrics.era_minus(era, ctx) or 999.0
                fip_minus_val = kbo_metrics.fip_minus(fip_val, ctx) or 999.0
                kbb_pct = kbo_metrics.k_minus_bb_pct(k, bb, pa) or -99.0
                
                score = kbo_metrics.pitcher_rank_score(era_minus_val, fip_minus_val, kbb_pct, whip, ip)

                processed_pitchers.append({
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
                })
            elif doc.get("source_table") == "player_season_batting":
                pa = int(_get_safe_stat(meta, "plate_appearances", 0) or 0)
                if pa < MIN_PA_BATTER:
                    warnings.add(
                        f"'{meta.get('player_name', 'N/A')}' 선수는 표본 부족(PA < {MIN_PA_BATTER})으로 제외되었습니다."
                    )
                    continue

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

                league_ops = (ctx.lg_OBP + ctx.lg_SLG) if ctx.lg_OBP and ctx.lg_SLG else None
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
                    ctx,
                )
                if wrc_plus is None and woba_val is not None and pa > 0:
                    wrc_plus = kbo_metrics.wrc_plus(woba_val, pa, ctx)

                if war is None and woba_val is not None:
                    war = kbo_metrics.war_batter(
                        woba_val,
                        pa,
                        baserunning_runs=0.0,
                        fielding_runs=0.0,
                        positional_runs=0.0,
                        league_adj_runs=0.0,
                        ctx=ctx,
                    )

                score = batter_rank_score(
                    wrc_plus if wrc_plus is not None else 90,
                    war if war is not None else 0,
                )

                processed_batters.append({
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
                })

        # Sort by rank score (lower is better)
        processed_pitchers.sort(key=lambda p: p["score"])
        processed_batters.sort(key=lambda b: b["score"], reverse=True)

        logger.info(f"[RAG] Filtered {filtered_playoff_count} playoff records")
        logger.info(f"[RAG] Final processed: {len(processed_pitchers)} pitchers, {len(processed_batters)} batters")

        return {
            "pitchers": processed_pitchers,
            "batters": processed_batters, # Placeholder for batter logic
            "warnings": list(warnings),
            "context": ctx,
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
        embeddings = await async_embed_texts([search_query], self.settings)
        if not embeddings:
            return []

        keyword = query if len(query.split()) <= 5 else None
        docs = similarity_search(
            self.connection,
            embeddings[0],
            limit=limit,
            filters=filters,
            keyword=keyword,
        )
        return docs

    async def _generate(self, messages: Sequence[Dict[str, str]]) -> str:
        provider = self.settings.llm_provider
        if provider == "gemini":
            # (Implementation for Gemini - not shown for brevity)
            pass
        elif provider == "openrouter":
            return await self._generate_with_openrouter(messages)
        raise RuntimeError(f"지원되지 않는 LLM 공급자: {provider}")

    async def _generate_with_openrouter(self, messages: Sequence[Dict[str, str]]) -> str:
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

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.settings.openrouter_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
        
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("OpenRouter 응답이 비어 있습니다.")
        return content

    async def run(
        self,
        query: str,
        *,
        intent: str = "freeform",
        filters: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        # 1. 관련 문서 검색
        filters = filters or {}
        year = filters.get("season_year") or filters.get("year", 2025)

        # Auto-detect if question is about season stats (not game-specific)
        season_keywords = ["시즌", "최고", "순위", "랭킹", "1위", "제일", "가장"]
        is_season_query = any(keyword in query for keyword in season_keywords)

        logger.info(f"[RAG] Query: {query}, is_season_query: {is_season_query}")

        # For season queries, prioritize season stat tables
        if is_season_query and "source_table" not in filters:
            logger.info("[RAG] Season query detected, searching season stats first")
            # Try season stats first, fallback to all if no results
            season_filters = dict(filters)
            # Filter for season stats only and regular season
            season_filters["meta.league"] = "정규시즌"

            # Try pitchers first
            pitcher_filters = dict(season_filters)
            pitcher_filters["source_table"] = "player_season_pitching"
            logger.info(f"[RAG] Searching pitchers with filters: {pitcher_filters}")
            docs_pitchers = await self.retrieve(query, filters=pitcher_filters, limit=15)
            logger.info(f"[RAG] Found {len(docs_pitchers)} pitcher docs")

            # Try batters
            batter_filters = dict(season_filters)
            batter_filters["source_table"] = "player_season_batting"
            logger.info(f"[RAG] Searching batters with filters: {batter_filters}")
            docs_batters = await self.retrieve(query, filters=batter_filters, limit=15)
            logger.info(f"[RAG] Found {len(docs_batters)} batter docs")

            # Combine results
            docs = docs_pitchers + docs_batters
            logger.info(f"[RAG] Total season docs: {len(docs)}")
            if not docs:
                # Fallback to all tables if no season stats found
                logger.info("[RAG] No season stats found, falling back to all tables")
                docs = await self.retrieve(query, filters=filters, limit=20)
        else:
            logger.info("[RAG] Regular search (not season-specific)")
            # Regular search with increased limit to compensate for filtering
            docs = await self.retrieve(query, filters=filters, limit=20)

        # 2. 데이터 처리 및 보강
        processed_data = await self._process_and_enrich_docs(docs, year)
        
        # 3. LLM 컨텍스트 생성
        context_parts = []
        warnings = processed_data["warnings"]
        
        # History context 우선 삽입
        history_block = _history_context_block(history)
        if history_block:
            context_parts.append(history_block)

        # Pitcher context
        sp_pitchers = [p for p in processed_data["pitchers"] if p["role"] == "SP"]
        rp_pitchers = [p for p in processed_data["pitchers"] if p["role"] == "RP"]
        batters = processed_data["batters"]

        if sp_pitchers:
            header = kbo_metrics.scope_header(year, len(set(p['team'] for p in sp_pitchers)), "SP", MIN_IP_SP)
            context_parts.append(f"### 선발 투수 랭킹\n{header}\n")
            for p in sp_pitchers:
                line = (
                    f"{p['name']}({p['team']}) — "
                    f"{kbo_metrics.describe_metric_ko('ERA-', p['era_minus'], 0)}, "
                    f"{kbo_metrics.describe_metric_ko('FIP-', p['fip_minus'], 0)}, "
                    f"{kbo_metrics.describe_metric_ko('K-BB%', p['kbb_pct'], 1)}%, "
                    f"{kbo_metrics.describe_metric_ko('WHIP', p['whip'])} "
                    f"({kbo_metrics.format_ip(p['ip'])} IP)"
                )
                context_parts.append(f"- {line}")
        
        if rp_pitchers:
            header = kbo_metrics.scope_header(year, len(set(p['team'] for p in rp_pitchers)), "RP", MIN_IP_RP)
            context_parts.append(f"\n### 불펜 투수 랭킹\n{header}\n")
            for p in rp_pitchers:
                line = (
                    f"{p['name']}({p['team']}) — "
                    f"{kbo_metrics.describe_metric_ko('ERA-', p['era_minus'], 0)}, "
                    f"{kbo_metrics.describe_metric_ko('FIP-', p['fip_minus'], 0)}, "
                    f"{kbo_metrics.describe_metric_ko('K-BB%', p['kbb_pct'], 1)}%, "
                    f"{kbo_metrics.describe_metric_ko('WHIP', p['whip'])} "
                    f"({kbo_metrics.format_ip(p['ip'])} IP)"
                )
                context_parts.append(f"- {line}")

        if batters:
            header = kbo_metrics.scope_header(
                year,
                len(set(b['team'] for b in batters)),
                "BAT",
                MIN_PA_BATTER,
            )
            context_parts.append(f"\n### 타자 랭킹\n{header}\n")
            for b in batters:
                metrics = []
                metrics.append(kbo_metrics.describe_metric_ko("wRC+", b["wrc_plus"], 0))
                metrics.append(kbo_metrics.describe_metric_ko("OPS", b["ops"], 3))
                metrics.append(kbo_metrics.describe_metric_ko("WAR", b["war"], 2))
                if b.get("ops_plus") is not None:
                    metrics.append(f"OPS+: {b['ops_plus']:.0f}")
                if b.get("avg") is not None:
                    metrics.append(f"타율 {b['avg']:.3f}")
                counting = []
                if b.get("home_runs"):
                    counting.append(f"HR {int(b['home_runs'])}")
                if b.get("rbi"):
                    counting.append(f"RBI {int(b['rbi'])}")
                if b.get("steals"):
                    counting.append(f"SB {int(b['steals'])}")
                if counting:
                    metrics.append("/".join(counting))
                line = f"{b['name']}({b['team']}) — {', '.join(metrics)} ({b['pa']} PA)"
                context_parts.append(f"- {line}")

        if not context_parts:
            context_parts.append("분석에 적합한 데이터가 없습니다. (최소 샘플 기준: 선발 70이닝, 불펜 30이닝)")

        # Add warnings and context info (disabled per user request)
        # if warnings:
        #     context_parts.append("\n### ⚠ 데이터 경고")
        #     context_parts.extend(f"- {w}" for w in warnings)

        # context_parts.append("\n### 분석 노트\n- 순위는 `pitcher_rank_score` (가중치: ERA- 40%, FIP- 30%, K-BB% 20%, WHIP 10%)를 기준으로 하며, IP에 따른 보정이 적용됩니다.")

        joined_context = "\n".join(context_parts)
        prompt = FOLLOWUP_PROMPT.format(question=query, context=joined_context)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(_history_for_messages(history))
        messages.append({"role": "user", "content": prompt})
        
        # 4. LLM을 호출하여 답변 생성
        answer = await self._generate(messages)

        # 5. 최종 결과 구성
        return {
            "answer": answer,
            "citations": [{"id": doc["id"], "title": doc.get("title", "")} for doc in docs],
            "intent": intent,
            "retrieved": docs,
            "strategy": "rag_v2_enriched",
        }
