"""Function Calling 기반 KBO 챗봇 어댑터.

레거시 환경에서 사용하던 동기식 인터페이스는 유지하면서,
OpenAI 호환 API(Function Calling)를 통해 Postgres 질의를 자동 생성·실행합니다.
"""

from __future__ import annotations

import datetime
import json
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from config import settings
from database import db_manager


class KBOChatbot:
    """Function Calling을 사용해 SQL 조회를 수행하는 KBO 챗봇."""

    def __init__(self) -> None:
        self.settings = settings
        self.client = self._create_client()
        self.model_name = self.settings.function_calling_model
        self.tools = self._define_tools()
        print(f"[CHATBOT] Function Calling 준비 완료 - Provider={self.settings.llm_provider}, Model={self.model_name}")

    def _create_client(self) -> OpenAI:
        """OpenAI 호환 클라이언트를 초기화합니다."""
        api_key = self.settings.function_calling_api_key
        if not api_key:
            raise RuntimeError("Function Calling 챗봇을 사용하려면 API 키가 필요합니다.")

        base_url = self.settings.function_calling_base_url.rstrip("/")
        # OpenAI SDK는 base_url에 /v1을 포함해야 합니다.
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        client_kwargs = {"base_url": base_url, "api_key": api_key}
        if self.settings.llm_provider == "openrouter":
            default_headers = {}
            if self.settings.openrouter_referer:
                default_headers["HTTP-Referer"] = self.settings.openrouter_referer
            if self.settings.openrouter_app_title:
                default_headers["X-Title"] = self.settings.openrouter_app_title
            if default_headers:
                client_kwargs["default_headers"] = default_headers
        return OpenAI(**client_kwargs)

    def _define_tools(self) -> List[dict]:
        """Function Calling에서 사용할 Tool 스키마를 정의합니다."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_sql_query",
                    "description": (
                        "야구 통계나 순위 등 사용자 질문에 필요한 데이터를 PostgreSQL 데이터베이스에서 조회할 때 사용합니다. "
                        "반드시 안전한 SQL SELECT 쿼리만 생성해야 하며, INSERT/UPDATE/DELETE 등은 절대 금지입니다."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "실행할 완전한 SQL SELECT 쿼리입니다.\n\n"
                                    "주요 테이블:\n"
                                    "1) player_season_pitching - 투수 시즌 통계\n"
                                    "   컬럼: player_id, season, team_code, games, games_started, wins, losses, saves, holds, "
                                    "innings_pitched, strikeouts, walks_allowed, era, whip, fip, complete_games, shutouts\n"
                                    "   - season: 연도 정수 (2025, 2024 등)\n"
                                    "   - player_id로 player_basic과 조인하여 이름 가져오기\n"
                                    "   예: SELECT pb.name, psp.era, psp.wins FROM player_season_pitching psp "
                                    "JOIN player_basic pb ON psp.player_id = pb.player_id WHERE psp.season = 2025\n\n"
                                    "2) player_season_batting - 타자 시즌 통계\n"
                                    "   컬럼: player_id, season, team_code, games, at_bats, runs, hits, doubles, triples, "
                                    "home_runs, rbi, walks, strikeouts, stolen_bases, avg, obp, slg, ops\n"
                                    "   - player_id로 player_basic과 조인하여 이름 가져오기\n\n"
                                    "3) player_basic - 선수 정보\n"
                                    "   컬럼: player_id, name, uniform_no, position, birth_date, team_id\n\n"
                                    "4) game - 경기 정보\n"
                                    "   컬럼: game_id, game_date, home_team, away_team, home_score, away_score, "
                                    "winning_team, home_pitcher, away_pitcher, stadium, season_id\n\n"
                                    "5) hitter_record, pitcher_record - 경기별 선수 기록\n"
                                    "   player_name 컬럼 포함 (조인 불필요)\n\n"
                                    "6) teams - 팀 정보 (team_id, team_name, city)\n"
                                    "7) kbo_seasons - 시즌 정보 (season_id, season_year, league_type_code)\n\n"
                                    "중요 규칙:\n"
                                    "- season 컬럼: 연도 정수 (2025, 2024)\n"
                                    "- 선수 이름: player_basic.name과 JOIN\n"
                                    "- 최소 기준: 투수 innings_pitched >= 30, 타자 at_bats >= 100"
                                ),
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    def _create_system_prompt(self) -> str:
        """시스템 프롬프트를 생성합니다."""
        current_dt = datetime.datetime.now()
        current_year = current_dt.year
        current_date = current_dt.strftime("%Y-%m-%d")
        return (
            f"당신은 한국 프로야구(KBO) 전문 챗봇입니다.\n"
            f"**현재 날짜는 {current_date}**, **현재 연도는 {current_year}년**입니다.\n\n"
            "## 대화 맥락 처리\n"
            "- 이전 사용자 발화를 항상 참조하세요.\n"
            "- '이 경기', '그 선수' 등 지시 대명사는 직전 문맥을 사용합니다.\n"
            "- 직전 대화에서 특정 경기/선수를 다뤘다면 같은 대상의 추가 질문으로 간주하세요.\n\n"
            "## 데이터베이스 활용 가이드\n"
            "- **투수 시즌 통계**: `player_season_pitching` 테이블 사용\n"
            "  * season 컬럼에 연도 저장 (2025, 2024 등)\n"
            "  * 선수 이름은 player_basic과 JOIN 필수 (ON player_id)\n"
            "  * 주요 컬럼: era, wins, losses, saves, innings_pitched, strikeouts, whip\n"
            "  * 최소 기준: innings_pitched >= 30\n"
            "- **타자 시즌 통계**: `player_season_batting` 테이블 사용\n"
            "  * season 컬럼에 연도 저장\n"
            "  * player_basic과 JOIN 필수\n"
            "  * 주요 컬럼: avg, ops, home_runs, rbis, hits\n"
            "- 경기 개요는 `game` 테이블 사용\n"
            "- 주요 플레이는 `game_summary` 사용\n"
            "- 사용자가 연도를 말하지 않으면 현재 연도({current_year})를 사용하세요.\n\n"
            "## 답변 작성 원칙\n"
            "1. 사용자가 특정 경기의 '결과'만 물으면 core 정보(날짜/장소/스코어/선발/승리팀)만 제공하고, "
            "   마지막에 '주요 플레이가 궁금하신가요?'와 같은 후속 질문을 추가하세요.\n"
            "2. 박스스코어·주요 플레이·선수 기록 등 상세 정보는 사용자가 명시적으로 요청할 때만 포함하세요.\n"
            "3. 통계 질문은 숫자, 표, 맥락 설명(의미/비교)을 순서대로 전달하세요.\n"
            "4. 데이터가 없거나 오류가 발생하면 DB 세부 정보를 노출하지 말고 사용자 친화적인 문장으로 사과하세요.\n"
            "5. 팀명은 정식 명칭(예: LG 트윈스), 점수는 콜론(5:2), 큰 수치는 천 단위 구분(23,680명)을 사용하세요.\n"
            "6. SQL, 테이블명, 컬럼명 등 기술 용어는 답변에 포함하지 마세요.\n"
        )

    def _handle_tool_calls(
        self,
        tool_calls,
        messages: List[dict],
    ) -> Tuple[List[dict], List[str]]:
        """Function Calling으로 생성된 tool 호출을 처리합니다."""
        executed_queries: List[str] = []
        for tool_call in tool_calls or []:
            name = getattr(tool_call.function, "name", "")
            if name != "execute_sql_query":
                continue
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            sql_query = args.get("query")
            if not sql_query:
                continue
            executed_queries.append(sql_query)
            print(f"[CHATBOT] 실행할 SQL: {sql_query}")
            tool_output = db_manager.execute_query(sql_query)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": name,
                    "content": tool_output,
                }
            )
            print(f"[CHATBOT] Tool 응답 길이: {len(tool_output)} 문자")
        return messages, executed_queries

    def process_question(
        self,
        user_question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Optional[str]]:
        """질문을 처리하고 답변/실행 쿼리/시간 정보를 반환합니다."""
        print(f"\n[CHATBOT] 새로운 질문: {user_question}")
        start_time = datetime.datetime.now()
        messages: List[dict] = [{"role": "system", "content": self._create_system_prompt()}]

        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_question})

        executed_query: Optional[str] = None
        try:
            print("[CHATBOT] Step 1: Tool 사용 여부 판별 중...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.0,
            )
            response_message = response.choices[0].message

            if response_message.tool_calls:
                print(f"[CHATBOT] Step 2: Tool 호출 {len(response_message.tool_calls)}건 감지")
                tool_calls_payload = [
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in response_message.tool_calls
                ]
                messages.append(
                    {
                        "role": response_message.role,
                        "content": response_message.content or "",
                        "tool_calls": tool_calls_payload,
                    }
                )
                messages, executed_queries = self._handle_tool_calls(response_message.tool_calls, messages)
                executed_query = executed_queries[-1] if executed_queries else None

                print("[CHATBOT] Step 3: Tool 결과 기반 최종 답변 생성 중...")
                final_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                )
                answer = final_response.choices[0].message.content
            else:
                print("[CHATBOT] Tool 호출 없이 직접 응답")
                answer = response_message.content

            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            print(f"[CHATBOT] 완료 - 실행 시간: {elapsed:.2f}초")
            return {
                "answer": answer,
                "query_executed": executed_query,
                "execution_time": elapsed,
            }
        except Exception as exc:
            print(f"[ERROR] 챗봇 처리 중 오류: {exc}")
            return {
                "answer": "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "query_executed": executed_query,
                "execution_time": None,
            }

    def test_connection(self) -> bool:
        """LLM API 연결을 확인합니다."""
        try:
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "안녕"}],
                max_tokens=8,
            )
            success = bool(test_response.choices)
            if success:
                print("[CHATBOT] API 연결 테스트 성공")
            else:
                print("[CHATBOT] API 연결 테스트 실패: 응답 없음")
            return success
        except Exception as exc:
            print(f"[ERROR] API 연결 테스트 실패: {exc}")
            return False


# 전역 인스턴스 (레거시 호환 목적)
chatbot = KBOChatbot()


if __name__ == "__main__":
    print("=" * 60)
    print("Function Calling 챗봇 테스트")
    print("=" * 60)
    if chatbot.test_connection():
        demo_questions = [
            "이번 시즌 가장 큰 점수차로 이긴 경기 알려줘",
            "LG 트윈스의 최근 5경기 결과는?",
            "잠실야구장에서 열린 경기 수는?",
        ]
        for idx, question in enumerate(demo_questions, 1):
            print(f"\n[질문 {idx}] {question}")
            result = chatbot.process_question(question)
            print(f"[답변]\n{result['answer']}")
            if result["query_executed"]:
                print(f"[실행된 쿼리]\n{result['query_executed']}")
            if result["execution_time"] is not None:
                print(f"[실행 시간] {result['execution_time']:.2f}초")
            print("-" * 40)
    else:
        print("❌ API 연결 실패")
    print("=" * 60)
