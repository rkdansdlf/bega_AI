"""
챗봇 로직 파일
Gemini API 호출 및 Function Calling 처리를 담당합니다.
"""

from openai import OpenAI
import json
import datetime
from typing import Dict, List, Optional

from config import settings
from database import db_manager


class KBOChatbot:
    """KBO 야구 전문 챗봇 클래스"""
    
    def __init__(self):
        """챗봇 초기화"""
        self.client = OpenAI(
            base_url=settings.GEMINI_BASE_URL,
            api_key=settings.GEMINI_API_KEY
        )
        self.model_name = settings.MODEL_NAME
        self.tools = self._define_tools()
        
        print(f"[CHATBOT] 초기화 완료 - Model: {self.model_name}")
    
    def _define_tools(self) -> List[dict]:
        """
        GPT Function Calling Tools 정의
        
        Returns:
            Tools 리스트
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_mysql_query",
                    "description": (
                        "야구 통계나 순위 등 사용자 질문에 필요한 데이터를 MySQL 데이터베이스에서 조회할 때 사용합니다. "
                        "반드시 유효하고 실행 가능한 SQL SELECT 쿼리만 생성해야 합니다. "
                        "다른 SQL 문(INSERT, UPDATE, DELETE 등)은 사용해서는 안 됩니다."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "실행할 완벽하고 안전한 SQL SELECT 쿼리입니다. "
                                    "DB 구조: "
                                    "1) game(game_id, game_date, stadium, home_team, away_team, away_score, home_score, "
                                    "away_pitcher, home_pitcher, winning_team, winning_score), "
                                    "2) box_score(game_id, stadium, crowd, start_time, end_time, game_time, away_record, home_record, "
                                    "away_1~away_15, home_1~home_15, away_r, away_h, away_e, away_b, home_r, home_h, home_e, home_b), "
                                    "3) game_summary(game_id, summary_type, summary_text). "
                                    "테이블 간 JOIN은 game_id로 가능합니다. 날짜는 'YYYY-MM-DD' 형식을 사용하세요."
                                ),
                            }
                        },
                        "required": ["query"],
                    }
                }
            }
        ]
    
    def _create_system_prompt(self) -> str:
        """
        시스템 프롬프트 생성
        
        Returns:
            시스템 프롬프트 문자열
        """
        current_year = datetime.datetime.now().year
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        return (
            f"당신은 한국 야구(KBO) 전문가 챗봇입니다.\n"
            f"**현재 날짜는 {current_date}**입니다.\n"
            f"**현재 연도는 {current_year}년**입니다.\n\n"
            f"사용자 질문에 연도가 명시되지 않았다면, **{current_year}년**을 기본 연도로 가정하고 SQL 쿼리를 생성하세요.\n"
            f"사용 가능한 테이블:\n"
            f"1. game: 기본 경기 정보 (날짜, 팀, 점수, 투수, 승리팀)\n"
            f"2. box_score: 박스스코어 (관중, 시간, 이닝별 점수, R/H/E/B)\n"
            f"3. game_summary: 경기 요약 (결승타, 홈런, 도루, 실책 등)\n"
            f"테이블 간 JOIN은 game_id로 연결하세요.\n\n"
            f"**답변 작성 원칙:**\n"
            f"1. 질문에 직접 답변한 후, 관련된 추가 정보도 함께 제공하세요.\n"
            f"2. 경기 정보 질문 시 포함할 내용:\n"
            f"   - 기본: 날짜, 장소, 양팀 이름, 최종 스코어, 승리팀\n"
            f"   - 이닝별 점수가 있으면 반드시 포함\n"
            f"   - 주요 플레이(결승타, 홈런, 도루 등)가 있으면 반드시 포함\n"
            f"   - 선발 투수 정보\n"
            f"   - 관중 수, 경기 시간 등 부가 정보\n"
            f"3. 여러 테이블에서 정보를 가져올 수 있다면 JOIN하여 풍부한 답변을 제공하세요.\n"
            f"4. 데이터가 없으면 솔직하게 '해당 정보가 없습니다'라고 답변하세요.\n"
            f"5. 날짜는 'YYYY년 MM월 DD일' 형식으로 표현하세요.\n"
            f"6. 팀명은 정식 명칭을 사용하세요 (예: LG 트윈스, 두산 베어스).\n\n"
            f"**예시:**\n"
            f"질문: '10월 14일 경기 어땠어?'\n"
            f"나쁜 답변: '삼성이 5:2로 이겼습니다.'\n"
            f"좋은 답변: '2025년 10월 14일 대구 삼성라이온즈파크에서 열린 경기에서 삼성 라이온즈가 SSG 랜더스를 5:2로 꺾었습니다. \n"
            f"8회 디아즈의 2점 홈런이 결승타가 되었으며, 이재현도 솔로 홈런을 추가했습니다. \n"
            f"선발 투수는 삼성 원태인, SSG 김광현이었습니다. 23,680명의 관중이 3시간 3분간 진행된 경기를 관람했습니다.'"
            f"**중요: 단순 나열이 아닌 스토리텔링**\n"
            f"- ❌ 나쁜 예: '홈런: 디아즈, 이재현'\n"
            f"- ✅ 좋은 예: '8회 디아즈의 극적인 2점 홈런이 터졌고, 이재현이 쐐기 솔로포를 날렸습니다'\n"
            f"- 경기 흐름을 이야기처럼 풀어서 설명하세요\n"
            f"- 주요 순간의 상황과 의미를 함께 전달하세요\n"
        )
    
    def _handle_tool_calls(self, tool_calls, messages: List[dict]) -> List[dict]:
        """
        Tool Calling 처리
        
        Args:
            tool_calls: OpenAI tool_calls 객체
            messages: 대화 메시지 리스트
            
        Returns:
            업데이트된 메시지 리스트
        """
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            sql_query = function_args.get("query")
            
            if function_name == "execute_mysql_query":
                print(f"[CHATBOT] 생성된 SQL: {sql_query}")
                
                # MySQL 쿼리 실행
                tool_output = db_manager.execute_query(sql_query)
                
                # Tool 결과를 대화에 추가
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": tool_output,
                })
                
                print(f"[CHATBOT] 쿼리 결과 길이: {len(tool_output)} 문자")
        
        return messages
    
    def process_question(self, user_question: str) -> Dict:
        """
        사용자 질문 처리 메인 함수
        
        Args:
            user_question: 사용자의 질문
            
        Returns:
            dict: {
                "answer": str,
                "query_executed": str or None,
                "execution_time": float or None
            }
        """
        start_time = datetime.datetime.now()
        executed_query = None
        
        print(f"\n[CHATBOT] 새로운 질문: {user_question}")
        
        # 메시지 구성
        messages = [
            {"role": "system", "content": self._create_system_prompt()},
            {"role": "user", "content": user_question}
        ]
        
        try:
            # 1단계: 초기 응답 생성 (Tool Calling 판단)
            print("[CHATBOT] Step 1: 초기 응답 생성 중...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.0
            )
            
            response_message = response.choices[0].message
            
            # 2단계: Tool Calling 여부 확인 및 처리
            if response_message.tool_calls:
                print(f"[CHATBOT] Step 2: Tool Calling 감지 ({len(response_message.tool_calls)}개)")
                
                # 응답 메시지를 대화에 추가
                messages.append(response_message)
                
                # Tool 실행 및 결과 수집
                for tool_call in response_message.tool_calls:
                    function_args = json.loads(tool_call.function.arguments)
                    executed_query = function_args.get("query")
                
                messages = self._handle_tool_calls(response_message.tool_calls, messages)
                
                # 3단계: Tool 결과 기반 최종 답변 생성
                print("[CHATBOT] Step 3: 최종 답변 생성 중...")
                second_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=[],
                    temperature=0.7
                )
                
                answer = second_response.choices[0].message.content
                
            else:
                # Tool Calling 없이 바로 응답
                print("[CHATBOT] Tool Calling 없음 - 직접 응답")
                answer = response_message.content
            
            # 실행 시간 계산
            end_time = datetime.datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            print(f"[CHATBOT] 완료 - 실행 시간: {execution_time:.2f}초")
            
            return {
                "answer": answer,
                "query_executed": executed_query,
                "execution_time": execution_time
            }
            
        except Exception as e:
            print(f"[ERROR] 챗봇 처리 중 오류: {e}")
            
            return {
                "answer": f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}",
                "query_executed": executed_query,
                "execution_time": None
            }
    
    def test_connection(self) -> bool:
        """
        API 연결 테스트
        
        Returns:
            연결 성공 여부
        """
        try:
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "안녕"}],
                max_tokens=10
            )
            
            if test_response.choices:
                print("[CHATBOT] API 연결 테스트 성공")
                return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] API 연결 테스트 실패: {e}")
            return False


# 싱글톤 인스턴스 생성
chatbot = KBOChatbot()


# 테스트 코드
if __name__ == "__main__":
    print("=" * 50)
    print("챗봇 테스트")
    print("=" * 50)
    
    # API 연결 테스트
    if chatbot.test_connection():
        print("✅ API 연결 성공\n")
        
        # 테스트 질문들
        test_questions = [
            "이번 시즌 가장 큰 점수차로 이긴 경기 알려줘",
            "LG 트윈스의 최근 5경기 결과는?",
            "잠실야구장에서 열린 경기 수는?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[질문 {i}] {question}")
            result = chatbot.process_question(question)
            print(f"[답변] {result['answer']}")
            
            if result['query_executed']:
                print(f"[실행된 쿼리] {result['query_executed']}")
            
            print(f"[실행 시간] {result['execution_time']:.2f}초")
            print("-" * 50)
    else:
        print("❌ API 연결 실패")
    
    print("=" * 50)