import requests
import json
import sys
import time

BASE_URL = "http://localhost:8001"  # Port 8001 for AI service

SCENARIOS = [
    # 1. Player Data
    ("Player Info", "김도영 선수의 포지션은 무엇인가요?"),
    ("Player Season Stats", "2024년 최형우의 타율과 홈런 수는?"),
    
    # 2. Team Data
    ("Team History", "롯데 자이언츠의 창단 연도는?"),
    ("Stadium Info", "잠실야구장의 수용 인원은 몇 명인가요?"),
    
    # 3. Game Data
    ("Game Result", "2024년 5월 1일 KIA vs KT 경기 결과는?"),
    ("Game Lineup", "2024년 5월 1일 KIA 타이거즈의 4번 타자는 누구였나요?"),
    
    # 4. Awards
    ("Awards", "2023년 KBO MVP는 누구인가요?"),
    
    # 5. Regulations (If ingested)
    ("Regulations", "KBO의 타이브레이크 규정에 대해 설명해줘."),
    
    # 6. Advanced / Analysis
    ("Analysis", "2024년 시즌 KIA 타이거즈의 팀 방어율은 어떠했나요?"),
]

def run_test():
    print(f"\n{'='*60}")
    print(f"  BEGA Chatbot Comprehensive Evaluation")
    print(f"{'='*60}\n")
    
    failed_cases = []
    
    for category, question in SCENARIOS:
        print(f"[{category}] Q: {question}")
        
        try:
            start_time = time.time()
            
            # Using /chat/completion for non-streaming simple response if available, 
            # otherwise handle /chat/stream. Let's assume /chat/stream for now as it's the main one.
            # But wait, test_diagnosis used /chat/completion for health check. Let's try that first for simpler parsing.
            
            # Note: The User's API might only accept 'stream'. Let's check test_chatbot_scenarios again.
            # It used /chat/stream. Let's stick to that to be safe.
            
            url = f"{BASE_URL}/chat/stream"
            payload = {"question": question}
            
            with requests.post(url, json=payload, stream=True, timeout=30) as r:
                r.raise_for_status()
                
                full_content = ""
                tools_used = []
                
                for line in r.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("data: "):
                            data_str = decoded[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data:
                                    delta = data["choices"][0].get("delta", {}).get("content", "")
                                    if delta:
                                        full_content += delta
                                if "tool_calls" in data:
                                    tools_used.extend([t['function']['name'] for t in data['tool_calls']])
                            except:
                                pass
                
                elapsed = time.time() - start_time
                print(f"  -> Answer: {full_content[:100]}... (Total {len(full_content)} chars)")
                print(f"  -> Time: {elapsed:.2f}s")
                if tools_used:
                    print(f"  -> Tools: {list(set(tools_used))}")
                print(f"  -> Status: ✅ OK\n")
                
        except Exception as e:
            print(f"  -> Error: {e}")
            print(f"  -> Status: ❌ FAIL\n")
            failed_cases.append((category, str(e)))

    print(f"{'='*60}")
    if failed_cases:
        print(f"Failures: {len(failed_cases)}/{len(SCENARIOS)}")
        for cat, err in failed_cases:
            print(f" - {cat}: {err}")
    else:
        print("All scenarios executed successfully (HTTP 200).")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_test()
