
import requests
import json
import sys

BASE_URL = "http://localhost:8002"

def run_test(scenario_name, question):
    print(f"\n[Testing Scenario: {scenario_name}]")
    print(f"Q: {question}")
    
    url = f"{BASE_URL}/chat/stream"
    payload = {"question": question, "style": "markdown"}
    headers = {"Content-Type": "application/json"}
    
    try:
        with requests.post(url, json=payload, headers=headers, stream=True) as r:
            r.raise_for_status()
            
            answer_content = ""
            tool_calls = []
            
            for line in r.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("event: meta"):
                        # Meta event often follows with tool info in data
                        pass 
                    elif decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data_json = json.loads(data_str)
                            
                            # Collect tool calls if present in meta or direct data
                            if "tool_calls" in data_json:
                                tool_calls.extend(data_json["tool_calls"])
                                
                            # If it's a content chunk (streaming text)
                            if "choices" in data_json: # OpenAI format often used
                                delta = data_json["choices"][0].get("delta", {}).get("content", "")
                                if delta:
                                    answer_content += delta
                            elif isinstance(data_json, dict) and "tool_calls" in data_json:
                                # Sometimes meta data comes as the data payload
                                print(f"  > Tools Used: {[t['tool_name'] for t in data_json['tool_calls']]}")
                            
                        except json.JSONDecodeError:
                            # Sometimes plain text or just partial buffer
                            pass
            
            # Since the specific streaming format might vary (Server-Sent Events),
            # Let's just print the raw concatenated text if we managed to capture it,
            # or rely on observing the 'meta' event for tool usage verification.
            
            # NOTE: The current chatbot implementation streams the final answer at the end or in chunks.
            # Based on previous curl output, looks like standard SSE.
            # Let's try to capture the final answer if possible, but tool usage is key.
            pass

    except Exception as e:
        print(f"Error: {e}")
        return

    # For this test, we might not capture full streaming text easily without exact protocol knowledge,
    # but we can see the HTTP 200 and the log output. 
    # Let's rely on printing what we found.
    print("  > Request Successful")

def basic_curl_test():
    # Because writing a robust SSE client in one go is tricky without dependencies, 
    # let's write a wrapper that calls curl for simplicity and reliability in showing output to User.
    import subprocess
    
    scenarios = [
        ("Player Stats (SQL)", "김도영 선수의 2024년 시즌 타율은?"),
        ("Game Info (SQL)", "2024년 5월 1일 경기 결과 알려줘"),
        ("RAG Analysis", "2024년 KIA 타이거즈 투수진 성적 분석해줘")
    ]
    
    for name, q in scenarios:
        print(f"\n{'='*50}")
        print(f"SCENARIO: {name}")
        print(f"QUERY: {q}")
        print(f"{'='*50}")
        
        cmd = [
            "curl", "-N", "-X", "POST", f"{BASE_URL}/chat/stream",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({"question": q})
        ]
        
        # Run curl and print output directly
        subprocess.run(cmd)

if __name__ == "__main__":
    basic_curl_test()
