
import asyncio
import httpx
import json
import time
import sys
import os
import traceback

# Add root to python path to import app modules if needed, but we will test via HTTP
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def verify_streaming():
    url = "http://localhost:8001/coach/analyze"
    payload = {
        "team_id": "KIA",
        "focus": ["recent_form"]
    }
    
    print(f"Connecting to {url}...")
    start_time = time.time()
    first_activity_time = None
    first_token_time = None
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                print(f"Response status: {response.status_code}")
                
                if response.status_code != 200:
                    print("Error: Non-200 status code")
                    return

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                        
                    if line.startswith("data: "):
                        current_time = time.time()
                        if first_activity_time is None:
                            first_activity_time = current_time
                            ttfa = first_activity_time - start_time
                            print(f"✅ First ACTIVITY received after {ttfa:.4f} seconds")

                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                            
                        try:
                            data = json.loads(data_str)
                            
                            # Handle different event types from coach.py
                            # Note process_query_stream yields {type: ...} but coach.py sends {event: ..., data: ...}
                            # Wait, SSE format is:
                            # event: status
                            # data: {"message": "..."}
                            #
                            # httpx.stream reads lines. So we need to handle 'event:' line and 'data:' line.
                            # But my simple parser only looks at 'data:'.
                            # Coach.py sends:
                            # event: status
                            # data: {...}
                            
                            # If I ignore 'event:' line, I just process 'data:'.
                            # 'data' content is JSON string.
                            
                            if "message" in data:
                                print(f"\n[STATUS] {data['message']}")
                            elif "tool" in data:
                                print(f"\n[TOOL] {data['tool']}")
                            elif "delta" in data:
                                if first_token_time is None:
                                    first_token_time = current_time
                                    ttft = first_token_time - start_time
                                    print(f"✅ First TOKEN received after {ttft:.4f} seconds")
                                print(data["delta"], end="", flush=True)
                        except json.JSONDecodeError:
                            pass
                            
    except Exception as e:
        traceback.print_exc()
        print(f"\nError during verification: {repr(e)}")
        return

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n✅ Streaming completed in {total_time:.4f} seconds")
    
    if first_activity_time:
        ttfa = first_activity_time - start_time
        if ttfa < 2.0:
             print(f"SUCCESS: Activity started quickly ({ttfa:.2f}s)")
        else:
             print(f"WARNING: Slow start ({ttfa:.2f}s)")
    else:
        print("FAILURE: No tokens received")

if __name__ == "__main__":
    asyncio.run(verify_streaming())
