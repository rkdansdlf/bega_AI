import requests
import json

url = "http://localhost:8001/chat/stream"
payload = {"question": "Hello"}

print("Sending request...")
with requests.post(url, json=payload, stream=True) as r:
    print(f"Status: {r.status_code}")
    for line in r.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            print(f"RAW: {decoded}")
