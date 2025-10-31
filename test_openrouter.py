"""OpenRouter API 연결 테스트 스크립트"""
import httpx
import json
from app.config import get_settings

def test_openrouter_embedding():
    settings = get_settings()

    print(f"EMBED_PROVIDER: {settings.embed_provider}")
    print(f"OPENROUTER_API_KEY: {settings.openrouter_api_key[:20]}...")
    print(f"OPENROUTER_EMBED_MODEL: {settings.openrouter_embed_model}")
    print(f"OPENROUTER_BASE_URL: {settings.openrouter_base_url}")
    print()

    url = f"{settings.openrouter_base_url}/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "X-Title": "KBO-Embedding-Test"
    }
    payload = {
        "model": settings.openrouter_embed_model,
        "input": ["테스트 텍스트입니다."]
    }

    print(f"URL: {url}")
    print(f"Model: {settings.openrouter_embed_model}")
    print()

    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        print(f"Response (first 500 chars):")
        print(response.text[:500])
        print()

        if response.status_code == 200:
            try:
                data = response.json()
                print("JSON Response:")
                print(json.dumps(data, indent=2, ensure_ascii=False)[:500])
            except:
                print("Failed to parse JSON")
        else:
            print("Error response!")

    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    test_openrouter_embedding()
