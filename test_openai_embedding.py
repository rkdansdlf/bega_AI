"""OpenAI Embedding API 연결 테스트 스크립트"""
import httpx
import json
from app.config import get_settings

def test_openai_embedding():
    settings = get_settings()

    print("=" * 60)
    print("OpenAI Embedding API 테스트")
    print("=" * 60)
    print(f"EMBED_PROVIDER: {settings.embed_provider}")
    print(f"OPENAI_API_KEY: {settings.openai_api_key[:20] if settings.openai_api_key else 'NOT SET'}...")
    print(f"OPENAI_EMBED_MODEL: {settings.openai_embed_model}")
    print()

    if not settings.openai_api_key:
        print("❌ OPENAI_API_KEY가 설정되어 있지 않습니다!")
        print("   .env 파일에 OPENAI_API_KEY를 추가해주세요.")
        return

    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.openai_embed_model,
        "input": ["KBO 리그는 한국 프로야구 리그입니다."]
    }

    print(f"URL: {url}")
    print(f"Model: {settings.openai_embed_model}")
    print(f"Test Text: {payload['input'][0]}")
    print()

    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            embeddings = data.get("data", [])
            if embeddings:
                embedding_vector = embeddings[0].get("embedding", [])
                print(f"✅ 임베딩 성공!")
                print(f"   벡터 차원: {len(embedding_vector)}")
                print(f"   벡터 샘플 (처음 5개): {embedding_vector[:5]}")
                print(f"   토큰 사용량: {data.get('usage', {})}")
            else:
                print("❌ 임베딩 데이터가 없습니다.")
        else:
            print(f"❌ API 오류:")
            print(response.text[:500])

    except Exception as e:
        print(f"❌ 예외 발생: {e}")

if __name__ == "__main__":
    test_openai_embedding()
