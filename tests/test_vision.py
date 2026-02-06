import httpx
import base64
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
# Testing vision_model from env
VISION_MODEL = os.getenv("VISION_MODEL", "openrouter/free")


async def _test_vision():
    print(f"Testing Vision Model: {VISION_MODEL}")

    # Simple 1x1 black pixel image in base64
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    prompt = "What is in this image?"

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print("Success!")
                print(response.json()["choices"][0]["message"]["content"])
            else:
                print(f"Error Body: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")


def test_vision():
    import asyncio

    asyncio.run(_test_vision())


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_vision())
