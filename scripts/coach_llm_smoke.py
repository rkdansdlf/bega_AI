"""Smoke test the Coach-specific LLM path.

Default mode is dry-run so operators can verify configuration without making a
network call. Pass ``--real`` to call the configured Coach OpenRouter model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BEGA_SKIP_APP_INIT", "1")

from app.config import get_settings


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coach LLM smoke check")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print Coach LLM configuration without calling the provider.",
    )
    mode.add_argument(
        "--real",
        action="store_true",
        help="Call the configured Coach LLM provider.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum response tokens for the real smoke call.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Request timeout in seconds for the real smoke call.",
    )
    return parser.parse_args(argv)


def _print_configuration() -> bool:
    settings = get_settings()
    models: list[str] = []
    for model in [
        settings.coach_openrouter_model or settings.openrouter_model,
        *settings.coach_openrouter_fallback_models,
    ]:
        clean_model = str(model or "").strip()
        if clean_model and clean_model not in models:
            models.append(clean_model)
    key_present = bool(settings.openrouter_api_key)

    print(f"coach_llm_provider={settings.coach_llm_provider}")
    print(f"coach_openrouter_models={','.join(models)}")
    print(f"openrouter_base_url={settings.openrouter_base_url.rstrip('/')}")
    print(f"openrouter_api_key={'present' if key_present else 'absent'}")
    return key_present


async def _run_real_smoke(max_tokens: int, timeout: float) -> int:
    from app.deps import get_coach_llm_generator

    llm = get_coach_llm_generator()
    messages = [
        {
            "role": "system",
            "content": "You are a concise smoke-test responder. Return compact JSON only.",
        },
        {
            "role": "user",
            "content": 'Return exactly this JSON shape with ok=true and source="coach_smoke".',
        },
    ]

    chunks: list[str] = []
    async for chunk in llm(
        messages,
        max_tokens=max_tokens,
        empty_chunk_retry_limit=0,
        request_timeout_seconds=timeout,
    ):
        chunks.append(str(chunk))

    text = "".join(chunks).strip()
    if not text:
        print("ok=false error=empty_response")
        return 1

    print(f"ok=true chars={len(text)} sample={json.dumps(text[:240], ensure_ascii=False)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    key_present = _print_configuration()

    if not args.real:
        print("mode=dry_run")
        return 0

    print("mode=real")
    if not key_present:
        print("skipped=openrouter_api_key_absent")
        return 2

    return asyncio.run(_run_real_smoke(args.max_tokens, args.timeout))


if __name__ == "__main__":
    raise SystemExit(main())
