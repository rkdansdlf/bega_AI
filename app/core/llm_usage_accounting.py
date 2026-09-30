"""Bill-grade LLM usage accounting.

Preference order for token counts: provider-reported usage
(``usage_source="provider"``) → local estimate (``"estimate"``). Cost is priced
from the model catalog and only derived from the same counts that were
recorded, so provider-reported rows can be reconciled against the daily bill.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, Optional

from ..observability.metrics import (
    AI_LLM_CIRCUIT_STATE,
    AI_LLM_USAGE_COST_USD_TOTAL,
    AI_LLM_USAGE_TOKENS_TOTAL,
)
from .chat_model_usage import ModelPricingCatalog, serialize_messages
from .llm_provider import LLMResult, Usage, estimate_tokens
from .provider_circuit import CircuitState

logger = logging.getLogger(__name__)

_MILLION = Decimal(1_000_000)
_CIRCUIT_GAUGE = {
    CircuitState.CLOSED.value: 0,
    CircuitState.HALF_OPEN.value: 1,
    CircuitState.OPEN.value: 2,
}


def resolve_usage(usage: Usage, *, messages: Any, output_text: str) -> Dict[str, Any]:
    """Token counts with their source; estimates fill only what is missing."""
    if usage.is_reported and usage.prompt_tokens is not None:
        input_tokens = int(usage.prompt_tokens)
        output_tokens = int(
            usage.completion_tokens
            if usage.completion_tokens is not None
            else max(0, int(usage.total_tokens or 0) - input_tokens)
        )
        source = "provider"
    else:
        input_tokens = estimate_tokens(serialize_messages(messages))
        output_tokens = estimate_tokens(output_text)
        source = "estimate"
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usage_source": source,
    }


def account_llm_usage(
    result: LLMResult,
    *,
    messages: Any,
    catalog: Optional[ModelPricingCatalog] = None,
) -> Dict[str, Any]:
    """Record metrics for one completed call and return the accounting row."""
    counts = resolve_usage(result.usage, messages=messages, output_text=result.text)
    provider = result.provider
    model = result.model or "unknown"
    cost: Optional[Decimal] = None
    pricing_source = "unpriced"
    price = catalog.lookup(provider, model) if catalog is not None else None
    if price is not None:
        cost = (
            Decimal(counts["input_tokens"]) * price.input_usd_per_1m_tokens
            + Decimal(counts["output_tokens"]) * price.output_usd_per_1m_tokens
        ) / _MILLION
        pricing_source = "model_catalog"

    source = counts["usage_source"]
    try:
        AI_LLM_USAGE_TOKENS_TOTAL.labels(
            provider=provider, model=model, token_type="input", usage_source=source
        ).inc(counts["input_tokens"])
        AI_LLM_USAGE_TOKENS_TOTAL.labels(
            provider=provider, model=model, token_type="output", usage_source=source
        ).inc(counts["output_tokens"])
        if cost is not None:
            AI_LLM_USAGE_COST_USD_TOTAL.labels(
                provider=provider, model=model, usage_source=source
            ).inc(float(cost))
    except Exception:  # noqa: BLE001 - metrics never break answers
        logger.debug("usage metrics failed", exc_info=True)

    return {
        "provider": provider,
        "model": model,
        **counts,
        "cost_usd": format(cost, ".12f") if cost is not None else None,
        "pricing_source": pricing_source,
    }


def publish_circuit_states(snapshots: list) -> None:
    for snap in snapshots:
        try:
            AI_LLM_CIRCUIT_STATE.labels(provider=snap["provider"]).set(
                _CIRCUIT_GAUGE.get(snap["state"], 0)
            )
        except Exception:  # noqa: BLE001
            pass
