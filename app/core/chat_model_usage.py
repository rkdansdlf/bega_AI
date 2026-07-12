"""Pure pricing catalog and deterministic chat model usage estimates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from types import MappingProxyType
from typing import Literal, Mapping

ModelRole = Literal["planner", "answer"]
ModelCallOutcome = Literal["success", "failed"]
PricingSource = Literal["model_catalog", "unpriced"]

_CHARS_PER_TOKEN = Decimal("3.5")
_MILLION = Decimal("1000000")


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_usd_per_1m_tokens: Decimal
    output_usd_per_1m_tokens: Decimal


@dataclass(frozen=True, slots=True)
class ModelUsageEstimate:
    role: ModelRole
    provider: str
    model: str
    outcome: ModelCallOutcome
    pricing_source: PricingSource
    input_chars: int
    output_chars: int
    input_tokens: int
    output_tokens: int
    input_cost_usd: Decimal | None
    output_cost_usd: Decimal | None
    total_cost_usd: Decimal | None

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "outcome": self.outcome,
            "pricing_source": self.pricing_source,
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost_usd": format_usd(self.input_cost_usd),
            "output_cost_usd": format_usd(self.output_cost_usd),
            "total_cost_usd": format_usd(self.total_cost_usd),
        }


@dataclass(frozen=True, slots=True)
class ModelPricingCatalog:
    prices: Mapping[tuple[str, str], ModelPrice]

    @classmethod
    def from_json(cls, raw: str | None) -> "ModelPricingCatalog":
        if raw is None or not raw.strip():
            return cls(prices={})
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("pricing root must be an object")
        return cls(prices=MappingProxyType(_parse_price_entries(parsed)))

    def lookup(self, provider: str, model: str) -> ModelPrice | None:
        return self.prices.get((provider.strip(), model.strip()))


def _parse_price_entries(parsed: dict[object, object]) -> dict[tuple[str, str], ModelPrice]:
    prices: dict[tuple[str, str], ModelPrice] = {}
    normalized_providers: set[str] = set()
    for provider, provider_entries in parsed.items():
        if not isinstance(provider, str) or not isinstance(provider_entries, dict):
            raise ValueError("pricing providers must map to objects")
        normalized_provider = provider.strip()
        if not normalized_provider:
            raise ValueError("pricing provider cannot be blank")
        if normalized_provider in normalized_providers:
            raise ValueError("pricing provider normalization collision")
        normalized_providers.add(normalized_provider)

        normalized_models: set[str] = set()
        for model, raw_price in provider_entries.items():
            if not isinstance(model, str) or not isinstance(raw_price, dict):
                raise ValueError("pricing models must map to objects")
            normalized_model = model.strip()
            if not normalized_model:
                raise ValueError("pricing model cannot be blank")
            if normalized_model in normalized_models:
                raise ValueError("pricing model normalization collision")
            normalized_models.add(normalized_model)
            if set(raw_price) != {
                "input_usd_per_1m_tokens",
                "output_usd_per_1m_tokens",
            }:
                raise ValueError(
                    "pricing entries must contain exactly input and output rates"
                )

            input_rate = _parse_rate(
                raw_price["input_usd_per_1m_tokens"],
                "input_usd_per_1m_tokens",
            )
            output_rate = _parse_rate(
                raw_price["output_usd_per_1m_tokens"],
                "output_usd_per_1m_tokens",
            )
            prices[(normalized_provider, normalized_model)] = ModelPrice(
                input_usd_per_1m_tokens=input_rate,
                output_usd_per_1m_tokens=output_rate,
            )
    return prices


def _parse_rate(raw_rate: object, field_name: str) -> Decimal:
    if not isinstance(raw_rate, str):
        raise ValueError(f"{field_name} must be a string")
    try:
        rate = Decimal(raw_rate)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name} must be a decimal") from exc
    if not rate.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if rate < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return rate


def serialize_messages(messages: object) -> str:
    return json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _estimate_tokens(chars: int) -> int:
    return int((Decimal(chars) / _CHARS_PER_TOKEN).to_integral_value(rounding=ROUND_CEILING))


def estimate_model_usage(
    catalog: ModelPricingCatalog,
    *,
    role: ModelRole,
    provider: str,
    model: str,
    messages: object,
    output_text: str,
    outcome: ModelCallOutcome,
) -> ModelUsageEstimate:
    input_chars = len(serialize_messages(messages))
    output_chars = len(output_text)
    input_tokens = _estimate_tokens(input_chars)
    output_tokens = _estimate_tokens(output_chars)
    price = catalog.lookup(provider, model)

    if price is None:
        pricing_source: PricingSource = "unpriced"
        input_cost_usd = None
        output_cost_usd = None
        total_cost_usd = None
    else:
        pricing_source = "model_catalog"
        input_cost_usd = Decimal(input_tokens) * price.input_usd_per_1m_tokens / _MILLION
        output_cost_usd = Decimal(output_tokens) * price.output_usd_per_1m_tokens / _MILLION
        total_cost_usd = input_cost_usd + output_cost_usd

    return ModelUsageEstimate(
        role=role,
        provider=provider,
        model=model,
        outcome=outcome,
        pricing_source=pricing_source,
        input_chars=input_chars,
        output_chars=output_chars,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_cost_usd,
        output_cost_usd=output_cost_usd,
        total_cost_usd=total_cost_usd,
    )


def format_usd(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value, ".12f")
