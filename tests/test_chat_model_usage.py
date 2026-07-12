from decimal import Decimal

import pytest

from app.core.chat_model_usage import ModelPricingCatalog, estimate_model_usage


PRICING_JSON = """{
  "openrouter": {
    "vendor/planner": {
      "input_usd_per_1m_tokens": "1.00",
      "output_usd_per_1m_tokens": "2.00"
    }
  }
}"""


def test_catalog_uses_exact_provider_and_model_lookup() -> None:
    catalog = ModelPricingCatalog.from_json(PRICING_JSON)
    price = catalog.lookup("openrouter", "vendor/planner")

    assert price is not None
    assert price.input_usd_per_1m_tokens == Decimal("1.00")
    assert catalog.lookup("openrouter", "vendor/other") is None


def test_catalog_normalizes_padded_provider_and_model_keys() -> None:
    catalog = ModelPricingCatalog.from_json(
        PRICING_JSON.replace('"openrouter"', '" openrouter "').replace(
            '"vendor/planner"', '" vendor/planner "'
        )
    )

    assert catalog.lookup("openrouter", "vendor/planner") is not None


@pytest.mark.parametrize(
    "raw",
    [
        '{"   ":{"model":{"input_usd_per_1m_tokens":"1","output_usd_per_1m_tokens":"2"}}}',
        '{"provider":{"   ":{"input_usd_per_1m_tokens":"1","output_usd_per_1m_tokens":"2"}}}',
    ],
)
def test_catalog_rejects_blank_normalized_keys(raw: str) -> None:
    with pytest.raises(ValueError, match="blank"):
        ModelPricingCatalog.from_json(raw)


def test_catalog_rejects_provider_normalization_collisions() -> None:
    raw = """{
      "provider": {"one": {"input_usd_per_1m_tokens": "1", "output_usd_per_1m_tokens": "2"}},
      " provider ": {"two": {"input_usd_per_1m_tokens": "1", "output_usd_per_1m_tokens": "2"}}
    }"""

    with pytest.raises(ValueError, match="provider.*collision"):
        ModelPricingCatalog.from_json(raw)


def test_catalog_rejects_model_normalization_collisions() -> None:
    raw = """{
      "provider": {
        "model": {"input_usd_per_1m_tokens": "1", "output_usd_per_1m_tokens": "2"},
        " model ": {"input_usd_per_1m_tokens": "1", "output_usd_per_1m_tokens": "2"}
      }
    }"""

    with pytest.raises(ValueError, match="model.*collision"):
        ModelPricingCatalog.from_json(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        "[]",
        '{"openrouter":{"vendor/planner":{"input_usd_per_1m_tokens":"-1","output_usd_per_1m_tokens":"2"}}}',
        '{"openrouter":{"vendor/planner":{"input_usd_per_1m_tokens":"NaN","output_usd_per_1m_tokens":"2"}}}',
        '{"openrouter":{"vendor/planner":{"input_usd_per_1m_tokens":"1"}}}',
    ],
)
def test_catalog_rejects_invalid_pricing(raw: str) -> None:
    with pytest.raises(ValueError):
        ModelPricingCatalog.from_json(raw)


def test_usage_estimate_is_deterministic_and_sanitized() -> None:
    catalog = ModelPricingCatalog.from_json(PRICING_JSON)
    record = estimate_model_usage(
        catalog,
        role="planner",
        provider="openrouter",
        model="vendor/planner",
        messages=[{"role": "user", "content": "abcd"}],
        output_text="1234567",
        outcome="success",
    )

    assert record.input_tokens > 0
    assert record.output_tokens == 2
    assert record.pricing_source == "model_catalog"
    assert record.total_cost_usd == (
        Decimal(record.input_tokens) / Decimal("1000000")
        + Decimal("0.000004")
    )
    payload = record.to_dict()
    assert payload["role"] == "planner"
    assert payload["total_cost_usd"] == format(record.total_cost_usd, ".12f")
    assert "abcd" not in str(payload)
    assert "1234567" not in str(payload)
