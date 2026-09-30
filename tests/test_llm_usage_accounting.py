from decimal import Decimal

from app.core.chat_model_usage import ModelPricingCatalog
from app.core.llm_provider import LLMResult, Usage
from app.core.llm_usage_accounting import account_llm_usage, resolve_usage

CATALOG = ModelPricingCatalog.from_json(
    '{"openrouter": {"vendor/m": {"input_usd_per_1m_tokens": "2", '
    '"output_usd_per_1m_tokens": "10"}}}'
)
MSGS = [{"role": "user", "content": "질문"}]


def test_provider_reported_usage_wins_over_estimate():
    counts = resolve_usage(
        Usage(1000, 200, 1200, "provider"), messages=MSGS, output_text="x" * 9999
    )
    assert counts == {
        "input_tokens": 1000,
        "output_tokens": 200,
        "usage_source": "provider",
    }


def test_estimate_used_only_when_provider_usage_missing():
    counts = resolve_usage(Usage(), messages=MSGS, output_text="가" * 30)
    assert counts["usage_source"] == "estimate"
    assert counts["output_tokens"] == 10


def test_cost_is_priced_from_reported_tokens():
    row = account_llm_usage(
        LLMResult(
            "답", "openrouter", "vendor/m", Usage(1_000_000, 100_000, None, "provider")
        ),
        messages=MSGS,
        catalog=CATALOG,
    )
    assert row["usage_source"] == "provider"
    assert Decimal(row["cost_usd"]) == Decimal("3")  # 1M*$2 + 0.1M*$10
    assert row["pricing_source"] == "model_catalog"


def test_unpriced_model_has_no_cost_but_still_counts_tokens():
    row = account_llm_usage(
        LLMResult("답", "gemini", "g", Usage(10, 5, 15, "provider")),
        messages=MSGS,
        catalog=CATALOG,
    )
    assert row["cost_usd"] is None and row["pricing_source"] == "unpriced"
    assert row["input_tokens"] == 10


def test_total_only_usage_derives_output_tokens():
    counts = resolve_usage(
        Usage(40, None, 100, "provider"), messages=MSGS, output_text=""
    )
    assert counts["output_tokens"] == 60
