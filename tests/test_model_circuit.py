from app.core.model_circuit import ModelCircuitBreaker


def test_circuit_opens_after_bounded_failures_and_resets_on_success() -> None:
    circuit = ModelCircuitBreaker(failure_threshold=3, cooldown_seconds=60)

    circuit.record_failure()
    circuit.record_failure()
    assert circuit.allow_request() is True

    circuit.record_failure()
    assert circuit.allow_request() is False

    circuit.record_success()
    assert circuit.allow_request() is True
