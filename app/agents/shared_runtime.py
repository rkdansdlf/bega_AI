from __future__ import annotations

from threading import RLock
from typing import Any

from .baseball_agent import BaseballAgentRuntime
from .runtime_factory import create_baseball_agent_runtime

_RUNTIME_LOCK = RLock()
_default_runtime: BaseballAgentRuntime | None = None
_runtime_by_settings_id: dict[int, BaseballAgentRuntime] = {}


def initialize_shared_baseball_agent_runtime(
    settings: Any | None = None,
) -> BaseballAgentRuntime:
    """Return a shared runtime, keyed by the explicit settings object when provided."""
    global _default_runtime

    with _RUNTIME_LOCK:
        if settings is None:
            if _default_runtime is None:
                _default_runtime = create_baseball_agent_runtime()
            return _default_runtime

        settings_key = id(settings)
        runtime = _runtime_by_settings_id.get(settings_key)
        if runtime is None:
            runtime = create_baseball_agent_runtime(settings)
            _runtime_by_settings_id[settings_key] = runtime
        return runtime


def get_shared_baseball_agent_runtime() -> BaseballAgentRuntime:
    runtime = _default_runtime
    if runtime is None:
        raise RuntimeError(
            "BaseballAgentRuntime is not initialized. Ensure FastAPI lifespan has started."
        )
    return runtime


def reset_shared_baseball_agent_runtime() -> None:
    global _default_runtime
    with _RUNTIME_LOCK:
        _default_runtime = None
        _runtime_by_settings_id.clear()
