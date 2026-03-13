from __future__ import annotations

from importlib import import_module

__all__ = ["chat_stream", "search", "ingest", "release_decision"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
