"""Shared fixtures and markers for the Explorer test suite."""

from __future__ import annotations

import urllib.error
import urllib.request
import json

import pytest


def _ollama_available(model: str = "qwen2.5-coder:32b") -> tuple[bool, str]:
    """Check whether Ollama is reachable and the required model is pulled.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` when ready, ``(False, reason)`` otherwise.
    """
    try:
        req = urllib.request.Request(  # noqa: S310 — localhost only
            "http://localhost:11434/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, TimeoutError):
        return False, "Ollama is not reachable at localhost:11434"

    available_models = [m.get("name", "") for m in data.get("models", [])]
    # Ollama may report the model with or without a tag suffix
    if not any(model in m for m in available_models):
        return False, (
            f"Model '{model}' not found in Ollama. "
            f"Available: {available_models}"
        )
    return True, ""


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip tests marked with ``@pytest.mark.ollama`` when Ollama is unavailable."""
    ok, reason = _ollama_available()
    if ok:
        return
    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        if "ollama" in item.keywords:
            item.add_marker(skip_marker)
