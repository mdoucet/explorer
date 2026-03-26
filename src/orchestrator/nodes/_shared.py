"""Shared LLM infrastructure and prompt helpers for graph nodes."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: F401
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from ..state import ScientificState  # noqa: F401
from ..transcript import format_history, make_entry  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------
_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"


def _load_prompt(env_var: str, default_filename: str) -> str:
    """Load a system prompt from a file.

    Checks the environment variable ``env_var`` first; falls back to the
    default file inside the ``prompts/`` directory.
    """
    path_str = os.environ.get(env_var)
    if path_str:
        path = Path(path_str)
    else:
        path = _PROMPTS_DIR / default_filename

    return path.read_text().strip()

# ---------------------------------------------------------------------------
# Shared LLM (lazily overridden in tests via monkeypatch)
# ---------------------------------------------------------------------------
_llm: ChatOpenAI | ChatOllama | None = None
_llm_provider: str = "ollama"
_llm_model: str = "qwen2.5-coder:32b"
_llm_base_url: str | None = None
_llm_temperature: float = 0.0


def configure_llm(
    provider: str = "ollama",
    model: str = "qwen2.5-coder:32b",
    base_url: str | None = None,
    temperature: float = 0.0,
) -> None:
    """Set the LLM provider and model for subsequent ``get_llm()`` calls.

    Resets the cached instance so the next call creates a fresh one.
    """
    global _llm, _llm_provider, _llm_model, _llm_base_url, _llm_temperature  # noqa: PLW0603
    _llm_provider = provider
    _llm_model = model
    _llm_base_url = base_url
    _llm_temperature = temperature
    _llm = None  # force re-creation on next get_llm()


def get_llm() -> ChatOpenAI | ChatOllama:
    """Return the shared LLM instance, creating it on first call."""
    global _llm  # noqa: PLW0603
    if _llm is None:
        if _llm_provider == "openai":
            kwargs: dict[str, Any] = {"model": _llm_model, "temperature": _llm_temperature}
            if _llm_base_url:
                kwargs["base_url"] = _llm_base_url
            _llm = ChatOpenAI(**kwargs)
        else:
            kwargs = {"model": _llm_model, "temperature": _llm_temperature}
            if _llm_base_url:
                kwargs["base_url"] = _llm_base_url
            _llm = ChatOllama(**kwargs)
    return _llm


def _invoke_llm(llm: Any, messages: list, *, max_retries: int = 3, retry_delay: float = 5.0) -> Any:
    """Invoke the LLM with streaming and retry on transient errors.

    Uses ``llm.stream()`` to avoid first-byte timeouts on large prompts
    (the server proxy drops connections that don't produce a token within
    ~60s).  Falls back to ``llm.invoke()`` for test mocks that don't
    support streaming.
    """
    if not hasattr(llm, "stream"):
        return llm.invoke(messages)

    for attempt in range(1, max_retries + 1):
        try:
            chunks: list[Any] = []
            for chunk in llm.stream(messages):
                chunks.append(chunk)
            if not chunks:
                return llm.invoke(messages)
            result = chunks[0]
            for c in chunks[1:]:
                result = result + c
            return result
        except Exception as exc:
            err_str = str(exc).lower()
            transient = any(k in err_str for k in (
                "incomplete chunked read", "peer closed", "connection",
                "timeout", "502", "503", "504", "remotedisconnected",
                "remoteprotocolerror",
            ))
            if transient and attempt < max_retries:
                logger.warning(
                    "Transient LLM error (attempt %d/%d): %s — retrying in %.0fs",
                    attempt, max_retries, exc, retry_delay,
                )
                import time
                time.sleep(retry_delay)
                continue
            raise


def _format_code_listing(
    code_drafts: dict[str, str],
    *,
    clean_files: set[str] | None = None,
    only_failing: bool = False,
) -> list[str]:
    """Format code_drafts as fenced Markdown blocks for LLM prompts.

    Parameters
    ----------
    code_drafts : dict
        Mapping of file paths to source code.
    clean_files : set or None
        If given, clean files are annotated with a ✅ marker.
    only_failing : bool
        If True, skip files present in *clean_files*.

    Returns a list of formatted sections (one per .py file).
    """
    sections: list[str] = []
    clean = clean_files or set()
    for fpath in sorted(code_drafts):
        if not fpath.endswith(".py"):
            continue
        if only_failing and fpath in clean:
            continue
        source = code_drafts[fpath]
        if clean_files is not None and fpath in clean:
            sections.append(
                f"### {fpath} ✅ (NO ERRORS — do NOT modify)\n"
                f"```python\n{source}\n```"
            )
        else:
            sections.append(f"### {fpath}\n```python\n{source}\n```")
    return sections
