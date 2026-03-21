"""Core graph nodes for the Scientific Loop.

Each function accepts the current ``ScientificState`` and returns a partial
state dict that LangGraph merges back into the graph state.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from .state import ScientificState

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------
_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


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


# ---------------------------------------------------------------------------
# 1. Planner
# ---------------------------------------------------------------------------


def planner(state: ScientificState) -> dict[str, Any]:
    """Analyse the task and produce a mathematical spec + file-tree plan.

    Uses the ``reflection`` field (if present) to incorporate learnings from
    previous failed iterations.
    """
    llm = get_llm()

    user_parts: list[str] = [f"## Task\n{state['task_description']}"]
    if state.get("reflection"):
        user_parts.append(f"## Previous error analysis\n{state['reflection']}")
    if state.get("mathematical_constants"):
        user_parts.append(
            f"## Constants\n{state['mathematical_constants']}"
        )
    if state.get("skills_context"):
        user_parts.append(state["skills_context"])

    prompt = _load_prompt("EXPLORER_PROMPT_PLANNER", "planner.md")
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="\n\n".join(user_parts)),
    ])
    return {"plan": response.content}


# ---------------------------------------------------------------------------
# 2. Coder
# ---------------------------------------------------------------------------

def _write_code_drafts(
    code_drafts: dict[str, str],
    output_dir: str,
) -> None:
    """Write *code_drafts* to *output_dir* with path-traversal protection.

    Every resolved target path must remain inside *output_dir*.  A
    ``ValueError`` is raised for any draft whose relative path escapes the
    output root (e.g. ``../escape.py``).
    """
    root = Path(output_dir).resolve()
    for rel_path, source in code_drafts.items():
        target = (root / rel_path).resolve()
        if not target.is_relative_to(root):
            raise ValueError(
                f"Path traversal blocked: {rel_path!r} resolves outside "
                f"output directory {root}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)


def coder(state: ScientificState) -> dict[str, Any]:
    """Generate Python source code from the current plan.

    Returns
    -------
    dict
        ``code_drafts``: ``{filepath: source_code}`` mapping extracted from
        the LLM response.
    """
    llm = get_llm()
    prompt = _load_prompt("EXPLORER_PROMPT_CODER", "coder.md")

    user_parts: list[str] = [f"## Plan\n{state['plan']}"]
    if state.get("skills_context"):
        user_parts.append(state["skills_context"])

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="\n\n".join(user_parts)),
    ])

    code_drafts = _parse_code_blocks(response.content)

    # In write mode, persist files to the output directory
    output_dir = state.get("output_dir", "")
    if output_dir:
        _write_code_drafts(code_drafts, output_dir)

    return {"code_drafts": code_drafts}


def _parse_code_blocks(text: str) -> dict[str, str]:
    """Extract fenced code blocks whose info-strings look like file paths."""
    drafts: dict[str, str] = {}
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```") and "/" in line:
            path = line.strip().removeprefix("```").strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            drafts[path] = "".join(code_lines)
        i += 1
    return drafts


# ---------------------------------------------------------------------------
# 3. Verifier
# ---------------------------------------------------------------------------


def _run_pytest(root: Path) -> list[str]:
    """Run ``pytest`` inside *root* and return failure logs (empty on success)."""
    result = subprocess.run(  # noqa: S603
        ["python", "-m", "pytest", str(root), "-v", "--tb=short"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(root),
    )
    if result.returncode != 0:
        return [result.stdout + "\n" + result.stderr]
    return []


def verifier(state: ScientificState) -> dict[str, Any]:
    """Run ``pytest`` against the generated code.

    In **sandbox mode** (default, no ``output_dir``), code drafts are written
    to a temporary directory that is cleaned up after the run.

    In **write mode** (``output_dir`` is set), ``pytest`` runs directly inside
    the output directory — the same directory the coder already wrote to.
    """
    code_drafts: dict[str, str] = state.get("code_drafts", {})
    if not code_drafts:
        return {"test_logs": ["No code drafts to verify."]}

    output_dir = state.get("output_dir", "")

    if output_dir:
        # Write mode — run pytest in the real output directory
        root = Path(output_dir).resolve()
        logs = _run_pytest(root)
    else:
        # Sandbox mode — temp dir with auto-cleanup
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for rel_path, source in code_drafts.items():
                target = root / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source)

            # Generate a minimal conftest so imports resolve
            (root / "conftest.py").write_text(
                "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parent))\n"
            )

            logs = _run_pytest(root)

    return {
        "test_logs": logs,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


# ---------------------------------------------------------------------------
# 4. Reflector
# ---------------------------------------------------------------------------

def reflector(state: ScientificState) -> dict[str, Any]:
    """Analyse test failures and extract key findings for ground_truth.md."""
    llm = get_llm()

    reflector_prompt = _load_prompt("EXPLORER_PROMPT_REFLECTOR", "reflector.md")
    findings_prompt = _load_prompt("EXPLORER_PROMPT_FINDINGS", "findings.md")

    log_text = "\n---\n".join(state.get("test_logs", []))
    response = llm.invoke([
        SystemMessage(content=reflector_prompt),
        HumanMessage(content=f"## Test logs\n```\n{log_text}\n```"),
    ])
    reflection = response.content

    # Extract key findings
    findings_response = llm.invoke([
        SystemMessage(content=findings_prompt),
        HumanMessage(
            content=f"## Reflection\n{reflection}\n\n## Test logs\n```\n{log_text}\n```"
        ),
    ])

    existing = list(state.get("ground_truth", []))
    raw = findings_response.content.strip()
    if raw != "NONE":
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("- "):
                existing.append(line)

    return {"reflection": reflection, "ground_truth": existing}
