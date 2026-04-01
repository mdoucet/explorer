"""Reflector node — analyses test failures and extracts findings."""

from __future__ import annotations

import time as _time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from . import _shared
from ._shared import (
    _format_code_listing,
    _invoke_llm,
    _load_prompt,
    make_llm_call_record,
)
from ..state import ScientificState
from ..transcript import format_history, make_entry


def _extract_findings(text: str) -> tuple[str, list[str]]:
    """Split a reflector response into analysis and key findings.

    The reflector prompt asks the LLM to end its response with a
    ``## Key Findings`` section.  This helper splits on that header
    and extracts ``"- …"`` bullet lines as individual findings.

    Returns ``(analysis_text, findings_list)``.
    """
    # Strip ## Action section first (if present) before parsing findings
    action_marker = "## Action"
    action_idx = text.find(action_marker)
    if action_idx >= 0:
        text = text[:action_idx]

    marker = "## Key Findings"
    idx = text.find(marker)
    if idx < 0:
        return text.strip(), []

    analysis = text[:idx].strip()
    raw_findings = text[idx + len(marker):].strip()

    if raw_findings.upper() == "NONE":
        return analysis, []

    findings: list[str] = []
    for line in raw_findings.splitlines():
        line = line.strip()
        if line.startswith("- "):
            findings.append(line)

    return analysis, findings


def _extract_action(text: str) -> str:
    """Parse the ``## Action`` recommendation from a reflector response.

    Returns ``"replan"`` or ``"retry"`` (default).
    """
    marker = "## Action"
    idx = text.find(marker)
    if idx < 0:
        return "retry"
    raw = text[idx + len(marker):].strip().splitlines()
    if raw:
        first = raw[0].strip().upper()
        if "REPLAN" in first:
            return "replan"
    return "retry"


def reflector(state: ScientificState) -> dict[str, Any]:
    """Analyse test failures and extract key findings for ground_truth.md.

    Makes a single LLM call that produces both the debugging analysis
    and a ``## Key Findings`` section.  Findings are parsed locally and
    deduplicated against existing ``ground_truth``.
    """
    llm = _shared.get_llm()

    reflector_prompt = _load_prompt("EXPLORER_PROMPT_REFLECTOR", "reflector.md")

    log_text = "\n---\n".join(state.get("test_logs", []))

    # Include failing source code so the reflector can see what went wrong
    code_drafts: dict[str, str] = state.get("code_drafts", {})
    clean = set(state.get("clean_files") or [])
    failing_sources = _format_code_listing(
        code_drafts, only_failing=True, clean_files=clean,
    )

    source_section = ""
    if failing_sources:
        source_section = (
            "\n\n## Source code of files with errors\n"
            + "\n\n".join(failing_sources)
        )

    reflector_parts: list[str] = []

    # Inject run history so the reflector sees what was tried before
    # (replaces ad-hoc ground_truth, prior_reflection injection).
    history = format_history(state.get("transcript") or [])
    if history:
        reflector_parts.append(history)

    reflector_parts.append(f"## Test logs\n```\n{log_text}\n```")
    if source_section:
        reflector_parts.append(source_section)

    reflector_parts.append(
        "## IMPORTANT: direction of fixes\n"
        "Tests are the SPECIFICATION.  NEVER suggest changing test files.\n"
        "Always fix the implementation (source code) to match the tests.\n"
        "If a test calls `f(V0=50)` but the function uses `depth`, change "
        "the function parameter name to `V0`, NOT the test."
    )

    reflector_user_msg = "\n\n".join(reflector_parts)
    messages = [
        SystemMessage(content=reflector_prompt),
        HumanMessage(content=reflector_user_msg),
    ]
    t0 = _time.monotonic()
    response = _invoke_llm(llm, messages)
    duration = _time.monotonic() - t0

    llm_calls: list[dict] = []
    llm_calls.append(make_llm_call_record(
        node="reflector",
        messages=messages,
        response=response,
        duration_s=duration,
    ))

    # Single-call: parse analysis, key findings, and action from the response.
    full_response = response.content
    reflection, new_raw_findings = _extract_findings(full_response)
    action = _extract_action(full_response)

    # Deduplicate against existing ground_truth
    existing = list(state.get("ground_truth", []))
    seen = set(existing)
    for finding in new_raw_findings:
        if finding not in seen:
            existing.append(finding)
            seen.add(finding)

    # Build transcript entry with analysis and any new findings
    original_gt = set(state.get("ground_truth") or [])
    new_findings = [f for f in existing if f not in original_gt]
    transcript_content = reflection
    if new_findings:
        transcript_content += "\n\n## Key Findings\n" + "\n".join(new_findings)
    transcript: list[dict] = []
    transcript.append(make_entry(
        "ai", transcript_content,
        node="reflector",
        step=state.get("iteration_count", 0),
        phase=state.get("current_phase", 0),
        summary=f"Analysis: {reflection.split(chr(10), 1)[0][:150]}",
    ))
    return {
        "reflection": reflection,
        "ground_truth": existing,
        "_reflector_action": action,
        "_prompt_summary": reflector_user_msg,
        "transcript": transcript,
        "_llm_calls": llm_calls,
    }


# ---------------------------------------------------------------------------
# Lightweight auto-reflect (no LLM call — for tool-calling mode)
# ---------------------------------------------------------------------------

_REPLAN_ERROR_REPEAT_THRESHOLD = 3


def auto_reflect(state: ScientificState) -> dict[str, Any]:
    """Lightweight reflector that uses error fingerprinting instead of an LLM.

    When the coder has tool-calling capabilities, it already self-corrects.
    This node just:
    1. Checks if the same error has repeated enough times to warrant a replan.
    2. Passes test log summaries as ``reflection`` context for the coder.
    3. No LLM call — zero extra tokens.
    """
    error_repeat = state.get("_error_repeat_count", 0)
    phase_iter = state.get("_phase_iteration_count", 0)

    # Decide: replan or retry
    if error_repeat >= _REPLAN_ERROR_REPEAT_THRESHOLD:
        action = "replan"
        summary = (
            f"Same error repeated {error_repeat} times — "
            "recommending replan with a different approach."
        )
    else:
        action = "retry"
        summary = f"Retry (attempt {phase_iter}, error repeat {error_repeat})"

    # Build a compact reflection from test logs (no LLM needed)
    logs = state.get("test_logs") or []
    log_text = "\n---\n".join(logs)
    # Truncate to keep context budget manageable
    reflection = log_text[:3000] if log_text else "Tests failed (no details)."

    transcript: list[dict] = []
    transcript.append(make_entry(
        "ai", f"Auto-reflect: {summary}\n\n{reflection[:500]}",
        node="auto_reflect",
        step=state.get("iteration_count", 0),
        phase=state.get("current_phase", 0),
        summary=f"Auto-reflect: {summary}",
    ))

    return {
        "reflection": reflection,
        "_reflector_action": action,
        "transcript": transcript,
    }
