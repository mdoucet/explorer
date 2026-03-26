"""Planner and phase-advancement nodes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from . import _shared
from ._shared import (
    _invoke_llm,
    _load_prompt,
)
from ..state import ScientificState
from ..transcript import format_history, make_entry

# ---------------------------------------------------------------------------
# Plan phase helpers
# ---------------------------------------------------------------------------

_PHASE_RE = re.compile(
    r"^##\s+Phase\s+(\d+)\s*:\s*(.+)$",
    re.MULTILINE,
)


def _parse_plan_phases(text: str) -> list[dict[str, Any]]:
    """Parse ``## Phase N: Title`` sections from the planner's Markdown output.

    Returns a list of dicts with keys ``id``, ``title``, ``description``,
    ``status``, and ``files``.  Falls back to a single phase containing the
    entire text if no ``## Phase`` headers are found.
    """
    matches = list(_PHASE_RE.finditer(text))
    if not matches:
        return [{
            "id": 1,
            "title": "Implementation",
            "description": text.strip(),
            "status": "pending",
            "files": [],
        }]

    phases: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # Extract "Files:" line if present
        files: list[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("files:"):
                raw = stripped.split(":", 1)[1].strip()
                files = [f.strip() for f in raw.split(",") if f.strip()]
                break

        phases.append({
            "id": int(match.group(1)),
            "title": match.group(2).strip(),
            "description": body,
            "status": "pending",
            "files": files,
        })

    return phases


def _write_plan_artifact(phases: list[dict[str, Any]], current: int) -> None:
    """Write (or overwrite) ``plan.md`` with the phased plan and status."""
    lines = ["# Plan\n"]
    for phase in phases:
        pid = phase["id"]
        title = phase["title"]
        status = phase["status"]
        if status == "completed":
            check = "x"
        elif pid == phases[current]["id"]:
            check = "~"  # in-progress marker
        else:
            check = " "
        lines.append(f"## [{check}] Phase {pid}: {title}\n")
        lines.append(f"**Status:** {status}\n")
        if phase.get("files"):
            lines.append(f"**Files:** {', '.join(phase['files'])}\n")
        lines.append(phase["description"])
        lines.append("")
    lines.append("")
    Path("plan.md").write_text("\n".join(lines))


def _pick_replan_phase(
    new_phases: list[dict[str, Any]], target_title: str
) -> dict[str, Any] | None:
    """Select the best phase from a replan response.

    During replanning we ask the LLM for a single revised phase, but it may
    still emit a full multi-phase plan.  This helper picks the right one:

    1. If only one phase was parsed, return it.
    2. If multiple phases, prefer the one whose title best matches
       *target_title* (case-insensitive substring).
    3. Skip any phase whose description looks like pure scaffolding
       (contains "stub" + "pass" and no algorithm content).
    4. Fall back to the first non-scaffolding phase, or the first phase.
    """
    if not new_phases:
        return None
    if len(new_phases) == 1:
        return new_phases[0]

    target_lower = target_title.lower()

    # Try title match first
    for phase in new_phases:
        if target_lower in phase["title"].lower():
            return phase

    # Filter out scaffolding phases
    non_scaffold: list[dict[str, Any]] = []
    for phase in new_phases:
        desc = phase["description"].lower()
        is_scaffolding = (
            "stub" in desc
            and "pass" in desc
            and "scaffolding" in phase["title"].lower()
        )
        if not is_scaffolding:
            non_scaffold.append(phase)

    if non_scaffold:
        return non_scaffold[0]

    # All phases look like scaffolding — return last one (most likely
    # to contain implementation content)
    return new_phases[-1]


# ---------------------------------------------------------------------------
# 1. Planner
# ---------------------------------------------------------------------------


def planner(state: ScientificState) -> dict[str, Any]:
    """Analyse the task and produce a phased implementation plan.

    Called in two contexts:

    1. **Initial planning** (``plan_phases`` is empty) — produce a fresh
       phased plan from the task description.
    2. **Replanning** (``plan_phases`` exists and ``reflection`` is set) —
       the coder has been stuck on the same error for multiple iterations.
       Revise the plan for the *current* phase, taking the error analysis
       and test logs into account so the coder can try a different approach.
    """
    llm = _shared.get_llm()

    is_replan = bool(state.get("plan_phases")) and bool(state.get("reflection"))

    user_parts: list[str] = [f"## Task\n{state['task_description']}"]
    if state.get("mathematical_constants"):
        user_parts.append(
            f"## Constants\n{state['mathematical_constants']}"
        )

    # Inject run history so the planner sees what happened before
    history = format_history(state.get("transcript") or [])
    if history:
        user_parts.append(history)

    if is_replan:
        phases = state["plan_phases"]
        current = state.get("current_phase", 0)
        phase = phases[current]
        original_title = phase["title"]
        user_parts.append(
            f"## ⚠️ REPLAN REQUEST — Phase {phase['id']}: {original_title}\n"
            f"The coder has been stuck on this phase for multiple iterations "
            f"with the same error.  Revise the plan for THIS phase only.\n\n"
            f"**IMPORTANT: Do NOT regress the scope.** The phase title MUST "
            f"remain \"{original_title}\". Do NOT change it back to scaffolding "
            f"or reduce the scope.  Suggest a DIFFERENT implementation "
            f"approach for the same deliverables.\n\n"
            f"**Current phase description:**\n{phase['description']}"
        )

    if state.get("skills_context"):
        user_parts.append(state["skills_context"])

    # Use a dedicated replan prompt that asks for a SINGLE revised phase,
    # not a full multi-phase plan.  This prevents the LLM from regenerating
    # scaffolding when it should be revising the solver/implementation.
    if is_replan:
        prompt = _load_prompt("EXPLORER_PROMPT_PLANNER_REPLAN", "planner_replan.md")
    else:
        prompt = _load_prompt("EXPLORER_PROMPT_PLANNER", "planner.md")
    user_message = "\n\n".join(user_parts)
    response = _invoke_llm(llm, [
        SystemMessage(content=prompt),
        HumanMessage(content=user_message),
    ])

    raw_plan = response.content

    if is_replan:
        # Only update the current phase description — keep other phases intact
        phases = list(state["plan_phases"])
        current = state.get("current_phase", 0)
        original_title = phases[current]["title"]
        new_phases = _parse_plan_phases(raw_plan)
        # Try to find a phase matching the current title (case-insensitive)
        best = _pick_replan_phase(new_phases, original_title)
        if best:
            phases[current]["description"] = best["description"]
            # Preserve the original title to prevent scope regression
            phases[current]["title"] = original_title
            if best.get("files"):
                phases[current]["files"] = best["files"]
        plan_text = phases[current]["description"]
        _write_plan_artifact(phases, current)
        transcript = list(state.get("transcript") or [])
        transcript.append(make_entry(
            "ai", f"Revised plan for Phase {current + 1}:\n{plan_text}",
            node="planner", step=state.get("iteration_count", 0), phase=current,
            summary=f"Replanned Phase {current + 1}: {original_title}",
        ))
        return {
            "plan": plan_text,
            "plan_phases": phases,
            "_prompt_summary": user_message,
            "_error_repeat_count": 0,  # reset after replan
            "_replan_count": state.get("_replan_count", 0) + 1,
            "transcript": transcript,
        }

    plan_phases = _parse_plan_phases(raw_plan)
    current_phase = 0

    # Set `plan` to the current phase description for backward compatibility
    plan_text = plan_phases[current_phase]["description"]

    _write_plan_artifact(plan_phases, current_phase)

    transcript = list(state.get("transcript") or [])
    phase_titles = [p["title"] for p in plan_phases]
    transcript.append(make_entry(
        "ai", raw_plan,
        node="planner", step=0, phase=0,
        summary=f"Plan: {len(plan_phases)} phases — {', '.join(phase_titles)}",
    ))
    return {
        "plan": plan_text,
        "plan_phases": plan_phases,
        "current_phase": current_phase,
        "_prompt_summary": user_message,
        "transcript": transcript,
    }


# ---------------------------------------------------------------------------
# Phase advancement
# ---------------------------------------------------------------------------


def advance_phase(state: ScientificState) -> dict[str, Any]:
    """Mark the current phase as completed and advance to the next one.

    Updates ``plan`` to the next phase's description so the coder picks
    it up on the next iteration.  Also re-writes the plan artefact.
    """
    phases = list(state.get("plan_phases") or [])
    current = state.get("current_phase", 0)

    # Mark current phase completed
    phases[current]["status"] = "completed"

    next_idx = current + 1
    phases[next_idx]["status"] = "in-progress"
    plan_text = phases[next_idx]["description"]

    _write_plan_artifact(phases, next_idx)

    transcript = list(state.get("transcript") or [])
    prev_title = phases[current]["title"]
    next_title = phases[next_idx]["title"]
    transcript.append(make_entry(
        "human",
        f"Phase {current + 1} '{prev_title}' completed. "
        f"Moving to Phase {next_idx + 1}: '{next_title}'.",
        node="advance_phase",
        step=state.get("iteration_count", 0), phase=next_idx,
    ))

    return {
        "plan": plan_text,
        "plan_phases": phases,
        "current_phase": next_idx,
        "reflection": "",  # clear stale reflection for new phase
        "ground_truth": [],  # clear stale findings for new phase
        "_prev_error_fingerprint": "",  # reset for new phase
        "_error_repeat_count": 0,
        "_replan_count": 0,
        "_phase_iteration_count": 0,
        "_collection_error": False,
        "clean_files": [],
        "transcript": transcript,
    }
