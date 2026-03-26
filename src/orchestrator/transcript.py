"""Conversation transcript for the Scientific Loop.

Provides a growing list of structured messages that records every
significant action in a run — plans, code generation, test results,
and error analyses.  Nodes append entries; LLM-calling nodes format
the transcript as a "Run History" section in their prompt so the model
has continuity across iterations instead of starting from scratch.
"""

from __future__ import annotations

from typing import Any

# How many recent entries to show in full; older ones are condensed
# to their one-line summary.
DEFAULT_MAX_RECENT = 8


def make_entry(
    role: str,
    content: str,
    *,
    node: str = "",
    step: int = 0,
    phase: int = 0,
    summary: str = "",
) -> dict[str, Any]:
    """Create a transcript entry.

    Args:
        role: ``"human"`` for environment messages (test results, phase
            transitions) or ``"ai"`` for agent outputs (plans, code,
            analyses).
        content: Full message content.
        node: Which graph node produced this entry.
        step: Iteration count at time of creation.
        phase: Current phase index.
        summary: One-line summary for context windowing.  Falls back to
            the first line of *content* (truncated to 200 chars).

    Returns:
        A dict suitable for appending to ``state["transcript"]``.
    """
    if not summary:
        first_line = content.split("\n", 1)[0]
        summary = first_line[:200]
    return {
        "role": role,
        "content": content,
        "node": node,
        "step": step,
        "phase": phase,
        "summary": summary,
    }


def format_history(
    transcript: list[dict[str, Any]],
    max_recent: int = DEFAULT_MAX_RECENT,
) -> str:
    """Format transcript as a ``## Run History`` section for an LLM prompt.

    Recent entries (last *max_recent*) are shown in full.  Older entries
    are collapsed to their one-line summary.  Returns an empty string
    when the transcript is empty.
    """
    if not transcript:
        return ""

    parts: list[str] = ["## Run History"]

    if len(transcript) > max_recent:
        old = transcript[:-max_recent]
        recent = transcript[-max_recent:]

        parts.append("### Earlier iterations (condensed)")
        for entry in old:
            node = entry.get("node", "?")
            step = entry.get("step", "")
            summary = entry.get("summary") or entry["content"][:200]
            prefix = f"[step {step}, {node}]" if step else f"[{node}]"
            parts.append(f"- {prefix} {summary}")
        parts.append("")
    else:
        recent = transcript

    for entry in recent:
        node = entry.get("node", "")
        step = entry.get("step", "")
        if step and node:
            header = f"**[step {step} — {node}]**"
        elif node:
            header = f"**[{node}]**"
        else:
            header = ""
        parts.append(f"{header}\n{entry['content']}")

    return "\n\n".join(parts)
