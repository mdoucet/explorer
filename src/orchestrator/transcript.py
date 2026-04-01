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
    metadata: dict[str, Any] | None = None,
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
        metadata: Optional structured data for trajectory tracking.
            Each node type populates this differently — see
            ``_compute_diff_summary`` and individual node implementations.

    Returns:
        A dict suitable for appending to ``state["transcript"]``.
    """
    if not summary:
        first_line = content.split("\n", 1)[0]
        summary = first_line[:200]
    entry: dict[str, Any] = {
        "role": role,
        "content": content,
        "node": node,
        "step": step,
        "phase": phase,
        "summary": summary,
    }
    if metadata:
        entry["metadata"] = metadata
    return entry


def _compute_diff_summary(
    old_drafts: dict[str, str],
    new_drafts: dict[str, str],
) -> dict[str, Any]:
    """Compute a structured summary of what changed between two code snapshots.

    Args:
        old_drafts: Previous ``code_drafts`` mapping (path → source).
        new_drafts: Current ``code_drafts`` mapping (path → source).

    Returns:
        A dict with keys ``files_added``, ``files_deleted``,
        ``files_modified``, ``line_deltas`` (per-file +N/−M), and a
        human-readable ``summary`` string.
    """
    old_paths = set(old_drafts)
    new_paths = set(new_drafts)

    added = sorted(new_paths - old_paths)
    deleted = sorted(old_paths - new_paths)
    common = sorted(old_paths & new_paths)
    modified: list[str] = []
    line_deltas: dict[str, str] = {}

    total_plus = 0
    total_minus = 0

    for path in common:
        old_lines = old_drafts[path].splitlines()
        new_lines = new_drafts[path].splitlines()
        if old_drafts[path] != new_drafts[path]:
            modified.append(path)
            plus = max(0, len(new_lines) - len(old_lines))
            minus = max(0, len(old_lines) - len(new_lines))
            # Heuristic: count changed lines more precisely
            plus = sum(1 for l in new_lines if l not in old_lines)  # noqa: E741
            minus = sum(1 for l in old_lines if l not in new_lines)  # noqa: E741
            line_deltas[path] = f"+{plus}/−{minus}"
            total_plus += plus
            total_minus += minus

    for path in added:
        lc = len(new_drafts[path].splitlines())
        line_deltas[path] = f"+{lc}"
        total_plus += lc

    for path in deleted:
        lc = len(old_drafts[path].splitlines())
        line_deltas[path] = f"−{lc}"
        total_minus += lc

    parts: list[str] = []
    n_changed = len(modified) + len(added) + len(deleted)
    parts.append(
        f"{n_changed} file(s) changed (+{total_plus}/−{total_minus} lines)"
    )
    if added:
        parts.append(f"added: {', '.join(added)}")
    if deleted:
        parts.append(f"deleted: {', '.join(deleted)}")
    if modified:
        mod_details = [f"{p} ({line_deltas[p]})" for p in modified]
        parts.append(f"modified: {', '.join(mod_details)}")

    return {
        "files_added": added,
        "files_deleted": deleted,
        "files_modified": modified,
        "line_deltas": line_deltas,
        "summary": "; ".join(parts),
    }


def _condensed_summary(entry: dict[str, Any], meta: dict[str, Any]) -> str:
    """Build a rich one-line summary for an old transcript entry.

    Uses structured ``metadata`` when available; falls back to the
    plain ``summary`` field.
    """
    meta_type = meta.get("type", "")

    if meta_type == "code_change" and "diff_summary" in meta:
        return meta["diff_summary"]
    if meta_type == "test_result":
        if meta.get("passed"):
            return "Tests: All passed ✓"
        detail = ""
        if meta.get("error_repeat_count", 0) > 1:
            detail = f" (same error ×{meta['error_repeat_count']})"
        if meta.get("collection_error"):
            return f"Tests: collection error{detail}"
        return f"Tests: FAILED{detail}"
    if meta_type == "plan":
        n = len(meta.get("phases", []))
        label = "Replan" if meta.get("is_replan") else "Plan"
        return f"{label}: {n} phase(s)"
    if meta_type == "analysis":
        action = meta.get("action", "?")
        return f"Reflector → {action}"
    if meta_type == "phase_advance":
        return (
            f"Phase {meta.get('from_phase', '?')+1} → "
            f"Phase {meta.get('to_phase', '?')+1}"
        )

    return entry.get("summary") or entry["content"][:200]


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
            meta = entry.get("metadata", {})
            summary = _condensed_summary(entry, meta)
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
