"""Live reporting and chat-log persistence for the Scientific Loop.

The :func:`report_node` function prints a formatted summary for each node
as it completes during ``app.stream()``.  When a *chat_dir* is provided,
the full output of every node is also written to disk so the user can
review the agent's reasoning after the run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from orchestrator.state import ScientificState

# ---------------------------------------------------------------------------
# Console styles
# ---------------------------------------------------------------------------
_NODE_STYLE: dict[str, tuple[str, str]] = {
    "planner":       ("🔬", "fg=cyan"),
    "coder":         ("💻", "fg=yellow"),
    "verifier":      ("✅", "fg=green"),
    "reflector":     ("🔄", "fg=magenta"),
    "advance_phase": ("⏩", "fg=blue"),
}


def report_node(node_name: str, update: dict[str, Any]) -> None:
    """Print a concise, coloured summary of *update* from *node_name*."""
    emoji, colour = _NODE_STYLE.get(node_name, ("▸", "fg=white"))
    header = click.style(f" {node_name.capitalize()}", fg=colour.split("=")[1], bold=True)
    click.echo(f"\n{emoji}{header}")

    # Phase banner for planner / coder / advance_phase
    phases = update.get("plan_phases") or []
    current = update.get("current_phase")
    if phases and current is not None:
        total = len(phases)
        title = phases[current]["title"] if current < total else "?"
        click.echo(click.style(f"  Phase {current + 1}/{total}: {title}", dim=True))

    if node_name == "planner":
        plan = update.get("plan", "")
        # Show first few lines as a preview
        preview = "\n".join(plan.splitlines()[:8])
        click.echo(click.style("  Plan preview:", dim=True))
        for line in preview.splitlines():
            click.echo(f"  │ {line}")
        remaining = len(plan.splitlines()) - 8
        if remaining > 0:
            click.echo(click.style(f"  │ … ({remaining} more lines)", dim=True))

    elif node_name == "coder":
        drafts = update.get("code_drafts", {})
        if drafts:
            click.echo(click.style(f"  Generated {len(drafts)} file(s):", dim=True))
            for path, source in drafts.items():
                n_lines = len(source.splitlines())
                click.echo(f"  │ {path}  ({n_lines} lines)")
        else:
            raw = update.get("coder_raw_response", "")
            click.echo(click.style("  ⚠ No code blocks parsed from LLM response", fg="red"))
            if raw:
                preview = "\n".join(raw.splitlines()[:5])
                click.echo(click.style("  Raw response preview:", dim=True))
                for line in preview.splitlines():
                    click.echo(f"  │ {line}")

    elif node_name == "verifier":
        logs = update.get("test_logs", [])
        iteration = update.get("iteration_count", "?")
        if not logs:
            click.echo(click.style(f"  All tests passed (iteration {iteration})", fg="green"))
        else:
            click.echo(click.style(f"  Tests FAILED (iteration {iteration})", fg="red"))
            # Show last 15 lines of failure output
            for log_block in logs:
                tail = log_block.strip().splitlines()[-15:]
                for line in tail:
                    click.echo(f"  │ {line}")

    elif node_name == "reflector":
        reflection = update.get("reflection", "")
        preview = "\n".join(reflection.splitlines()[:6])
        click.echo(click.style("  Error analysis:", dim=True))
        for line in preview.splitlines():
            click.echo(f"  │ {line}")
        findings = update.get("ground_truth", [])
        if findings:
            click.echo(click.style(f"  Findings ({len(findings)}):", dim=True))
            for f in findings[-3:]:
                click.echo(f"  │ {f}")

    elif node_name == "advance_phase":
        new_phase = update.get("current_phase", "?")
        ap_phases = update.get("plan_phases") or []
        if ap_phases and isinstance(new_phase, int) and new_phase < len(ap_phases):
            title = ap_phases[new_phase]["title"]
            click.echo(click.style(
                f"  Advancing to phase {new_phase + 1}: {title}", fg="blue", bold=True,
            ))


# ---------------------------------------------------------------------------
# Chat-log persistence
# ---------------------------------------------------------------------------

class ChatLogger:
    """Writes node outputs to a timestamped directory for post-run review.

    Directory layout::

        chat_dir/
        ├── 00_run_info.md
        ├── 01_planner.md
        ├── 02_coder.md
        ├── 03_verifier.md
        ├── 04_reflector.md
        ├── 05_planner.md
        └── summary.json
    """

    def __init__(
        self,
        chat_dir: str,
        *,
        task: str = "",
        skills: list[str] | None = None,
        provider: str = "",
        model: str = "",
        max_iterations: int = 0,
    ) -> None:
        self._dir = Path(chat_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Remove stale log files from a previous run
        for old in self._dir.glob("[0-9][0-9]_*.md"):
            old.unlink()
        for old in self._dir.glob("summary.json"):
            old.unlink()
        self._step = 0
        self._start = datetime.now(timezone.utc)
        self._task = task
        self._skills = skills or []
        self._provider = provider
        self._model = model
        self._max_iterations = max_iterations

        # Write run info file
        self._write_run_info()

    def _write_run_info(self) -> None:
        """Write a ``00_run_info.md`` file with run configuration."""
        lines = [
            "# Run Info\n",
            f"**Started:** {self._start.strftime('%Y-%m-%d %H:%M:%S UTC')}\n",
            f"**Model:** {self._provider} / {self._model}\n",
            f"**Max iterations:** {self._max_iterations}\n",
        ]
        if self._skills:
            lines.append(f"**Skills matched:** {', '.join(self._skills)}\n")
        else:
            lines.append("**Skills matched:** _(none)_\n")
        if self._task:
            lines.append("## Task\n")
            lines.append(self._task + "\n")
        filename = f"{self._step:02d}_run_info.md"
        self._step += 1
        (self._dir / filename).write_text("\n".join(lines) + "\n")

    def log_node(self, node_name: str, update: dict[str, Any]) -> None:
        """Persist the output of a single node to a Markdown file."""
        now = datetime.now(timezone.utc)
        filename = f"{self._step:02d}_{node_name}.md"
        self._step += 1

        emoji, _ = _NODE_STYLE.get(node_name, ("▸", "fg=white"))
        lines = [f"# {emoji} {node_name.capitalize()}\n"]
        lines.append(f"*{now.strftime('%H:%M:%S UTC')}*\n")

        # ── Phase banner ──
        phases = update.get("plan_phases") or []
        current = update.get("current_phase")
        if phases and current is not None:
            total = len(phases)
            title = phases[current]["title"] if current < total else "?"
            lines.append(f"**Phase {current + 1} of {total}:** {title}\n")

        # ── Skills banner (once per planner step) ──
        if node_name == "planner" and self._skills:
            lines.append(f"**Skills in use:** {', '.join(self._skills)}\n")

        # ── Input: what was sent to the LLM ──
        prompt_summary = update.get("_prompt_summary", "")
        if prompt_summary:
            lines.append("<details>\n<summary>Prompt sent to LLM</summary>\n")
            lines.append(f"```\n{prompt_summary}\n```\n")
            lines.append("</details>\n")

        # ── Output: what was produced ──
        lines.append("---\n")

        if node_name == "planner":
            lines.append("## Plan\n")
            lines.append(update.get("plan", "_(empty)_"))

        elif node_name == "coder":
            drafts = update.get("code_drafts", {})
            raw = update.get("coder_raw_response", "")
            if drafts:
                lines.append(f"## Generated Files ({len(drafts)})\n")
                for path, source in drafts.items():
                    n_lines = len(source.splitlines())
                    lines.append(f"\n### {path} ({n_lines} lines)\n")
                    lines.append(f"```python\n{source}```\n")
            elif raw:
                lines.append("## ⚠ No code blocks parsed\n")
                lines.append("Raw LLM response:\n")
                lines.append(f"```\n{raw}\n```\n")
            else:
                lines.append("_(empty — no response from LLM)_")

        elif node_name == "verifier":
            logs = update.get("test_logs", [])
            iteration = update.get("iteration_count", "?")
            if not logs:
                lines.append(f"## ✅ All tests passed (iteration {iteration})\n")
            else:
                lines.append(f"## ❌ Tests failed (iteration {iteration})\n")
                lines.append("```\n" + "\n---\n".join(logs) + "\n```\n")

        elif node_name == "reflector":
            lines.append("## Error Analysis\n")
            lines.append(update.get("reflection", "_(empty)_"))
            findings = update.get("ground_truth", [])
            if findings:
                lines.append("\n\n## Key Findings\n")
                lines.extend(findings)

        elif node_name == "advance_phase":
            ap_phases = update.get("plan_phases") or []
            ap_current = update.get("current_phase", 0)
            if ap_phases:
                prev = ap_current - 1 if ap_current > 0 else 0
                lines.append(f"Completed **Phase {ap_phases[prev]['id']}: "
                             f"{ap_phases[prev]['title']}**\n")
                lines.append(f"Now starting **Phase {ap_phases[ap_current]['id']}: "
                             f"{ap_phases[ap_current]['title']}**\n")

        (self._dir / filename).write_text("\n".join(lines) + "\n")

    def write_summary(self, final_state: ScientificState) -> None:
        """Write a JSON summary of the run."""
        phases = final_state.get("plan_phases") or []
        phase_summary = [
            {"id": p["id"], "title": p["title"], "status": p["status"]}
            for p in phases
        ]
        summary = {
            "task": final_state.get("task_description", ""),
            "model": f"{self._provider}/{self._model}",
            "skills": self._skills,
            "phases": phase_summary,
            "iterations": final_state.get("iteration_count", 0),
            "passed": not bool(final_state.get("test_logs")),
            "files_generated": list(final_state.get("code_drafts", {}).keys()),
            "findings": final_state.get("ground_truth", []),
            "started": self._start.isoformat(),
            "finished": datetime.now(timezone.utc).isoformat(),
        }
        (self._dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        click.echo(f"📂 Chat log written to {self._dir}/")
