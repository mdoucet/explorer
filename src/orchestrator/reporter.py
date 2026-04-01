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
    "auto_reflect":  ("🔄", "fg=magenta"),
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
        inner_loops = update.get("_inner_loop_count", 0)
        if drafts:
            click.echo(click.style(f"  Generated {len(drafts)} file(s):", dim=True))
            for path, source in drafts.items():
                n_lines = len(source.splitlines())
                click.echo(f"  │ {path}  ({n_lines} lines)")
            if inner_loops:
                click.echo(click.style(
                    f"  ⟳ {inner_loops} inner-loop self-correction(s)",
                    fg="yellow",
                ))
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

    elif node_name == "auto_reflect":
        action = update.get("_reflector_action", "retry")
        reflection = update.get("reflection", "")
        if action == "replan":
            click.echo(click.style("  ⚠ Recommending replan (repeated errors)", fg="yellow"))
        else:
            click.echo(click.style("  Retry — passing test logs to coder", dim=True))
        if reflection:
            preview = "\n".join(reflection.splitlines()[:4])
            for line in preview.splitlines():
                click.echo(f"  │ {line}")

    elif node_name == "advance_phase":
        new_phase = update.get("current_phase", "?")
        ap_phases = update.get("plan_phases") or []
        if ap_phases and isinstance(new_phase, int) and new_phase < len(ap_phases):
            title = ap_phases[new_phase]["title"]
            click.echo(click.style(
                f"  Advancing to phase {new_phase + 1}: {title}", fg="blue", bold=True,
            ))

    # Show LLM call timing summary for any node
    node_calls = update.get("_llm_calls") or []
    if node_calls:
        total_s = sum(c.get("duration_s", 0) for c in node_calls)
        click.echo(click.style(
            f"  ⏱ {len(node_calls)} LLM call(s), {total_s:.1f}s total",
            dim=True,
        ))


# ---------------------------------------------------------------------------
# Chat-log persistence
# ---------------------------------------------------------------------------

class ChatLogger:
    """Writes node outputs to a timestamped directory for post-run review.

    The chat output is structured as a **workflow trace** with separate
    files for each LLM API call::

        chat_dir/
        ├── 00_run_info.md
        ├── 01_planner.md                  ← workflow trace
        ├── 01_planner_llm_call_01.md      ← full LLM I/O
        ├── 02_coder.md
        ├── 02_coder_llm_call_01.md
        ├── 02_coder_llm_call_02.md
        ├── 03_verifier.md
        ├── 03_verifier_llm_call_01.md     ← triage call
        ├── 04_auto_reflect.md
        ├── summary.json
        └── llm_calls.json
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
        for old in self._dir.glob("llm_calls.json"):
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

    def _write_llm_call_file(
        self, step: int, node_name: str, call_idx: int, call: dict[str, Any],
    ) -> str:
        """Write a single LLM call to its own Markdown file.

        Returns the filename (for linking from the trace file).
        """
        filename = f"{step:02d}_{node_name}_llm_call_{call_idx:02d}.md"
        label = call.get("label") or f"call-{call_idx}"
        duration = call.get("duration_s", 0)
        timestamp = call.get("timestamp", "")

        usage = call.get("token_usage") or {}
        usage_str = ""
        if usage:
            parts = []
            if "prompt_tokens" in usage:
                parts.append(f"in={usage['prompt_tokens']}")
            if "completion_tokens" in usage:
                parts.append(f"out={usage['completion_tokens']}")
            if "total_tokens" in usage:
                parts.append(f"total={usage['total_tokens']}")
            usage_str = f" | tokens: {', '.join(parts)}"

        lines = [
            f"# LLM Call: {node_name} — {label}\n",
            f"**Duration:** {duration:.1f}s{usage_str}  \n",
        ]
        if timestamp:
            lines.append(f"**Timestamp:** {timestamp}\n")

        # System prompt
        sys_prompt = call.get("system_prompt", "")
        if sys_prompt:
            lines.append(f"\n## System Prompt ({len(sys_prompt)} chars)\n")
            lines.append(f"```\n{sys_prompt}\n```\n")

        # User prompt
        user_prompt = call.get("user_prompt", "")
        if user_prompt:
            lines.append(f"\n## User Prompt ({len(user_prompt)} chars)\n")
            lines.append(f"```\n{user_prompt}\n```\n")

        # Response
        resp = call.get("response_text", "")
        thinking = call.get("thinking", "")
        if thinking:
            lines.append(f"\n## Thinking ({len(thinking)} chars)\n")
            lines.append(f"```\n{thinking}\n```\n")
        if resp:
            lines.append(f"\n## LLM Response ({len(resp)} chars)\n")
            lines.append(f"```\n{resp}\n```\n")

        # Tool calls
        tc_out = call.get("tool_calls") or []
        if tc_out:
            lines.append(f"\n## Tool Calls Issued ({len(tc_out)})\n")
            for tc in tc_out:
                lines.append(f"- **{tc['name']}**({tc['args']})\n")

        # Tool results
        tool_msgs = call.get("tool_messages") or []
        if tool_msgs:
            lines.append(f"\n## Tool Results ({len(tool_msgs)})\n")
            for tm in tool_msgs:
                result_preview = tm.get("result", "")[:2000]
                lines.append(
                    f"### {tm['tool']}({tm.get('args', {})})\n"
                    f"```\n{result_preview}\n```\n"
                )

        (self._dir / filename).write_text("\n".join(lines) + "\n")
        return filename

    def log_node(self, node_name: str, update: dict[str, Any]) -> None:
        """Persist the output of a single node to a workflow trace file.

        LLM call details are written to separate companion files.
        """
        now = datetime.now(timezone.utc)
        step = self._step
        filename = f"{step:02d}_{node_name}.md"
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

        # ── Write LLM call companion files and link from trace ──
        # With the add-reducer on _llm_calls, the update contains only
        # this node's new calls — no need to filter by node name.
        node_calls = update.get("_llm_calls") or []
        if node_calls:
            total_s = sum(c.get("duration_s", 0) for c in node_calls)
            lines.append(
                f"**LLM calls:** {len(node_calls)} "
                f"({total_s:.1f}s total)\n"
            )
            for idx, call in enumerate(node_calls, 1):
                call_file = self._write_llm_call_file(
                    step, node_name, idx, call,
                )
                label = call.get("label") or f"call-{idx}"
                duration = call.get("duration_s", 0)
                lines.append(f"- [{label} ({duration:.1f}s)]({call_file})\n")

        # ── Node-specific output ──
        lines.append("\n---\n")

        if node_name == "planner":
            lines.append("## Plan\n")
            lines.append(update.get("plan", "_(empty)_"))

        elif node_name == "coder":
            drafts = update.get("code_drafts", {})
            raw = update.get("coder_raw_response", "")
            inner_loops = update.get("_inner_loop_count", 0)
            if drafts:
                lines.append(f"## Generated Files ({len(drafts)})\n")
                if inner_loops:
                    lines.append(
                        f"*{inner_loops} inner-loop self-correction(s)*\n"
                    )
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

        elif node_name == "auto_reflect":
            action = update.get("_reflector_action", "retry")
            reflection = update.get("reflection", "")
            lines.append(f"**Action:** {action}\n")
            if reflection:
                lines.append("## Test Log Summary\n")
                lines.append(f"```\n{reflection[:3000]}\n```\n")

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

        # Aggregate LLM call statistics
        all_calls = final_state.get("_llm_calls") or []
        total_duration = sum(c.get("duration_s", 0) for c in all_calls)
        calls_by_node: dict[str, int] = {}
        for c in all_calls:
            n = c.get("node", "unknown")
            calls_by_node[n] = calls_by_node.get(n, 0) + 1

        summary = {
            "task": final_state.get("task_description", ""),
            "model": f"{self._provider}/{self._model}",
            "skills": self._skills,
            "phases": phase_summary,
            "iterations": final_state.get("iteration_count", 0),
            "passed": not bool(final_state.get("test_logs")),
            "files_generated": list(final_state.get("code_drafts", {}).keys()),
            "findings": final_state.get("ground_truth", []),
            "llm_calls_total": len(all_calls),
            "llm_calls_by_node": calls_by_node,
            "llm_total_duration_s": round(total_duration, 2),
            "started": self._start.isoformat(),
            "finished": datetime.now(timezone.utc).isoformat(),
        }
        (self._dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

        # Also write the full LLM call log as a separate file
        if all_calls:
            (self._dir / "llm_calls.json").write_text(
                json.dumps(all_calls, indent=2, default=str) + "\n"
            )

        click.echo(f"📂 Chat log written to {self._dir}/")
