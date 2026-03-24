"""Scientific Loop — LangGraph entry point.

Assembles the Plan → Code → Verify → (Reflect?) state graph and exposes a
CLI for launching or resuming long-running scientific-coding sessions.

Usage
-----
    explorer run --task "Implement a factorial function" --thread-id run_001
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any

# Suppress urllib3 warning on macOS system Python compiled with LibreSSL
warnings.filterwarnings("ignore", message=".*LibreSSL.*", category=UserWarning)

import click
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from orchestrator.nodes import advance_phase, coder, configure_llm, get_llm, planner, reflector, verifier
from orchestrator.state import ScientificState, make_checkpointer

logger = logging.getLogger(__name__)

# Load .env file (if present) so env vars are available as CLI defaults
load_dotenv()

MAX_ITERATIONS = int(os.environ.get("EXPLORER_MAX_ITERATIONS", "20"))  # 0 = unlimited
REPLAN_THRESHOLD = 3  # N total phase failures → re-engage planner
MAX_PHASE_ITERATIONS = 8  # max iterations per single phase before force-advancing


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _make_should_continue(max_iters: int):
    """Return a conditional edge function with a configurable iteration cap.

    Four outcomes:
    - ``"end"``           — tests passed and all phases are done
    - ``"advance_phase"`` — tests passed but more phases remain
    - ``"advance_phase"`` — per-phase iteration cap exceeded (force-advance)
    - ``"reflect"``       — tests failed and we haven't hit the cap
    """
    def _should_continue(state: ScientificState) -> str:
        if not state.get("test_logs"):
            # Tests passed — check if there are more phases
            phases = state.get("plan_phases") or []
            current = state.get("current_phase", 0)
            if phases and current + 1 < len(phases):
                return "advance_phase"
            return "end"
        if max_iters > 0 and state.get("iteration_count", 0) >= max_iters:
            return "end"
        # Per-phase iteration cap: force-advance if stuck too long on one phase.
        # BUT: never force-advance on collection errors (no tests ran at all),
        # since that propagates completely broken code to the next phase.
        phase_iters = state.get("_phase_iteration_count", 0)
        phases = state.get("plan_phases") or []
        current = state.get("current_phase", 0)
        is_collection_error = state.get("_collection_error", False)
        if (
            phase_iters >= MAX_PHASE_ITERATIONS
            and phases
            and current + 1 < len(phases)
            and not is_collection_error
        ):
            logger.warning(
                "Phase %d exceeded %d iterations — force-advancing to next phase",
                current, MAX_PHASE_ITERATIONS,
            )
            return "advance_phase"
        return "reflect"
    return _should_continue


MAX_REPLANS = 2  # max replans per phase before giving up on replanning


def _after_reflector(state: ScientificState) -> str:
    """Route after the reflector: re-plan on persistent failures.

    If the total number of failures in the current phase reaches
    the replan threshold, the coder is clearly stuck — send the problem back
    to the planner so it can decompose the current phase differently or
    suggest a new algorithm.

    The replan threshold uses exponential backoff: 3 errors for the first
    replan, 6 for the second, etc.  This gives the coder more attempts
    to self-recover after each replan before triggering another one.

    Replanning is capped at ``MAX_REPLANS`` per phase to avoid infinite loops.
    """
    phase_errors = state.get("_phase_error_count", 0)
    replan_count = state.get("_replan_count", 0)
    # Exponential backoff: threshold doubles after each replan
    current_threshold = REPLAN_THRESHOLD * (2 ** replan_count)
    if phase_errors >= current_threshold and replan_count < MAX_REPLANS:
        return "replan"
    return "coder"


def build_graph(max_iterations: int = MAX_ITERATIONS) -> StateGraph:
    """Construct and compile the Scientific Loop state graph."""
    graph = StateGraph(ScientificState)

    graph.add_node("planner", planner)
    graph.add_node("coder", coder)
    graph.add_node("verifier", verifier)
    graph.add_node("reflector", reflector)
    graph.add_node("advance_phase", advance_phase)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "verifier")
    graph.add_conditional_edges(
        "verifier",
        _make_should_continue(max_iterations),
        {"reflect": "reflector", "advance_phase": "advance_phase", "end": END},
    )
    graph.add_conditional_edges(
        "reflector",
        _after_reflector,
        {"coder": "coder", "replan": "planner"},
    )
    graph.add_edge("advance_phase", "coder")

    return graph


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_ground_truth(state: ScientificState) -> None:
    """Persist accumulated findings to ``ground_truth.md``."""
    findings: list[str] = state.get("ground_truth", [])
    task = state.get("task_description", "Unknown task")
    iters = state.get("iteration_count", 0)

    lines = [
        "# Ground Truth",
        "",
        f"> Task: {task}",
        f"> Iterations: {iters}",
        "",
        "## Key Findings",
        "",
    ]
    if findings:
        lines.extend(findings)
    else:
        lines.append("_No findings recorded (all tests passed on the first attempt)._")
    lines.append("")

    Path("ground_truth.md").write_text("\n".join(lines))
    print(f"📝 ground_truth.md written ({len(findings)} finding(s))")


@click.group()
@click.version_option(version="0.1.0", prog_name="explorer")
def cli() -> None:
    """Explorer — autonomous scientific Python development agent."""


@cli.command()
@click.option(
    "--task", "-t",
    default=None,
    help="Natural-language description of the scientific task.",
)
@click.option(
    "--task-file", "-f",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a Markdown file containing the task description.",
)
@click.option(
    "--thread-id",
    default=None,
    help="Thread ID for checkpoint persistence.  "
         "Omit to start a fresh run with an auto-generated ID.",
)
@click.option(
    "--db",
    default="checkpoints.sqlite",
    show_default=True,
    type=click.Path(),
    help="SQLite file for checkpointing.",
)
@click.option(
    "--max-iterations",
    default=lambda: int(os.environ.get("EXPLORER_MAX_ITERATIONS", "20")),
    show_default="20",
    type=int,
    help="Maximum plan-code-verify cycles before stopping (0 = unlimited, env: EXPLORER_MAX_ITERATIONS).",
)
@click.option(
    "--provider",
    default=lambda: os.environ.get("EXPLORER_LLM_PROVIDER", "ollama"),
    show_default="ollama",
    type=click.Choice(["ollama", "openai"], case_sensitive=False),
    help="LLM provider to use (env: EXPLORER_LLM_PROVIDER).",
)
@click.option(
    "--model",
    default=lambda: os.environ.get("EXPLORER_LLM_MODEL", "qwen2.5-coder:32b"),
    show_default="qwen2.5-coder:32b",
    help="Model name for the chosen provider (env: EXPLORER_LLM_MODEL).",
)
@click.option(
    "--base-url",
    default=lambda: os.environ.get("EXPLORER_LLM_BASE_URL"),
    help="Base URL for the LLM API (env: EXPLORER_LLM_BASE_URL).",
)
@click.option(
    "--temperature",
    default=lambda: float(os.environ.get("EXPLORER_LLM_TEMPERATURE", "0.0")),
    show_default="0.0",
    type=float,
    help="Sampling temperature (env: EXPLORER_LLM_TEMPERATURE).",
)
@click.option(
    "--output-dir", "-o",
    default=None,
    type=click.Path(file_okay=False),
    help="Write generated files to this directory (write mode). "
         "Default: sandbox mode (temp directory, auto-cleaned).",
)
@click.option(
    "--skills", "-s",
    multiple=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directories containing skill folders (repeatable).",
)
@click.option(
    "--chat-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory to save the full chat log of the run. "
         "Default: no chat log.",
)
@click.option(
    "--resume", "-r",
    is_flag=True,
    default=False,
    help="Resume a previous run from its checkpoint. "
         "Requires --thread-id.",
)
def run(task: str | None, task_file: str | None, thread_id: str, db: str,
        max_iterations: int, provider: str, model: str,
        base_url: str | None, temperature: float,
        output_dir: str | None, skills: tuple[str, ...],
        chat_dir: str | None, resume: bool) -> None:
    """Run the Scientific Loop agent on a task.

    Provide either --task/-t with inline text, or --task-file/-f pointing to a
    Markdown file.  At least one is required.
    """
    if resume and not thread_id:
        raise click.UsageError("--resume requires --thread-id.")
    if not resume:
        if task and task_file:
            raise click.UsageError("Use --task or --task-file, not both.")
        if task_file:
            task = Path(task_file).read_text()
        if not task:
            raise click.UsageError("Provide either --task or --task-file.")
    else:
        # When resuming, task is loaded from the checkpoint; allow
        # an optional override via --task / --task-file.
        if task_file:
            task = Path(task_file).read_text()

    # Set LangSmith project name if tracing is enabled but no project set
    if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true":
        if not os.environ.get("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = "explorer"
        click.echo(f"🔍 LangSmith tracing → project: {os.environ['LANGCHAIN_PROJECT']}")

    configure_llm(
        provider=provider, model=model,
        base_url=base_url, temperature=temperature,
    )
    # Generate a unique thread ID when none is supplied, so each run
    # starts with a clean slate and stale checkpoints never interfere.
    if thread_id is None:
        thread_id = uuid.uuid4().hex[:12]

    checkpointer = make_checkpointer(db)

    if not resume:
        # Fresh run — clear any leftover checkpoint so the graph
        # always starts from initial_state.
        checkpointer.delete_thread(thread_id)

    graph = build_graph(max_iterations=max_iterations)
    app = graph.compile(checkpointer=checkpointer)

    # Ensure the output directory exists when running in write mode
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load and match skills (progressive disclosure)
    # Always include the built-in skills/ directory shipped with the project
    from orchestrator.skills import load_skills, match_skills, format_skills_context

    builtin_skills_dir = Path(__file__).resolve().parent.parent / "skills"
    skill_dirs = list(skills)
    if builtin_skills_dir.is_dir() and str(builtin_skills_dir) not in skill_dirs:
        skill_dirs.insert(0, str(builtin_skills_dir))

    skills_context = ""
    all_skills = load_skills(skill_dirs)
    matched = match_skills(all_skills, task)
    if matched:
        skills_context = format_skills_context(matched)
        click.echo(f"📚 Matched {len(matched)} skill(s): "
                   f"{', '.join(s.name for s in matched)}")
    elif all_skills:
        click.echo(f"📚 Loaded {len(all_skills)} skill(s), none matched the task.")

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 2**31}

    # When resuming, restore the full accumulated state from the
    # checkpoint and restart the graph cleanly from the beginning.
    # This handles both "completed-with-failures" (hit max-iterations)
    # and "interrupted" (GraphRecursionError / Ctrl-C) cases.
    if resume:
        saved = app.get_state(config)
        if not saved or not saved.values:
            raise click.UsageError(
                f"No checkpoint found for thread '{thread_id}'. "
                "Nothing to resume."
            )

        # Capture the full accumulated state before touching the checkpoint
        input_state: dict[str, Any] = dict(saved.values)

        # Override with CLI-supplied values
        input_state["output_dir"] = output_dir or input_state.get("output_dir", "")
        input_state["skills_context"] = skills_context or input_state.get("skills_context", "")
        if task:
            input_state["task_description"] = task

        iters = input_state.get("iteration_count", 0)
        drafts = len(input_state.get("code_drafts", {}))
        phase_info = ""
        phases = input_state.get("plan_phases") or []
        if phases:
            cur = input_state.get("current_phase", 0)
            phase_info = f", phase {cur + 1}/{len(phases)}"
        click.echo(
            f"♻ Restoring state: {iters} iteration(s), "
            f"{drafts} file(s){phase_info}"
        )

        # Delete the old checkpoint so the graph starts from the entry
        # point (planner) with the full accumulated context.
        checkpointer.delete_thread(thread_id)
    else:
        input_state = {
            "task_description": task,
            "mathematical_constants": {},
            "plan": "",
            "code_drafts": {},
            "test_logs": [],
            "reflection": "",
            "iteration_count": 0,
            "ground_truth": [],
            "output_dir": output_dir or "",
            "skills_context": skills_context,
            "plan_phases": [],
            "current_phase": 0,
        }

    # Set up reporting
    from orchestrator.reporter import ChatLogger, report_node

    matched_skill_names: list[str] = [s.name for s in matched]
    chat_logger = ChatLogger(
        chat_dir,
        task=task or "",
        skills=matched_skill_names,
        provider=provider,
        model=model,
        max_iterations=max_iterations,
    ) if chat_dir else None

    verb = "Resuming" if resume else "Starting"
    click.echo(f"▶ {verb} Scientific Loop (thread={thread_id})")

    # Stream node-by-node for live reporting
    final_state: dict[str, Any] = dict(input_state)
    for event in app.stream(input_state, config=config):
        for node_name, update in event.items():
            report_node(node_name, update)
            if chat_logger:
                chat_logger.log_node(node_name, update)
            final_state.update(update)

    iters = final_state.get("iteration_count", 0)
    logs = final_state.get("test_logs", [])

    _write_ground_truth(final_state)

    if chat_logger:
        chat_logger.write_summary(final_state)

    if not logs:
        click.echo(f"✔ All tests passed after {iters} iteration(s).")
    else:
        click.echo(f"✘ Stopped after {iters} iteration(s) with errors.")
        click.echo("\n".join(logs))

    sys.exit(0 if not logs else 1)


@cli.command("status")
@click.argument("thread_id")
@click.option(
    "--db",
    default="checkpoints.sqlite",
    show_default=True,
    type=click.Path(),
    help="SQLite file for checkpointing.",
)
def status(thread_id: str, db: str) -> None:
    """Show the status of a workflow run.

    Reads the checkpoint for THREAD_ID and prints a summary including
    the task, iteration count, test status, phase progress, and files.
    """
    checkpointer = make_checkpointer(db)
    graph = build_graph()
    app = graph.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    saved = app.get_state(config)
    if not saved or not saved.values:
        click.echo(f"No checkpoint found for thread '{thread_id}'.")
        sys.exit(1)

    vals = saved.values
    task = vals.get("task_description", "")
    iters = vals.get("iteration_count", 0)
    test_logs = vals.get("test_logs", [])
    phases = vals.get("plan_phases") or []
    current_phase = vals.get("current_phase", 0)
    code_drafts = vals.get("code_drafts", {})
    output_dir = vals.get("output_dir", "")

    # Determine overall status
    if not test_logs:
        if phases and current_phase + 1 < len(phases):
            overall = "✔ Tests passing — more phases remain"
        else:
            overall = "✔ Completed successfully"
    else:
        overall = "✘ Stopped with failing tests"

    # Print summary
    click.echo(f"Thread   : {thread_id}")
    click.echo(f"Status   : {overall}")
    click.echo(f"Task     : {task[:120]}{'…' if len(task) > 120 else ''}")
    click.echo(f"Iterations: {iters}")
    click.echo(f"Files    : {len(code_drafts)}")
    if output_dir:
        click.echo(f"Output   : {output_dir}")

    # Phase progress
    if phases:
        click.echo(f"\nPhases ({current_phase + 1}/{len(phases)}):")
        for p in phases:
            marker = {"completed": "✔", "in-progress": "▶", "pending": "○"}.get(
                p.get("status", "pending"), "?"
            )
            files_str = f"  [{', '.join(p.get('files', []))}]" if p.get("files") else ""
            click.echo(f"  {marker} Phase {p['id']}: {p['title']}{files_str}")

    # File list
    if code_drafts:
        click.echo(f"\nGenerated files:")
        for path in sorted(code_drafts):
            lines = code_drafts[path].count("\n")
            click.echo(f"  {path} ({lines} lines)")

    # Last error excerpt
    if test_logs:
        click.echo(f"\nLast error (truncated):")
        excerpt = "\n".join(test_logs)
        if len(excerpt) > 500:
            excerpt = excerpt[:500] + "\n  …"
        for line in excerpt.split("\n"):
            click.echo(f"  │ {line}")


@cli.command("check-llm")
@click.option(
    "--provider",
    default=lambda: os.environ.get("EXPLORER_LLM_PROVIDER", "ollama"),
    show_default="ollama",
    type=click.Choice(["ollama", "openai"], case_sensitive=False),
    help="LLM provider to use (env: EXPLORER_LLM_PROVIDER).",
)
@click.option(
    "--model",
    default=lambda: os.environ.get("EXPLORER_LLM_MODEL", "qwen2.5-coder:32b"),
    show_default="qwen2.5-coder:32b",
    help="Model name for the chosen provider (env: EXPLORER_LLM_MODEL).",
)
@click.option(
    "--base-url",
    default=lambda: os.environ.get("EXPLORER_LLM_BASE_URL"),
    help="Base URL for the LLM API (env: EXPLORER_LLM_BASE_URL).",
)
def check_llm(provider: str, model: str, base_url: str | None) -> None:
    """Check connectivity to the configured LLM endpoint."""
    from langchain_core.messages import HumanMessage

    configure_llm(provider=provider, model=model, base_url=base_url)

    click.echo(f"Provider : {provider}")
    click.echo(f"Model    : {model}")
    if base_url:
        click.echo(f"Base URL : {base_url}")

    click.echo("\n⏳ Sending test prompt…")
    try:
        llm = get_llm()
        response = llm.invoke([HumanMessage(content="Reply with exactly: OK")])
        click.echo(f"✔ Response: {response.content.strip()}")
    except Exception as exc:
        click.echo(f"✘ Connection failed: {exc}")
        sys.exit(1)


def main() -> None:
    """Entry point for ``explorer`` console script."""
    cli()


if __name__ == "__main__":
    main()
