"""State definition for the Scientific Loop agent.

The central ``ScientificState`` TypedDict flows through every node in the
LangGraph graph.  ``SqliteSaver`` checkpointing is configured via the helper
exposed here so that long-running sessions can be interrupted and resumed.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver


class ScientificState(TypedDict, total=False):
    """Mutable state that travels through the Plan-Execute-Verify loop.

    Attributes
    ----------
    task_description : str
        Natural-language description of the scientific task to solve.
    mathematical_constants : dict[str, float]
        Named constants relevant to the computation (e.g. ``{"c": 3e8}``).
    plan : str
        LaTeX-formatted mathematical spec and file-tree plan produced by the
        Planner node.
    code_drafts : dict[str, str]
        Mapping of file paths to their generated source code.
    test_logs : list[str]
        Captured ``pytest`` output / stack traces from the Verifier.
    reflection : str
        Error analysis produced by the Reflector node.
    iteration_count : int
        Number of Plan→Code→Verify cycles completed so far.
    ground_truth : list[str]
        Accumulated key findings discovered during the run.
    output_dir : str
        Directory to write generated files to.  Empty string (default) means
        sandbox mode — code stays in memory and tests run in a temp dir.
    skills_context : str
        Concatenated content of matched skills, injected into planner and
        coder prompts.  Empty string when no skills apply.
    """

    task_description: str
    mathematical_constants: dict[str, float]
    plan: str
    code_drafts: dict[str, str]
    test_logs: list[str]
    reflection: str
    iteration_count: int
    ground_truth: list[str]
    output_dir: str
    skills_context: str


def make_checkpointer(db_path: str = "checkpoints.sqlite") -> SqliteSaver:
    """Return a ``SqliteSaver`` backed by the given SQLite file.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database used for checkpoint persistence.
    """
    return SqliteSaver.from_conn_string(db_path)
