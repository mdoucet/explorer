"""State definition for the Scientific Loop agent.

The central ``ScientificState`` TypedDict flows through every node in the
LangGraph graph.  ``SqliteSaver`` checkpointing is configured via the helper
exposed here so that long-running sessions can be interrupted and resumed.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator, TypedDict

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
    coder_raw_response : str
        Raw text returned by the LLM in the coder node, before parsing.
        Stored for diagnostic purposes when ``code_drafts`` is empty.
    plan_phases : list[dict]
        Ordered list of plan phases produced by the planner.  Each dict has
        keys ``id`` (int), ``title`` (str), ``description`` (str),
        ``status`` (str), and ``files`` (list[str]).
    current_phase : int
        Zero-based index of the phase the coder is currently working on.
    verified_fixes : list[str]
        Cumulative list of constraints the coder must honour (e.g. "use raw
        docstrings").  Populated by the verifier when a previously-failing
        check passes after the reflector suggested a fix.
    clean_files : list[str]
        File paths that had no syntax or import errors on the last verifier
        run.  The coder is told not to modify these unless necessary.
    best_code_drafts : dict[str, str]
        Snapshot of ``code_drafts`` from the iteration with the fewest
        errors in the current phase.  Used for rollback when the coder
        regresses.
    best_error_count : int
        Number of errors in ``best_code_drafts``.  -1 means no snapshot yet.
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
    coder_raw_response: str
    plan_phases: list[dict]
    current_phase: int
    _prev_error_fingerprint: str
    _error_repeat_count: int
    _replan_count: int
    _phase_error_count: int
    _phase_iteration_count: int
    verified_fixes: list[str]
    clean_files: list[str]
    best_code_drafts: dict[str, str]
    best_error_count: int
    _prev_syntax_file_count: int
    _collection_error: bool


def make_checkpointer(db_path: str = "checkpoints.sqlite") -> SqliteSaver:
    """Return a ``SqliteSaver`` backed by the given SQLite file.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database used for checkpoint persistence.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)
