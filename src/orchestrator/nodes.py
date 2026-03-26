"""Core graph nodes for the Scientific Loop.

Each function accepts the current ``ScientificState`` and returns a partial
state dict that LangGraph merges back into the graph state.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from .state import ScientificState
from .transcript import format_history, make_entry

logger = logging.getLogger(__name__)

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


def _invoke_llm(llm: Any, messages: list, *, max_retries: int = 3, retry_delay: float = 5.0) -> Any:
    """Invoke the LLM with streaming and retry on transient errors.

    Uses ``llm.stream()`` to avoid first-byte timeouts on large prompts
    (the server proxy drops connections that don't produce a token within
    ~60s).  Falls back to ``llm.invoke()`` for test mocks that don't
    support streaming.
    """
    if not hasattr(llm, "stream"):
        return llm.invoke(messages)

    for attempt in range(1, max_retries + 1):
        try:
            chunks: list[Any] = []
            for chunk in llm.stream(messages):
                chunks.append(chunk)
            if not chunks:
                return llm.invoke(messages)
            result = chunks[0]
            for c in chunks[1:]:
                result = result + c
            return result
        except Exception as exc:
            err_str = str(exc).lower()
            transient = any(k in err_str for k in (
                "incomplete chunked read", "peer closed", "connection",
                "timeout", "502", "503", "504", "remotedisconnected",
                "remoteprotocolerror",
            ))
            if transient and attempt < max_retries:
                logger.warning(
                    "Transient LLM error (attempt %d/%d): %s — retrying in %.0fs",
                    attempt, max_retries, exc, retry_delay,
                )
                import time
                time.sleep(retry_delay)
                continue
            raise


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
    llm = get_llm()

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
        "_phase_error_count": 0,
        "_phase_iteration_count": 0,
        "_prev_syntax_file_count": 0,
        "_collection_error": False,
        "verified_fixes": [],
        "clean_files": [],
        "best_code_drafts": {},
        "best_error_count": -1,
        "transcript": transcript,
    }


# ---------------------------------------------------------------------------
# 2. Coder
# ---------------------------------------------------------------------------


def _extract_signatures(source: str) -> list[str]:
    """Extract function and class signatures from Python *source*.

    Returns a list of concise signature strings, e.g.::

        ["def solve_square_well(n: int, L: float) -> np.ndarray",
         "class Solver"]

    Non-Python files or unparseable source silently return an empty list.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    sigs: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sigs.append(f"def {node.name}({ast.unparse(node.args)})")
        elif isinstance(node, ast.ClassDef):
            methods = [
                f"  def {n.name}({ast.unparse(n.args)})"
                for n in ast.iter_child_nodes(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            sigs.append(f"class {node.name}")
            sigs.extend(methods)
    return sigs


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


# Cache environment info — doesn't change during a run
_env_info_cache: str | None = None


def _get_environment_info() -> str:
    """Return a short summary of the Python environment (version + key packages).

    Cached after the first call since the environment doesn't change mid-run.
    """
    global _env_info_cache  # noqa: PLW0603
    if _env_info_cache is not None:
        return _env_info_cache

    lines = [f"Python {sys.version.split()[0]}"]
    try:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pip", "list", "--format=freeze"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            # Pick key scientific packages
            interesting = {"numpy", "scipy", "matplotlib", "click", "pytest"}
            for line in result.stdout.splitlines():
                pkg = line.split("==")[0].lower()
                if pkg in interesting:
                    lines.append(line.strip())
    except Exception:
        pass

    _env_info_cache = "\n".join(lines)
    return _env_info_cache


def _append_coder_transcript(
    state: ScientificState,
    new_drafts: dict[str, str],
    deletions: set[str],
) -> list[dict]:
    """Build updated transcript with coder summary entry."""
    transcript = list(state.get("transcript") or [])
    file_list = ", ".join(sorted(new_drafts.keys()))
    parts = [f"Generated {len(new_drafts)} file(s): {file_list}"]
    if deletions:
        parts.append(f"Deleted: {', '.join(sorted(deletions))}")
    transcript.append(make_entry(
        "ai", "\n".join(parts),
        node="coder",
        step=state.get("iteration_count", 0),
        phase=state.get("current_phase", 0),
    ))
    return transcript


def coder(state: ScientificState) -> dict[str, Any]:
    """Generate Python source code from the current plan phase.

    Returns
    -------
    dict
        ``code_drafts``: ``{filepath: source_code}`` mapping extracted from
        the LLM response.  Accumulated over phases — prior-phase files are
        preserved.
    """
    llm = get_llm()
    prompt = _load_prompt("EXPLORER_PROMPT_CODER", "coder.md")

    # Phase context for the coder
    phases = state.get("plan_phases") or []
    current = state.get("current_phase", 0)
    total = len(phases) if phases else 1
    phase_title = phases[current]["title"] if phases else "Implementation"

    user_parts: list[str] = [
        f"## Current Phase ({current + 1} of {total}): {phase_title}\n{state['plan']}",
    ]

    # Inject Python environment info so the coder knows available library versions
    env_info = _get_environment_info()
    if env_info:
        user_parts.append(f"## Environment\n{env_info}")

    # Inject run history so the coder sees what happened in prior iterations
    # (replaces ad-hoc ground_truth, reflection, test_logs injection).
    history = format_history(state.get("transcript") or [])
    if history:
        user_parts.append(history)

    # Inject verified fixes as non-negotiable constraints
    verified_fixes = state.get("verified_fixes") or []
    if verified_fixes:
        rules = "\n".join(f"- {f}" for f in verified_fixes)
        user_parts.append(
            f"## ⚠️ MANDATORY CONSTRAINTS (verified from prior iterations)\n"
            f"These rules have been validated.  You MUST follow them:\n{rules}"
        )

    # Determine if this is a revision iteration (has prior reflection).
    is_revision = bool(state.get("reflection"))

    # Inform coder about files that already exist from prior phases.
    # On revision iterations, show FULL source so the coder can see
    # exactly what needs changing.  On first pass, show signatures only.
    existing_drafts = state.get("code_drafts") or {}
    if existing_drafts:
        if is_revision:
            # Full source context for revisions — the coder needs to see
            # its own code to fix specific lines.
            clean = set(state.get("clean_files") or [])
            source_parts: list[str] = []
            for fpath in sorted(existing_drafts.keys()):
                if fpath in clean:
                    source_parts.append(
                        f"### {fpath} ✅ (NO ERRORS — do NOT modify)\n"
                        f"```python\n{existing_drafts[fpath]}\n```"
                    )
                else:
                    source_parts.append(
                        f"### {fpath}\n"
                        f"```python\n{existing_drafts[fpath]}\n```"
                    )
            user_parts.append(
                "## Your current source files\n"
                "Files marked ✅ have no errors — do NOT regenerate them.\n"
                "Fix ONLY the files with errors.\n\n"
                + "\n\n".join(source_parts)
            )
        else:
            sig_lines: list[str] = []
            for fpath in sorted(existing_drafts.keys()):
                sigs = _extract_signatures(existing_drafts[fpath])
                if sigs:
                    sig_lines.append(f"### {fpath}\n" + "\n".join(sigs))
                else:
                    sig_lines.append(f"### {fpath}")
            user_parts.append(
                "## Existing files from prior phases\n"
                "These files already exist and can be imported.  Use the "
                "exact names shown below when importing.\n\n"
                + "\n\n".join(sig_lines)
            )

    if state.get("skills_context"):
        user_parts.append(state["skills_context"])

    user_message = "\n\n".join(user_parts)
    response = _invoke_llm(llm, [
        SystemMessage(content=prompt),
        HumanMessage(content=user_message),
    ])

    raw_content = response.content
    new_drafts = _parse_code_blocks(raw_content)

    # Handle explicit deletion markers: if coder emits "# DELETE" as the
    # sole content of a code block, it signals that file should be removed.
    _DELETE_MARKER = "# DELETE"
    deletions: set[str] = set()
    for fp, content in list(new_drafts.items()):
        if content.strip() == _DELETE_MARKER:
            deletions.add(fp)
            del new_drafts[fp]

    # On revision iterations, protect files that are verified clean ONLY
    # after the coder has already had one chance to fix (_phase_error_count >= 2).
    # Both test files and source files use the same criterion: only protect
    # files present in clean_files (all tests passing, no errors).
    # This ensures test files with bugs (e.g. wrong subprocess invocation)
    # can still be fixed while truly passing test files remain untouched.
    phase_errors = state.get("_phase_error_count", 0)
    if is_revision and existing_drafts and phase_errors >= 2:
        clean = set(state.get("clean_files") or [])
        protected = {fp for fp in clean if fp in existing_drafts}
        for fp in protected:
            new_drafts.pop(fp, None)

    # Merge new drafts into accumulated drafts (new files override old)
    merged_drafts = dict(existing_drafts)
    merged_drafts.update(new_drafts)

    # Apply explicit deletions
    for fp in deletions:
        merged_drafts.pop(fp, None)

    # Auto-resolve duplicate layouts (flat vs src/).
    # If both exist, keep whichever layout the new_drafts prefer.
    _resolve_duplicate_layouts(merged_drafts, new_drafts)

    # In write mode, persist files and clean up deleted/pruned files
    output_dir = state.get("output_dir", "")
    if output_dir:
        _write_code_drafts(new_drafts, output_dir)
        # Remove any files that were deleted or pruned from disk
        root = Path(output_dir).resolve()
        all_removed = deletions | (
            (set(existing_drafts) | set(new_drafts)) - set(merged_drafts)
        )
        for fp in all_removed:
            target = (root / fp).resolve()
            if target.is_relative_to(root) and target.exists():
                logger.warning("Removing file from disk: %s", fp)
                target.unlink()

    return {
        "code_drafts": merged_drafts,
        "coder_raw_response": raw_content,
        "_prompt_summary": user_message,
        "transcript": _append_coder_transcript(state, new_drafts, deletions),
    }


_FILE_EXTENSIONS = frozenset({
    ".py", ".toml", ".cfg", ".ini", ".txt", ".md", ".rst",
    ".yaml", ".yml", ".json", ".csv", ".sh", ".bat",
    ".html", ".css", ".js", ".ts", ".sql", ".r", ".jl",
})


def _looks_like_filepath(info_string: str) -> bool:
    """Return True if *info_string* appears to be a file path rather than a
    language tag (e.g. ``python``, ``bash``).

    Heuristic: contains ``/`` **or** ends with a recognised file extension.
    """
    if "/" in info_string:
        return True
    # Check for known file extension (e.g. "pyproject.toml", "conftest.py")
    _, dot, ext = info_string.rpartition(".")
    return dot == "." and f".{ext}" in _FILE_EXTENSIONS


def _parse_code_blocks(text: str) -> dict[str, str]:
    """Extract fenced code blocks whose info-strings look like file paths.

    Falls back to :func:`_parse_unfenced_blocks` when no fenced blocks are
    found -- some LLMs emit file paths as bare header lines without fence markers.
    """
    drafts: dict[str, str] = {}
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            info = line.strip().removeprefix("```").strip()
            if info and _looks_like_filepath(info):
                code_lines: list[str] = []
                i += 1
                while i < len(lines) and not lines[i].startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                drafts[info] = "".join(code_lines)
        i += 1

    if not drafts:
        drafts = _parse_unfenced_blocks(text)

    return drafts


def _parse_unfenced_blocks(text: str) -> dict[str, str]:
    """Fallback parser for LLM output that omits fenced code blocks.

    Detects standalone lines that look like file paths and collects everything
    between consecutive path lines as that file's content.  Only activates
    when at least two filepath-like headers are found (a single match is too
    ambiguous to be trustworthy).
    """
    lines = text.splitlines(keepends=True)
    headers: list[tuple[int, str]] = []
    for idx, raw in enumerate(lines):
        candidate = raw.strip()
        if (
            candidate
            and " " not in candidate
            and _looks_like_filepath(candidate)
        ):
            headers.append((idx, candidate))

    if len(headers) < 2:
        return {}

    drafts: dict[str, str] = {}
    for pos, (line_idx, filepath) in enumerate(headers):
        start = line_idx + 1
        end = headers[pos + 1][0] if pos + 1 < len(headers) else len(lines)
        content = "".join(lines[start:end]).strip("\n")
        if content:
            drafts[filepath] = content + "\n"
    return drafts


# ---------------------------------------------------------------------------
# 3. Verifier
# ---------------------------------------------------------------------------


_PYTEST_TIMING_RE = re.compile(r" in \d+\.\d+s")


def _normalize_pytest_output(text: str) -> str:
    """Strip variable parts from pytest output for fingerprinting.

    Removes timing information (``in 0.41s``) so that consecutive runs
    with the same failures produce identical fingerprints.
    """
    return _PYTEST_TIMING_RE.sub("", text)


def _run_pytest(root: Path, extra_env: dict[str, str] | None = None) -> list[str]:
    """Run ``pytest`` inside *root* and return failure logs (empty on success)."""
    env = {**os.environ, **(extra_env or {})}
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", str(root), "-v", "--tb=short"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(root),
        env=env,
    )
    if result.returncode != 0:
        return [result.stdout + "\n" + result.stderr]
    return []


def _prepare_sandbox(root: Path, code_drafts: dict[str, str]) -> None:
    """Write *code_drafts* into *root* and make imports work.

    Strategy:
    1. If the coder produced a ``pyproject.toml``, run ``pip install -e .``
       inside *root* so normal ``import pkg`` statements just work.
    2. Otherwise generate a ``conftest.py`` that adds all package-containing
       directories to ``sys.path`` (handles both flat and ``src/`` layouts).
    3. Never overwrite a coder-generated ``conftest.py``.
    """
    # Write all files
    for rel_path, source in code_drafts.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)

    # Attempt editable install if pyproject.toml is present
    if (root / "pyproject.toml").exists():
        pip_result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pip", "install", "-e", str(root),
             "--no-build-isolation", "--quiet"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(root),
        )
        if pip_result.returncode == 0:
            return  # install succeeded — imports will work

    # Fallback: generate a conftest that sets up sys.path
    if (root / "conftest.py").exists():
        return  # don't overwrite coder-generated conftest

    # Collect directories that look like Python packages (contain __init__.py)
    # and their parent directories.
    path_roots: set[str] = set()
    path_roots.add(str(root))  # always include the project root
    if (root / "src").is_dir():
        path_roots.add(str(root / "src"))
    for init in root.rglob("__init__.py"):
        # Add the parent of the top-level package directory
        pkg_dir = init.parent
        # Walk up to find the top-level package root
        while (pkg_dir.parent / "__init__.py").exists():
            pkg_dir = pkg_dir.parent
        path_roots.add(str(pkg_dir.parent))

    path_lines = "\n".join(
        f'    sys.path.insert(0, {p!r})' for p in sorted(path_roots)
    )
    (root / "conftest.py").write_text(
        "import sys\n\n\n"
        "def pytest_configure(config):\n"
        f"{path_lines}\n"
    )


def _ensure_importable(root: Path, code_drafts: dict[str, str]) -> None:
    """Make sure ``import pkg`` works when running pytest in *root*.

    Unlike ``_prepare_sandbox``, files already exist on disk (write mode).
    This function only does the import-plumbing step:
    1. ``pip install -e .`` if a pyproject.toml exists.
    2. Otherwise generate a ``conftest.py`` with ``sys.path`` entries.
    """
    if (root / "pyproject.toml").exists():
        pip_result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pip", "install", "-e", str(root),
             "--no-build-isolation", "--quiet"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(root),
        )
        if pip_result.returncode == 0:
            return

    # Fallback: conftest-based sys.path manipulation
    if (root / "conftest.py").exists():
        return

    path_roots: set[str] = set()
    path_roots.add(str(root))
    if (root / "src").is_dir():
        path_roots.add(str(root / "src"))
    for init in root.rglob("__init__.py"):
        pkg_dir = init.parent
        while (pkg_dir.parent / "__init__.py").exists():
            pkg_dir = pkg_dir.parent
        path_roots.add(str(pkg_dir.parent))

    path_lines = "\n".join(
        f'    sys.path.insert(0, {p!r})' for p in sorted(path_roots)
    )
    (root / "conftest.py").write_text(
        "import sys\n\n\n"
        "def pytest_configure(config):\n"
        f"{path_lines}\n"
    )


def _check_syntax(code_drafts: dict[str, str]) -> list[str]:
    """Check every ``.py`` draft for syntax errors using ``ast.parse``.

    Returns a list of human-readable error strings (empty if all OK).
    Running this *before* pytest gives the reflector a clear, actionable
    message instead of a cryptic collection error.
    """
    errors: list[str] = []
    for fpath, source in sorted(code_drafts.items()):
        if not fpath.endswith(".py"):
            continue
        try:
            ast.parse(source, filename=fpath)
        except SyntaxError as exc:
            lineno = exc.lineno or "?"
            msg = exc.msg or "unknown syntax error"
            errors.append(f"SyntaxError in {fpath} line {lineno}: {msg}")
    return errors


def _check_pyproject_toml(code_drafts: dict[str, str]) -> list[str]:
    """Validate ``pyproject.toml`` has the minimum fields for ``pip install -e .``.

    Checks:
    1. ``[build-system]`` table exists with ``requires`` and ``build-backend``.
    2. ``[project]`` table exists with ``name``.
    3. No mixed build-backend configuration (e.g. setuptools + hatchling).
    """
    source = code_drafts.get("pyproject.toml")
    if source is None:
        return []

    errors: list[str] = []

    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            try:
                import tomllib  # type: ignore[import-not-found]
            except ModuleNotFoundError:
                import tomli as tomllib  # type: ignore[import-not-found,no-redef]
        data = tomllib.loads(source)
    except Exception as exc:
        return [f"pyproject.toml: invalid TOML syntax — {exc}"]

    # Check [build-system]
    build_system = data.get("build-system")
    if build_system is None:
        errors.append(
            "pyproject.toml: missing [build-system] table. "
            "Add: [build-system]\\nrequires = [\"hatchling\"]\\n"
            "build-backend = \"hatchling.build\""
        )
    else:
        if "requires" not in build_system:
            errors.append(
                "pyproject.toml: [build-system] missing 'requires'. "
                "Add: requires = [\"hatchling\"]"
            )
        if "build-backend" not in build_system:
            errors.append(
                "pyproject.toml: [build-system] missing 'build-backend'. "
                "Add: build-backend = \"hatchling.build\""
            )

    # Check [project]
    project = data.get("project")
    if project is None:
        errors.append(
            "pyproject.toml: missing [project] table with 'name'."
        )
    elif "name" not in project:
        errors.append(
            "pyproject.toml: [project] missing 'name' field."
        )

    # Detect mixed build backends
    backend = ""
    if build_system:
        backend = str(build_system.get("build-backend", ""))
    has_setuptools_config = "tool" in data and "setuptools" in data["tool"]
    has_hatch_config = "tool" in data and "hatch" in data["tool"]
    if has_setuptools_config and "hatch" in backend:
        errors.append(
            "pyproject.toml: mixed backends — [tool.setuptools] config "
            "found but build-backend is hatchling. Remove [tool.setuptools] "
            "sections or switch build-backend to setuptools."
        )
    if has_hatch_config and "setuptools" in backend:
        errors.append(
            "pyproject.toml: mixed backends — [tool.hatch] config "
            "found but build-backend is setuptools. Remove [tool.hatch] "
            "sections or switch build-backend to hatchling."
        )

    return errors


def _llm_triage(code_drafts: dict[str, str], llm: Any) -> list[str]:
    """Use the LLM to review code drafts for structural issues before pytest.

    Replaces the deterministic ``_check_import_consistency``,
    ``_check_cross_module_contracts``, and ``_check_duplicate_modules``
    heuristics with a single LLM call that catches the same classes of
    bugs (import mismatches, argument-count errors, return-unpacking
    mismatches, layout conflicts) plus issues the heuristics couldn't.

    Returns a list of human-readable error strings (empty if LLM says OK).
    """
    prompt = _load_prompt("EXPLORER_PROMPT_TRIAGE", "triage.md")

    # Build a concise code listing for the LLM
    sections: list[str] = []
    for fpath in sorted(code_drafts):
        if not fpath.endswith(".py"):
            continue
        sections.append(f"### {fpath}\n```python\n{code_drafts[fpath]}\n```")

    if not sections:
        return []

    user_msg = "## Code files to review\n\n" + "\n\n".join(sections)

    response = _invoke_llm(llm, [
        SystemMessage(content=prompt),
        HumanMessage(content=user_msg),
    ])
    text = response.content.strip()

    if text == "LGTM" or not text:
        return []

    # Parse bullet-point issues
    errors: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            errors.append(line[2:])
        elif line and not line.startswith("#"):
            # Non-empty, non-header line — include as-is
            errors.append(line)
    return errors


def _warn_stale_files(
    root: Path, code_drafts: dict[str, str]
) -> None:
    """Remove ``.py`` files in *root* that are not part of *code_drafts*.

    When the coder rewrites all files each iteration, leftover files from a
    previous run may confuse ``pytest``.  This keeps the output directory in
    sync with the current code drafts.
    """
    expected = {root / fpath for fpath in code_drafts}
    for py_file in root.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        if py_file not in expected:
            logger.warning("Removing stale file: %s", py_file)
            py_file.unlink()


def _resolve_duplicate_layouts(
    merged_drafts: dict[str, str],
    new_drafts: dict[str, str],
) -> None:
    """Auto-prune one side of a flat-vs-src layout conflict in place.

    When *merged_drafts* contains the same package under both ``pkg/`` (flat)
    and ``src/pkg/`` (src layout), determine which layout the *new_drafts*
    predominantly use and remove conflicting files from the other layout.

    Mutates *merged_drafts* in place; returns nothing.
    """
    flat_pkgs: set[str] = set()
    src_pkgs: set[str] = set()
    for fpath in merged_drafts:
        parts = fpath.split("/")
        if len(parts) >= 2 and parts[0] == "src":
            src_pkgs.add(parts[1])
        elif len(parts) >= 2 and parts[0] != "tests":
            flat_pkgs.add(parts[0])

    dupes = flat_pkgs & src_pkgs
    if not dupes:
        return

    # Determine preferred layout from new_drafts: count how many new files
    # use src/ vs flat for each conflicting package.
    for pkg in dupes:
        src_count = sum(
            1 for fp in new_drafts
            if fp.split("/")[0] == "src"
            and len(fp.split("/")) >= 2
            and fp.split("/")[1] == pkg
        )
        flat_count = sum(
            1 for fp in new_drafts
            if fp.split("/")[0] == pkg
        )
        # If the coder is emitting src/ files, keep src layout; otherwise
        # keep flat.  On ties or no new files, prefer src/ (modern convention).
        keep_src = src_count >= flat_count
        prefix_to_remove = f"{pkg}/" if keep_src else f"src/{pkg}/"
        removed = [fp for fp in merged_drafts if fp.startswith(prefix_to_remove)]
        for fp in removed:
            logger.warning(
                "Auto-pruning duplicate layout file: %s (keeping %s layout)",
                fp,
                "src/" if keep_src else "flat",
            )
            del merged_drafts[fp]


def verifier(state: ScientificState) -> dict[str, Any]:
    """Run ``pytest`` against the generated code.

    In **sandbox mode** (default, no ``output_dir``), code drafts are written
    to a temporary directory that is cleaned up after the run.

    In **write mode** (``output_dir`` is set), ``pytest`` runs directly inside
    the output directory — the same directory the coder already wrote to.
    """
    code_drafts: dict[str, str] = state.get("code_drafts", {})
    if not code_drafts:
        transcript = list(state.get("transcript") or [])
        transcript.append(make_entry(
            "human", "No code drafts to verify.",
            node="verifier",
            step=state.get("iteration_count", 0) + 1,
            phase=state.get("current_phase", 0),
        ))
        return {
            "test_logs": ["No code drafts to verify."],
            "iteration_count": state.get("iteration_count", 0) + 1,
            "_phase_iteration_count": state.get("_phase_iteration_count", 0) + 1,
            "transcript": transcript,
        }

    # Quick structural and import checks before running pytest
    syntax_errors = _check_syntax(code_drafts)
    pre_errors = _check_pyproject_toml(code_drafts)
    pre_errors += syntax_errors
    if not pre_errors:
        # LLM triage replaces the old AST-based heuristics
        # (_check_duplicate_modules, _check_import_consistency,
        #  _check_cross_module_contracts).
        llm = get_llm()
        pre_errors = _llm_triage(code_drafts, llm)

    # Track which files are clean (no syntax or import errors)
    files_with_errors: set[str] = set()
    for err in syntax_errors:
        # Extract filename from "SyntaxError in <path> line ..."
        parts = err.split(" line ")
        if parts:
            fpath = parts[0].replace("SyntaxError in ", "").strip()
            files_with_errors.add(fpath)
    for err in pre_errors:
        # Extract filenames from import/duplicate error messages
        for fpath in code_drafts:
            if fpath in err:
                files_with_errors.add(fpath)

    clean_files = sorted(
        fpath for fpath in code_drafts
        if fpath.endswith(".py") and fpath not in files_with_errors
    )

    output_dir = state.get("output_dir", "")

    if output_dir:
        # Write mode — run pytest in the real output directory
        root = Path(output_dir).resolve()
        # Warn about stale .py files left from a previous run
        _warn_stale_files(root, code_drafts)
        _ensure_importable(root, code_drafts)
        logs = pre_errors or _run_pytest(root)
    else:
        # Sandbox mode — temp dir with auto-cleanup
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _prepare_sandbox(root, code_drafts)
            print(f"[explorer] Sandbox directory: {root}")
            logs = pre_errors or _run_pytest(root)

    # Stuck-loop detection: fingerprint the errors and track repeats
    normalised = [_normalize_pytest_output(l) for l in logs] if logs else []
    error_fingerprint = "\n".join(sorted(normalised)) if normalised else ""
    prev_fingerprint = state.get("_prev_error_fingerprint", "")
    if logs and error_fingerprint == prev_fingerprint:
        error_repeat_count = state.get("_error_repeat_count", 0) + 1
    else:
        error_repeat_count = 1 if logs else 0

    # Total failure count for this phase
    phase_error_count = state.get("_phase_error_count", 0) + (1 if logs else 0)
    phase_iter_count = state.get("_phase_iteration_count", 0) + 1

    # Count errors for best-iteration tracking
    current_error_count = len(logs)

    # Build verified_fixes: if a previous reflector suggestion fixed an
    # error category, record it as a permanent constraint.
    verified_fixes = list(state.get("verified_fixes") or [])
    prev_syntax_errors = state.get("_prev_syntax_file_count", 0)
    current_syntax_errors = len(files_with_errors)
    if prev_syntax_errors > 0 and current_syntax_errors < prev_syntax_errors:
        # Syntax errors decreased — the reflector's advice worked.
        # Extract the fix pattern from the reflection.
        reflection = state.get("reflection", "")
        if reflection and "raw" in reflection.lower() and "docstring" in reflection.lower():
            fix = "Use raw docstrings (r\"\"\"...\"\"\") for any string containing backslashes"
            if fix not in verified_fixes:
                verified_fixes.append(fix)
        if reflection and "signature" in reflection.lower():
            fix = "Function signatures MUST match what tests expect — check test imports before defining functions"
            if fix not in verified_fixes:
                verified_fixes.append(fix)

    # Track best code drafts for this phase (fewest errors)
    best_drafts = state.get("best_code_drafts") or {}
    best_count = state.get("best_error_count", -1)
    if best_count < 0 or current_error_count < best_count:
        best_drafts = dict(code_drafts)
        best_count = current_error_count

    # Rollback: if error count has increased for 3+ consecutive iterations,
    # revert to the best-known code.
    if (
        current_error_count > best_count > 0
        and error_repeat_count == 0  # different error each time (oscillating)
        and phase_error_count >= 4
        and phase_error_count % 3 == 0  # every 3 worsening iterations
    ):
        logger.warning(
            "Reverting to best code snapshot (%d errors vs current %d)",
            best_count, current_error_count,
        )
        code_drafts = dict(best_drafts)
        if output_dir:
            _write_code_drafts(code_drafts, output_dir)

    # Detect collection errors (no tests actually ran)
    log_text = "\n".join(logs) if logs else ""
    collection_error = bool(logs) and (
        "collected 0 items" in log_text
        or "Interrupted: " in log_text and "error during collection" in log_text
    )

    transcript = list(state.get("transcript") or [])
    if logs:
        transcript.append(make_entry(
            "human", f"Tests FAILED:\n```\n{log_text[:3000]}\n```",
            node="verifier",
            step=state.get("iteration_count", 0) + 1,
            phase=state.get("current_phase", 0),
        ))
    else:
        transcript.append(make_entry(
            "human", "All tests passed \u2713",
            node="verifier",
            step=state.get("iteration_count", 0) + 1,
            phase=state.get("current_phase", 0),
            summary="Tests: All passed \u2713",
        ))

    return {
        "test_logs": logs,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "_prev_error_fingerprint": error_fingerprint,
        "_error_repeat_count": error_repeat_count,
        "_phase_error_count": phase_error_count,
        "_phase_iteration_count": phase_iter_count,
        "_prev_syntax_file_count": current_syntax_errors,
        "_collection_error": collection_error,
        "clean_files": clean_files,
        "verified_fixes": verified_fixes,
        "best_code_drafts": best_drafts,
        "best_error_count": best_count,
        "code_drafts": code_drafts,
        "transcript": transcript,
    }


# ---------------------------------------------------------------------------
# 4. Reflector
# ---------------------------------------------------------------------------

def _extract_findings(text: str) -> tuple[str, list[str]]:
    """Split a reflector response into analysis and key findings.

    The reflector prompt asks the LLM to end its response with a
    ``## Key Findings`` section.  This helper splits on that header
    and extracts ``"- …"`` bullet lines as individual findings.

    Returns ``(analysis_text, findings_list)``.
    """
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


def reflector(state: ScientificState) -> dict[str, Any]:
    """Analyse test failures and extract key findings for ground_truth.md.

    Makes a single LLM call that produces both the debugging analysis
    and a ``## Key Findings`` section.  Findings are parsed locally and
    deduplicated against existing ``ground_truth``.
    """
    llm = get_llm()

    reflector_prompt = _load_prompt("EXPLORER_PROMPT_REFLECTOR", "reflector.md")

    log_text = "\n---\n".join(state.get("test_logs", []))

    # Include failing source code so the reflector can see what went wrong
    code_drafts: dict[str, str] = state.get("code_drafts", {})
    clean_files: list[str] = state.get("clean_files") or []
    failing_sources: list[str] = []
    for fpath, source in sorted(code_drafts.items()):
        if fpath.endswith(".py") and fpath not in clean_files:
            failing_sources.append(f"### {fpath}\n```python\n{source}\n```")

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
    response = _invoke_llm(llm, [
        SystemMessage(content=reflector_prompt),
        HumanMessage(content=reflector_user_msg),
    ])

    # Single-call: parse analysis and key findings from the response.
    reflection, new_raw_findings = _extract_findings(response.content)

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
    transcript = list(state.get("transcript") or [])
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
        "_prompt_summary": reflector_user_msg,
        "transcript": transcript,
    }
