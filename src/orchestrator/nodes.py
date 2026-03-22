"""Core graph nodes for the Scientific Loop.

Each function accepts the current ``ScientificState`` and returns a partial
state dict that LangGraph merges back into the graph state.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from .state import ScientificState

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


# ---------------------------------------------------------------------------
# 1. Planner
# ---------------------------------------------------------------------------


def planner(state: ScientificState) -> dict[str, Any]:
    """Analyse the task and produce a phased implementation plan.

    On the first call the planner generates a full multi-phase plan.
    On subsequent calls (reflection loop) it revises only the current phase.
    The full plan and its current phase are returned so that downstream
    nodes always receive structured phase information.
    """
    llm = get_llm()

    # Detect whether this is a revision (reflection loop) or a fresh plan
    existing_phases: list[dict[str, Any]] = list(state.get("plan_phases") or [])
    is_revision = bool(state.get("reflection")) and bool(existing_phases)

    user_parts: list[str] = [f"## Task\n{state['task_description']}"]
    if state.get("reflection"):
        user_parts.append(f"## Previous error analysis\n{state['reflection']}")
    if state.get("mathematical_constants"):
        user_parts.append(
            f"## Constants\n{state['mathematical_constants']}"
        )
    if state.get("skills_context"):
        user_parts.append(state["skills_context"])

    prompt = _load_prompt("EXPLORER_PROMPT_PLANNER", "planner.md")
    user_message = "\n\n".join(user_parts)
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=user_message),
    ])

    raw_plan = response.content

    if is_revision:
        # Only update the current phase's description; keep the rest intact
        current_idx = state.get("current_phase", 0)
        revised_phases = _parse_plan_phases(raw_plan)
        # Use the first revised phase to update the current one
        if revised_phases:
            existing_phases[current_idx]["description"] = revised_phases[0]["description"]
            if revised_phases[0]["files"]:
                existing_phases[current_idx]["files"] = revised_phases[0]["files"]
        plan_phases = existing_phases
        current_phase = current_idx
    else:
        # Fresh plan: parse all phases
        plan_phases = _parse_plan_phases(raw_plan)
        current_phase = 0

    # Set `plan` to the current phase description for backward compatibility
    plan_text = plan_phases[current_phase]["description"]

    _write_plan_artifact(plan_phases, current_phase)

    return {
        "plan": plan_text,
        "plan_phases": plan_phases,
        "current_phase": current_phase,
        "_prompt_summary": user_message,
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

    return {
        "plan": plan_text,
        "plan_phases": phases,
        "current_phase": next_idx,
        "reflection": "",  # clear stale reflection for new phase
        "ground_truth": [],  # clear stale findings for new phase
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

    # If we have error context from a previous iteration, show the coder
    # what went wrong so it can fix its own mistakes directly.
    if state.get("reflection"):
        error_repeat = state.get("_error_repeat_count", 0)
        if error_repeat >= 3:
            user_parts.append(
                f"## ⚠️  CRITICAL — same error for {error_repeat} consecutive iterations\n"
                "Your previous code did NOT fix the problem.  Read the error "
                "analysis below VERY carefully and make DIFFERENT choices this time."
            )
        user_parts.append(
            f"## Previous error analysis\n{state['reflection']}"
        )
    if state.get("test_logs"):
        log_text = "\n---\n".join(state["test_logs"])
        user_parts.append(
            f"## Test failures to fix\n```\n{log_text}\n```"
        )

    # Inform coder about files that already exist from prior phases
    existing_drafts = state.get("code_drafts") or {}
    if existing_drafts:
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
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=user_message),
    ])

    raw_content = response.content
    new_drafts = _parse_code_blocks(raw_content)

    # Merge new drafts into accumulated drafts (new files override old)
    merged_drafts = dict(existing_drafts)
    merged_drafts.update(new_drafts)

    # In write mode, persist files to the output directory
    output_dir = state.get("output_dir", "")
    if output_dir:
        _write_code_drafts(new_drafts, output_dir)

    return {"code_drafts": merged_drafts, "coder_raw_response": raw_content, "_prompt_summary": user_message}


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
    """Extract fenced code blocks whose info-strings look like file paths."""
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
    return drafts


# ---------------------------------------------------------------------------
# 3. Verifier
# ---------------------------------------------------------------------------


def _run_pytest(root: Path, extra_env: dict[str, str] | None = None) -> list[str]:
    """Run ``pytest`` inside *root* and return failure logs (empty on success)."""
    env = {**os.environ, **(extra_env or {})}
    result = subprocess.run(  # noqa: S603
        ["python", "-m", "pytest", str(root), "-v", "--tb=short"],
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
            ["python", "-m", "pip", "install", "-e", str(root),
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
            ["python", "-m", "pip", "install", "-e", str(root),
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


def _check_import_consistency(code_drafts: dict[str, str]) -> list[str]:
    """Check that names imported in test files are actually defined in source modules.

    Returns a list of human-readable error strings (empty if consistent).
    Only checks ``from <module> import <name>`` statements in test files
    against top-level definitions in non-test ``.py`` files.
    """
    # Build a map: module_dotted_path -> set of top-level names defined
    defined: dict[str, set[str]] = {}
    for fpath, source in code_drafts.items():
        if not fpath.endswith(".py"):
            continue
        # Skip test files and __init__.py
        basename = fpath.rsplit("/", 1)[-1] if "/" in fpath else fpath
        if basename.startswith("test_") or basename == "__init__.py":
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        names: set[str] = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
        # Convert file path to dotted module path
        mod_path = fpath.removesuffix(".py").replace("/", ".")
        # Also store without leading "src." for src-layout projects
        defined[mod_path] = names
        if mod_path.startswith("src."):
            defined[mod_path.removeprefix("src.")] = names

    # Scan test files for `from X import Y` and cross-check
    errors: list[str] = []
    for fpath, source in code_drafts.items():
        if not fpath.endswith(".py"):
            continue
        basename = fpath.rsplit("/", 1)[-1] if "/" in fpath else fpath
        if not basename.startswith("test_"):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            mod = node.module
            # Flag `from src.pkg...` imports — they never work at runtime
            if mod.startswith("src."):
                correct = mod.removeprefix("src.")
                errors.append(
                    f"Bad import path: {fpath} uses 'from {mod} import ...', "
                    f"but 'src' is not a package. Use 'from {correct} import ...' instead."
                )
                continue
            if mod not in defined:
                continue
            for alias in node.names:
                name = alias.name
                if name not in defined[mod]:
                    available = ", ".join(sorted(defined[mod])) or "(nothing)"
                    errors.append(
                        f"Import mismatch: {fpath} imports '{name}' from "
                        f"'{mod}', but '{mod}' only defines: {available}"
                    )
    return errors


def _check_duplicate_modules(code_drafts: dict[str, str]) -> list[str]:
    """Detect packages that appear in both flat and ``src/`` layouts.

    For example, if *code_drafts* contains both ``pkg/mod.py`` and
    ``src/pkg/mod.py`` (or even just ``pkg/__init__.py`` and
    ``src/pkg/__init__.py``), this is almost certainly a mistake that will
    confuse Python's import system.

    Returns a list of human-readable error strings (empty if no duplicates).
    """
    flat_pkgs: set[str] = set()
    src_pkgs: set[str] = set()
    for fpath in code_drafts:
        parts = fpath.split("/")
        if len(parts) >= 2 and parts[0] == "src":
            src_pkgs.add(parts[1])
        elif len(parts) >= 2 and parts[0] != "tests":
            flat_pkgs.add(parts[0])

    dupes = flat_pkgs & src_pkgs
    return [
        f"Duplicate layout: package '{pkg}' exists in both '{pkg}/' (flat) "
        f"and 'src/{pkg}/' (src layout). Use only ONE layout — remove the "
        f"duplicate and ensure all imports match the chosen layout."
        for pkg in sorted(dupes)
    ]


def verifier(state: ScientificState) -> dict[str, Any]:
    """Run ``pytest`` against the generated code.

    In **sandbox mode** (default, no ``output_dir``), code drafts are written
    to a temporary directory that is cleaned up after the run.

    In **write mode** (``output_dir`` is set), ``pytest`` runs directly inside
    the output directory — the same directory the coder already wrote to.
    """
    code_drafts: dict[str, str] = state.get("code_drafts", {})
    if not code_drafts:
        return {
            "test_logs": ["No code drafts to verify."],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    # Quick structural and import checks before running pytest
    pre_errors = _check_duplicate_modules(code_drafts)
    if not pre_errors:
        pre_errors = _check_import_consistency(code_drafts)

    output_dir = state.get("output_dir", "")

    if output_dir:
        # Write mode — run pytest in the real output directory
        root = Path(output_dir).resolve()
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
    error_fingerprint = "\n".join(sorted(logs)) if logs else ""
    prev_fingerprint = state.get("_prev_error_fingerprint", "")
    if logs and error_fingerprint == prev_fingerprint:
        error_repeat_count = state.get("_error_repeat_count", 0) + 1
    else:
        error_repeat_count = 1 if logs else 0

    return {
        "test_logs": logs,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "_prev_error_fingerprint": error_fingerprint,
        "_error_repeat_count": error_repeat_count,
    }


# ---------------------------------------------------------------------------
# 4. Reflector
# ---------------------------------------------------------------------------

def reflector(state: ScientificState) -> dict[str, Any]:
    """Analyse test failures and extract key findings for ground_truth.md."""
    llm = get_llm()

    reflector_prompt = _load_prompt("EXPLORER_PROMPT_REFLECTOR", "reflector.md")
    findings_prompt = _load_prompt("EXPLORER_PROMPT_FINDINGS", "findings.md")

    log_text = "\n---\n".join(state.get("test_logs", []))
    reflector_user_msg = f"## Test logs\n```\n{log_text}\n```"
    response = llm.invoke([
        SystemMessage(content=reflector_prompt),
        HumanMessage(content=reflector_user_msg),
    ])
    reflection = response.content

    # Extract key findings
    findings_response = llm.invoke([
        SystemMessage(content=findings_prompt),
        HumanMessage(
            content=f"## Reflection\n{reflection}\n\n## Test logs\n```\n{log_text}\n```"
        ),
    ])

    existing = list(state.get("ground_truth", []))
    raw = findings_response.content.strip()
    if raw != "NONE":
        seen = set(existing)
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("- ") and line not in seen:
                existing.append(line)
                seen.add(line)

    return {"reflection": reflection, "ground_truth": existing, "_prompt_summary": reflector_user_msg}
