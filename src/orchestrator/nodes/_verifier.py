"""Verifier node — runs pytest against generated code."""

from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
import sys
import tempfile
import time as _time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from . import _shared
from ._shared import (
    _format_code_listing,
    _invoke_llm,
    _load_prompt,
    make_llm_call_record,
)
from ..state import ScientificState
from ..transcript import make_entry


logger = logging.getLogger(__name__)

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
    # Force non-interactive matplotlib backend so plt.show() never blocks.
    env.setdefault("MPLBACKEND", "Agg")
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


# Packages that must never be shadowed by coder-generated directories.
# Includes both test infrastructure and common scientific-stack packages.
_SHADOW_BLACKLIST = frozenset({
    "pytest", "numpy", "scipy", "matplotlib", "click", "pip",
    "setuptools", "pkg_resources", "importlib", "typing",
})


def _check_shadowed_packages(code_drafts: dict[str, str]) -> list[str]:
    """Detect code-draft directories that shadow well-known packages.

    If the coder generates ``pytest/__init__.py`` or ``numpy/foo.py``,
    ``python -m pytest`` (or any import of that package) will find the
    local directory instead of the installed package, causing cryptic
    failures.

    Returns a list of human-readable error strings (empty if all OK).
    """
    # Collect top-level directory names from the draft paths
    top_dirs: set[str] = set()
    for fpath in code_drafts:
        parts = fpath.replace("\\", "/").split("/")
        if len(parts) >= 2:  # noqa: PLR2004 — has a subdirectory
            top_dirs.add(parts[0])

    errors: list[str] = []
    for dirname in sorted(top_dirs & _SHADOW_BLACKLIST):
        errors.append(
            f"CRITICAL: directory '{dirname}/' shadows the installed "
            f"'{dirname}' package.  Rename or remove all files under "
            f"'{dirname}/' — they will break imports and pytest execution."
        )
    return errors


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


def _llm_triage(code_drafts: dict[str, str], llm: Any) -> tuple[list[str], dict | None]:
    """Use the LLM to review code drafts for structural issues before pytest.

    Replaces the deterministic ``_check_import_consistency``,
    ``_check_cross_module_contracts``, and ``_check_duplicate_modules``
    heuristics with a single LLM call that catches the same classes of
    bugs (import mismatches, argument-count errors, return-unpacking
    mismatches, layout conflicts) plus issues the heuristics couldn't.

    Returns a tuple of (error_list, llm_call_record_or_None).
    """
    prompt = _load_prompt("EXPLORER_PROMPT_TRIAGE", "triage.md")

    # Build a concise code listing for the LLM
    sections = _format_code_listing(code_drafts)

    if not sections:
        return [], None

    user_msg = "## Code files to review\n\n" + "\n\n".join(sections)

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_msg),
    ]
    t0 = _time.monotonic()
    response = _invoke_llm(llm, messages)
    duration = _time.monotonic() - t0

    call_record = make_llm_call_record(
        node="verifier",
        messages=messages,
        response=response,
        duration_s=duration,
        label="triage",
    )

    text = response.content.strip()

    if text == "LGTM" or not text:
        return [], call_record

    # Parse bullet-point issues
    errors: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            errors.append(line[2:])
        elif line and not line.startswith("#"):
            # Non-empty, non-header line — include as-is
            errors.append(line)
    return errors, call_record


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


def verifier(state: ScientificState) -> dict[str, Any]:
    """Run ``pytest`` against the generated code.

    In **sandbox mode** (default, no ``output_dir``), code drafts are written
    to a temporary directory that is cleaned up after the run.

    In **write mode** (``output_dir`` is set), ``pytest`` runs directly inside
    the output directory — the same directory the coder already wrote to.
    """
    code_drafts: dict[str, str] = state.get("code_drafts", {})
    if not code_drafts:
        transcript: list[dict] = []
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
    shadow_errors = _check_shadowed_packages(code_drafts)
    syntax_errors = _check_syntax(code_drafts)
    pre_errors = shadow_errors + _check_pyproject_toml(code_drafts)
    pre_errors += syntax_errors
    llm_calls: list[dict] = []
    if not pre_errors and not state.get("tool_calling"):
        # LLM triage replaces the old AST-based heuristics
        # (_check_duplicate_modules, _check_import_consistency,
        #  _check_cross_module_contracts).
        # Skipped in tool-calling mode — the coder's inner tool loop
        # already catches structural issues.
        llm = _shared.get_llm()
        pre_errors, triage_record = _llm_triage(code_drafts, llm)
        if triage_record:
            llm_calls.append(triage_record)

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

    phase_iter_count = state.get("_phase_iteration_count", 0) + 1

    # Detect collection errors (no tests actually ran)
    log_text = "\n".join(logs) if logs else ""
    collection_error = bool(logs) and (
        "collected 0 items" in log_text
        or "Interrupted: " in log_text and "error during collection" in log_text
    )

    transcript: list[dict] = []
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
        "_phase_iteration_count": phase_iter_count,
        "_collection_error": collection_error,
        "clean_files": clean_files,
        "code_drafts": code_drafts,
        "transcript": transcript,
        "_llm_calls": llm_calls,
    }
