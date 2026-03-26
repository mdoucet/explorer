"""Coder node — generates Python source code from the plan."""

from __future__ import annotations

import ast
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from . import _shared
from ._shared import (
    _format_code_listing,
    _invoke_llm,
    _load_prompt,
)
from ..state import ScientificState
from ..transcript import format_history, make_entry

logger = logging.getLogger(__name__)


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
    llm = _shared.get_llm()
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
            source_parts = _format_code_listing(
                existing_drafts, clean_files=clean,
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
    # after the coder has already had one chance to fix (iteration >= 2).
    # Both test files and source files use the same criterion: only protect
    # files present in clean_files (all tests passing, no errors).
    # This ensures test files with bugs (e.g. wrong subprocess invocation)
    # can still be fixed while truly passing test files remain untouched.
    phase_iter = state.get("_phase_iteration_count", 0)
    if is_revision and existing_drafts and phase_iter >= 2:
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
