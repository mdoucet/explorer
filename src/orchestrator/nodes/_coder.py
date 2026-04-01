"""Coder node — generates Python source code from the plan."""

from __future__ import annotations

import ast
import logging
import subprocess
import sys
import tempfile
import time as _time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from . import _shared
from ._shared import (
    _format_code_listing,
    _invoke_llm,
    _load_prompt,
    make_llm_call_record,
)
from ._verifier import _check_shadowed_packages, _check_syntax, _prepare_sandbox, _run_pytest
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


# ---------------------------------------------------------------------------
# Inner-loop quick verification
# ---------------------------------------------------------------------------
MAX_INNER_ITERATIONS = 3


def _quick_verify(code_drafts: dict[str, str]) -> list[str]:
    """Run syntax checks + pytest in a temp sandbox.

    Returns a list of error strings (empty on success).
    This is a lightweight version of the verifier — no LLM triage,
    no stuck-loop detection, just fast feedback for the coder's inner loop.
    """
    if not code_drafts:
        return []

    shadow_errors = _check_shadowed_packages(code_drafts)
    if shadow_errors:
        return shadow_errors

    syntax_errors = _check_syntax(code_drafts)
    if syntax_errors:
        return syntax_errors

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _prepare_sandbox(root, code_drafts)
        return _run_pytest(root)


def _append_coder_transcript(
    state: ScientificState,
    new_drafts: dict[str, str],
    deletions: set[str],
    *,
    inner_iterations: int = 0,
) -> list[dict]:
    """Build updated transcript with coder summary entry."""
    transcript: list[dict] = []
    file_list = ", ".join(sorted(new_drafts.keys()))
    parts = [f"Generated {len(new_drafts)} file(s): {file_list}"]
    if deletions:
        parts.append(f"Deleted: {', '.join(sorted(deletions))}")
    if inner_iterations > 0:
        parts.append(f"Inner-loop self-corrections: {inner_iterations}")
    transcript.append(make_entry(
        "ai", "\n".join(parts),
        node="coder",
        step=state.get("iteration_count", 0),
        phase=state.get("current_phase", 0),
    ))
    return transcript


_DELETE_MARKER = "# DELETE"


def _split_deletions(drafts: dict[str, str]) -> tuple[dict[str, str], set[str]]:
    """Separate deletion markers from real code blocks.

    Returns ``(clean_drafts, deletions)`` where *clean_drafts* has
    deletion entries removed and *deletions* is the set of paths to delete.
    """
    deletions: set[str] = set()
    clean: dict[str, str] = {}
    for fp, content in drafts.items():
        if content.strip() == _DELETE_MARKER:
            deletions.add(fp)
        else:
            clean[fp] = content
    return clean, deletions


def _merge_and_clean(
    new_drafts: dict[str, str],
    existing_drafts: dict[str, str],
    deletions: set[str],
) -> dict[str, str]:
    """Merge new drafts into existing, apply deletions and layout resolution."""
    merged = dict(existing_drafts)
    merged.update(new_drafts)
    for fp in deletions:
        merged.pop(fp, None)
    _resolve_duplicate_layouts(merged, new_drafts)
    return merged


def _finalize_output(
    state: ScientificState,
    raw_content: str,
    new_drafts: dict[str, str],
    existing_drafts: dict[str, str],
    deletions: set[str],
    inner_iterations: int,
    user_message: str,
    *,
    merged_drafts: dict[str, str] | None = None,
    llm_calls: list[dict] | None = None,
) -> dict[str, Any]:
    """Build the coder node's return dict.

    Handles disk persistence (write mode) and builds the transcript entry.
    If *merged_drafts* is not provided, computes it from the parts.
    """
    if merged_drafts is None:
        merged_drafts = _merge_and_clean(new_drafts, existing_drafts, deletions)

    output_dir = state.get("output_dir", "")
    if output_dir:
        _write_code_drafts(new_drafts, output_dir)
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
        "_inner_loop_count": inner_iterations,
        "transcript": _append_coder_transcript(
            state, new_drafts, deletions,
            inner_iterations=inner_iterations,
        ),
        "_llm_calls": llm_calls or [],
    }


def _tool_calling_coder(
    state: ScientificState,
    llm: Any,
    prompt: str,
    user_message: str,
    existing_drafts: dict[str, str],
) -> dict[str, Any]:
    """Run the coder as a tool-calling ReAct agent.

    Creates a sandbox seeded with *existing_drafts*, binds tools to the
    LLM, and runs a write-test-fix loop.  Falls back to text parsing if
    the LLM doesn't emit tool calls on the first turn.
    """
    from ._tools import CoderSandbox, make_sandbox_tools, MAX_TOOL_ROUNDS
    from langchain_core.messages import ToolMessage

    sandbox = CoderSandbox(existing_drafts if existing_drafts else None)
    try:
        tools = make_sandbox_tools(sandbox)
        tool_map = {t.name: t for t in tools}
        llm_with_tools = llm.bind_tools(tools)

        messages: list = [
            SystemMessage(content=prompt),
            HumanMessage(content=user_message),
        ]

        tool_rounds = 0
        raw_content = ""
        llm_calls: list[dict] = []

        try:
            for round_num in range(MAX_TOOL_ROUNDS):
                t0 = _time.monotonic()
                response = _invoke_llm(llm_with_tools, messages)
                duration = _time.monotonic() - t0
                messages.append(response)
                raw_content = response.content or ""

                # Collect tool-call results for this round
                round_tool_msgs: list[dict] = []

                if not getattr(response, "tool_calls", None):
                    llm_calls.append(make_llm_call_record(
                        node="coder",
                        messages=messages[:-1],  # exclude the response we just appended
                        response=response,
                        duration_s=duration,
                        label=f"tool-round-{round_num}" if round_num > 0 else "tool-round-0 (text-only fallback)" if round_num == 0 else "",
                    ))
                    if round_num == 0:
                        # LLM ignored tools — fall back to text parsing
                        logger.info(
                            "Tool-calling LLM returned text only; "
                            "falling back to text parsing"
                        )
                        new_drafts = _parse_code_blocks(raw_content)
                        new_drafts, deletions = _split_deletions(new_drafts)
                        return _finalize_output(
                            state, raw_content, new_drafts, existing_drafts,
                            deletions, 0, user_message,
                            llm_calls=llm_calls,
                        )
                    break  # LLM finished after using tools

                for tc in response.tool_calls:
                    name = tc["name"]
                    args = tc["args"]
                    tid = tc["id"]
                    if name in tool_map:
                        try:
                            result = tool_map[name].invoke(args)
                        except Exception as exc:
                            result = f"Error: {exc}"
                    else:
                        result = f"Error: unknown tool '{name}'"
                    messages.append(ToolMessage(content=str(result), tool_call_id=tid))
                    round_tool_msgs.append({
                        "tool": name,
                        "args": args,
                        "result": str(result)[:2000],
                    })

                llm_calls.append(make_llm_call_record(
                    node="coder",
                    messages=messages[:round_num + 2],  # sys + human + prior rounds
                    response=response,
                    duration_s=duration,
                    tool_messages=round_tool_msgs,
                    label=f"tool-round-{round_num}",
                ))
                tool_rounds += 1
        except Exception as inner_exc:
            # Attach partial call records so the fallback path can
            # include them in its output.
            inner_exc.partial_llm_calls = llm_calls  # type: ignore[attr-defined]
            raise

        # Collect final state from sandbox
        final_drafts = sandbox.collect_drafts()
        new_drafts = {
            k: v for k, v in final_drafts.items()
            if k not in existing_drafts or existing_drafts[k] != v
        }
        deletions = set(existing_drafts) - set(final_drafts)

        return _finalize_output(
            state, raw_content, new_drafts, existing_drafts,
            deletions, tool_rounds, user_message,
            merged_drafts=dict(final_drafts),
            llm_calls=llm_calls,
        )
    finally:
        sandbox.cleanup()


def coder(state: ScientificState) -> dict[str, Any]:
    """Generate Python source code from the current plan phase.

    Includes an **inner self-correction loop**: after generating code the
    node runs a quick syntax-check + pytest in a sandbox.  If tests fail,
    the error output is fed back to the LLM for up to
    ``MAX_INNER_ITERATIONS - 1`` additional attempts before returning to
    the outer graph loop.

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

    # ── Tool-calling dispatch ───────────────────────────────────────────
    _fallback_llm_calls: list[dict] = []
    if _shared.supports_tool_calling(llm):
        tool_prompt = _load_prompt(
            "EXPLORER_PROMPT_CODER_TOOLS", "coder_tools.md",
        )
        try:
            return _tool_calling_coder(
                state, llm, tool_prompt, user_message, existing_drafts,
            )
        except Exception as exc:
            # Ollama (and some other providers) fail to parse tool-call JSON
            # when the code contains special characters.  Fall back to text
            # mode rather than crashing the entire run.
            err_str = str(exc).lower()
            if "error parsing tool call" in err_str or "tool_call" in err_str:
                logger.warning(
                    "Tool-calling failed (%s); falling back to text mode",
                    exc,
                )
                # Preserve any LLM calls recorded before the crash
                _fallback_llm_calls = getattr(exc, "partial_llm_calls", [])
            else:
                raise

    # ── Text-mode inner self-correction loop ────────────────────────────
    messages: list = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_message),
    ]
    inner_iterations = 0
    llm_calls: list[dict] = list(_fallback_llm_calls)

    for inner_attempt in range(MAX_INNER_ITERATIONS):
        t0 = _time.monotonic()
        response = _invoke_llm(llm, messages)
        duration = _time.monotonic() - t0
        raw_content = response.content

        llm_calls.append(make_llm_call_record(
            node="coder",
            messages=messages,
            response=response,
            duration_s=duration,
            label=f"inner-loop-{inner_attempt}" if inner_attempt > 0 else "initial",
        ))

        new_drafts = _parse_code_blocks(raw_content)
        new_drafts, deletions = _split_deletions(new_drafts)

        # On revision iterations, protect files verified clean by the
        # outer verifier.  Applied on every inner-loop iteration because
        # the clean-file list reflects the outer graph state.
        phase_iter = state.get("_phase_iteration_count", 0)
        if is_revision and existing_drafts and phase_iter >= 2:
            clean = set(state.get("clean_files") or [])
            protected = {fp for fp in clean if fp in existing_drafts}
            for fp in protected:
                new_drafts.pop(fp, None)

        merged_drafts = _merge_and_clean(new_drafts, existing_drafts, deletions)

        # Quick verify — skip on the last attempt (let outer verifier handle it)
        if inner_attempt < MAX_INNER_ITERATIONS - 1 and merged_drafts:
            errors = _quick_verify(merged_drafts)
            if not errors:
                break  # Tests pass — done
            # Feed errors back to the LLM for self-correction
            inner_iterations += 1
            error_text = "\n".join(errors)
            logger.info(
                "Inner-loop attempt %d/%d failed, feeding errors back to LLM",
                inner_attempt + 1, MAX_INNER_ITERATIONS,
            )
            # Build context for the fix: show the LLM its own output + errors
            messages.append(AIMessage(content=raw_content))
            messages.append(HumanMessage(
                content=(
                    f"## Test errors (inner-loop attempt {inner_attempt + 1})\n"
                    f"Your code has errors. Fix them and regenerate ALL files.\n\n"
                    f"```\n{error_text[:3000]}\n```"
                ),
            ))
            # Update existing_drafts so the next merge is correct
            existing_drafts = merged_drafts
        else:
            break

    # ── Post-loop: persist and return ───────────────────────────────────
    return _finalize_output(
        state, raw_content, new_drafts, existing_drafts,
        deletions, inner_iterations, user_message,
        merged_drafts=merged_drafts,
        llm_calls=llm_calls,
    )


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
