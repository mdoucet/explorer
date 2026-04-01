"""Sandbox tools for the tool-calling coder agent.

Provides :class:`CoderSandbox` (a managed temporary directory) and
:func:`make_sandbox_tools` which creates LangChain tools bound to a
sandbox instance.  The coder node uses these when the LLM supports
tool calling.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from langchain_core.tools import tool as langchain_tool

from ._verifier import _ensure_importable, _run_pytest

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 10  # max LLM round-trips in tool-calling mode


class CoderSandbox:
    """Temporary sandbox directory for the tool-calling coder.

    Files written via :meth:`write_file` land here.  :meth:`run_tests`
    executes pytest.  After the LLM finishes, :meth:`collect_drafts`
    reads everything back as a ``{path: content}`` dict.

    Parameters
    ----------
    initial_drafts : dict[str, str] | None
        Files from prior phases to seed the sandbox with.
    """

    def __init__(self, initial_drafts: dict[str, str] | None = None) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="explorer_sandbox_")
        self._root = Path(self._tmpdir).resolve()
        self._installed = False
        self._tracked_files: set[str] = set()

        if initial_drafts:
            for rel_path, content in initial_drafts.items():
                target = (self._root / rel_path).resolve()
                if not target.is_relative_to(self._root):
                    continue  # skip path traversal attempts
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
                self._tracked_files.add(rel_path)

    @property
    def root(self) -> Path:
        """Root directory of the sandbox."""
        return self._root

    def write_file(self, path: str, content: str) -> str:
        """Write *content* to *path* inside the sandbox.

        Returns a confirmation string or an error message for path
        traversal violations.
        """
        target = (self._root / path).resolve()
        if not target.is_relative_to(self._root):
            return f"Error: path traversal blocked for {path!r}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        self._tracked_files.add(path)
        self._installed = False  # new file may change project structure
        return f"Wrote {path} ({len(content)} bytes)"

    def read_file(self, path: str) -> str:
        """Read a file from the sandbox."""
        target = (self._root / path).resolve()
        if not target.is_relative_to(self._root):
            return f"Error: path traversal blocked for {path!r}"
        if not target.exists():
            return f"Error: {path} does not exist"
        try:
            return target.read_text()
        except UnicodeDecodeError:
            return f"Error: {path} is a binary file"

    def delete_file(self, path: str) -> str:
        """Delete a file from the sandbox."""
        target = (self._root / path).resolve()
        if not target.is_relative_to(self._root):
            return f"Error: path traversal blocked for {path!r}"
        if not target.exists():
            return f"Error: {path} does not exist"
        target.unlink()
        self._tracked_files.discard(path)
        return f"Deleted {path}"

    def list_files(self) -> str:
        """List all tracked files in the sandbox."""
        if not self._tracked_files:
            return "No files in the project."
        return "\n".join(sorted(self._tracked_files))

    def run_tests(self) -> str:
        """Run pytest in the sandbox.

        Ensures imports work (via editable install or conftest) before
        running tests.  Returns ``"All tests passed!"`` on success or
        the pytest output on failure (truncated to 4000 chars).
        """
        if not self._installed:
            _ensure_importable(self._root, {})
            self._installed = True
        errors = _run_pytest(self._root)
        if not errors:
            return "All tests passed!"
        return errors[0][:4000]

    def collect_drafts(self) -> dict[str, str]:
        """Read all tracked files back as a ``{path: content}`` dict."""
        drafts: dict[str, str] = {}
        for rel in sorted(self._tracked_files):
            target = self._root / rel
            if target.exists():
                try:
                    drafts[rel] = target.read_text()
                except UnicodeDecodeError:
                    continue
        return drafts

    def cleanup(self) -> None:
        """Remove the temporary sandbox directory."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)


def make_sandbox_tools(sandbox: CoderSandbox) -> list:
    """Create LangChain tools bound to *sandbox*.

    Returns a list of tools suitable for ``llm.bind_tools()``.
    """

    @langchain_tool
    def write_file(path: str, content: str) -> str:
        """Write or overwrite a file in the project.

        Args:
            path: Relative file path (e.g. 'my_package/solver.py').
            content: The complete file content to write.
        """
        return sandbox.write_file(path, content)

    @langchain_tool
    def read_file(path: str) -> str:
        """Read a file from the project.

        Args:
            path: Relative file path to read.
        """
        return sandbox.read_file(path)

    @langchain_tool
    def delete_file(path: str) -> str:
        """Delete a file from the project.

        Args:
            path: Relative file path to delete.
        """
        return sandbox.delete_file(path)

    @langchain_tool
    def run_tests() -> str:
        """Run pytest on the project to check for errors.

        Call this after writing all source files and tests.  If tests fail,
        read the output carefully, fix the files, and call run_tests again.
        """
        return sandbox.run_tests()

    @langchain_tool
    def list_files() -> str:
        """List all files currently in the project."""
        return sandbox.list_files()

    return [write_file, read_file, delete_file, run_tests, list_files]
