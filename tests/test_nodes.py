"""Unit tests for each node in the Scientific Loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from orchestrator.nodes import (
    _parse_code_blocks,
    _write_code_drafts,
    coder,
    planner,
    reflector,
    verifier,
)
from orchestrator.state import ScientificState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides: Any) -> ScientificState:
    base: ScientificState = {
        "task_description": "Implement a factorial function",
        "mathematical_constants": {},
        "plan": "",
        "code_drafts": {},
        "test_logs": [],
        "reflection": "",
        "iteration_count": 0,
        "ground_truth": [],
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


class _FakeLLM:
    """Mimics ChatOpenAI.invoke() returning a fixed message."""

    def __init__(self, content: str) -> None:
        self._msg = MagicMock()
        self._msg.content = content

    def invoke(self, _messages: Any) -> Any:
        return self._msg


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class TestPlanner:
    def test_returns_plan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeLLM("## Plan\nCompute $n!$ recursively.")
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        state = _base_state()
        result = planner(state)

        assert "plan" in result
        assert "Plan" in result["plan"]

    def test_includes_reflection_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeLLM("revised plan")
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        state = _base_state(reflection="Off-by-one error in loop")
        result = planner(state)
        assert result["plan"] == "revised plan"


# ---------------------------------------------------------------------------
# Coder
# ---------------------------------------------------------------------------

class TestCoder:
    def test_extracts_code_drafts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm_output = (
            "Here is the code:\n"
            "```src/math/factorial.py\n"
            "def factorial(n: int) -> int:\n"
            "    return 1 if n <= 1 else n * factorial(n - 1)\n"
            "```\n"
        )
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        state = _base_state(plan="implement factorial")
        result = coder(state)

        assert "src/math/factorial.py" in result["code_drafts"]
        assert "def factorial" in result["code_drafts"]["src/math/factorial.py"]


class TestParseCodeBlocks:
    def test_multiple_blocks(self) -> None:
        text = (
            "```src/a.py\nprint('a')\n```\n"
            "```src/b.py\nprint('b')\n```\n"
        )
        drafts = _parse_code_blocks(text)
        assert set(drafts.keys()) == {"src/a.py", "src/b.py"}

    def test_ignores_blocks_without_path(self) -> None:
        text = "```python\nprint('hello')\n```\n"
        assert _parse_code_blocks(text) == {}


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class TestVerifier:
    def test_passing_code_returns_empty_logs(self) -> None:
        state = _base_state(
            code_drafts={
                "src/math/factorial.py": (
                    "def factorial(n: int) -> int:\n"
                    "    return 1 if n <= 1 else n * factorial(n - 1)\n"
                ),
                "tests/test_factorial.py": (
                    "from src.math.factorial import factorial\n"
                    "def test_base():\n"
                    "    assert factorial(0) == 1\n"
                    "def test_five():\n"
                    "    assert factorial(5) == 120\n"
                ),
            },
        )
        result = verifier(state)
        assert result["test_logs"] == []
        assert result["iteration_count"] == 1

    def test_failing_code_captures_logs(self) -> None:
        state = _base_state(
            code_drafts={
                "src/math/factorial.py": "def factorial(n): return -1\n",
                "tests/test_factorial.py": (
                    "from src.math.factorial import factorial\n"
                    "def test_five():\n"
                    "    assert factorial(5) == 120\n"
                ),
            },
        )
        result = verifier(state)
        assert len(result["test_logs"]) > 0

    def test_no_code_drafts(self) -> None:
        state = _base_state(code_drafts={})
        result = verifier(state)
        assert result["test_logs"] == ["No code drafts to verify."]


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------

class TestReflector:
    def test_returns_reflection_and_findings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        responses = [
            "The factorial function always returns -1.",
            "- The function uses a hardcoded return value instead of recursion",
        ]
        idx = {"i": 0}

        class _SeqLLM:
            def invoke(self, _msgs: Any) -> Any:
                msg = MagicMock()
                msg.content = responses[idx["i"] % len(responses)]
                idx["i"] += 1
                return msg

        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: _SeqLLM())

        state = _base_state(test_logs=["AssertionError: assert -1 == 120"])
        result = reflector(state)

        assert "reflection" in result
        assert "-1" in result["reflection"]
        assert "ground_truth" in result
        assert len(result["ground_truth"]) >= 1


# ---------------------------------------------------------------------------
# _write_code_drafts
# ---------------------------------------------------------------------------

class TestWriteCodeDrafts:
    def test_writes_files_to_output_dir(self, tmp_path: Path) -> None:
        drafts = {
            "src/math/factorial.py": "def factorial(n): return 1\n",
            "tests/test_factorial.py": "def test_one(): assert True\n",
        }
        _write_code_drafts(drafts, str(tmp_path))

        assert (tmp_path / "src" / "math" / "factorial.py").read_text() == drafts["src/math/factorial.py"]
        assert (tmp_path / "tests" / "test_factorial.py").read_text() == drafts["tests/test_factorial.py"]

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        drafts = {"../escape.py": "import os\n"}
        with pytest.raises(ValueError, match="Path traversal blocked"):
            _write_code_drafts(drafts, str(tmp_path))

    def test_rejects_absolute_path(self, tmp_path: Path) -> None:
        drafts = {"/etc/passwd": "bad\n"}
        with pytest.raises(ValueError, match="Path traversal blocked"):
            _write_code_drafts(drafts, str(tmp_path))

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "hello.py"
        target.write_text("old content\n")
        _write_code_drafts({"hello.py": "new content\n"}, str(tmp_path))
        assert target.read_text() == "new content\n"


# ---------------------------------------------------------------------------
# Verifier – write mode
# ---------------------------------------------------------------------------

class TestVerifierWriteMode:
    def test_runs_pytest_in_output_dir(self, tmp_path: Path) -> None:
        """Verifier runs pytest in the output directory, not a temp dir."""
        # Pre-write code into the output directory
        (tmp_path / "src" / "math").mkdir(parents=True)
        (tmp_path / "src" / "math" / "factorial.py").write_text(
            "def factorial(n: int) -> int:\n"
            "    return 1 if n <= 1 else n * factorial(n - 1)\n"
        )
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_factorial.py").write_text(
            "import sys, pathlib\n"
            "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))\n"
            "from src.math.factorial import factorial\n"
            "def test_base():\n"
            "    assert factorial(0) == 1\n"
        )
        # Minimal conftest for the repo
        (tmp_path / "conftest.py").write_text(
            "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parent))\n"
        )

        state = _base_state(
            code_drafts={
                "src/math/factorial.py": (
                    "def factorial(n: int) -> int:\n"
                    "    return 1 if n <= 1 else n * factorial(n - 1)\n"
                ),
                "tests/test_factorial.py": (
                    "import sys, pathlib\n"
                    "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))\n"
                    "from src.math.factorial import factorial\n"
                    "def test_base():\n"
                    "    assert factorial(0) == 1\n"
                ),
            },
            output_dir=str(tmp_path),
        )
        result = verifier(state)
        assert result["test_logs"] == []
        assert result["iteration_count"] == 1
