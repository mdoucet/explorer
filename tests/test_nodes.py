"""Unit tests for each node in the Scientific Loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from orchestrator.nodes import (
    _looks_like_filepath,
    _parse_code_blocks,
    _parse_plan_phases,
    _prepare_sandbox,
    _write_code_drafts,
    _write_plan_artifact,
    advance_phase,
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
        "plan_phases": [],
        "current_phase": 0,
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

    def test_stores_raw_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm_output = "No code blocks here, just text."
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        state = _base_state(plan="implement something")
        result = coder(state)

        assert result["code_drafts"] == {}
        assert result["coder_raw_response"] == llm_output


class TestLooksLikeFilepath:
    """Tests for the _looks_like_filepath heuristic."""

    def test_path_with_slash(self) -> None:
        assert _looks_like_filepath("src/physics/fluid.py") is True

    def test_root_level_python_file(self) -> None:
        assert _looks_like_filepath("conftest.py") is True

    def test_root_level_toml_file(self) -> None:
        assert _looks_like_filepath("pyproject.toml") is True

    def test_root_level_yaml_file(self) -> None:
        assert _looks_like_filepath("config.yaml") is True

    def test_language_tag_python(self) -> None:
        assert _looks_like_filepath("python") is False

    def test_language_tag_bash(self) -> None:
        assert _looks_like_filepath("bash") is False

    def test_language_tag_json(self) -> None:
        assert _looks_like_filepath("json") is False

    def test_empty_string(self) -> None:
        assert _looks_like_filepath("") is False

    def test_unknown_extension(self) -> None:
        assert _looks_like_filepath("file.xyz") is False


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

    def test_root_level_files(self) -> None:
        text = (
            "```pyproject.toml\n[project]\nname = 'foo'\n```\n"
            "```conftest.py\nimport sys\n```\n"
        )
        drafts = _parse_code_blocks(text)
        assert set(drafts.keys()) == {"pyproject.toml", "conftest.py"}

    def test_mixed_valid_and_language_tags(self) -> None:
        text = (
            "Here's some explanation:\n"
            "```python\nprint('ignored')\n```\n"
            "And the actual file:\n"
            "```src/solver.py\ndef solve(): pass\n```\n"
        )
        drafts = _parse_code_blocks(text)
        assert list(drafts.keys()) == ["src/solver.py"]
        assert "def solve" in drafts["src/solver.py"]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class TestVerifier:
    def test_passing_code_flat_layout(self) -> None:
        """Flat layout: package at root, tests import from package name."""
        state = _base_state(
            code_drafts={
                "mymath/__init__.py": "",
                "mymath/factorial.py": (
                    "def factorial(n: int) -> int:\n"
                    "    return 1 if n <= 1 else n * factorial(n - 1)\n"
                ),
                "tests/test_factorial.py": (
                    "from mymath.factorial import factorial\n"
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

    def test_passing_code_src_layout(self) -> None:
        """src/ layout: package under src/, tests import from package name."""
        state = _base_state(
            code_drafts={
                "src/mymath/__init__.py": "",
                "src/mymath/factorial.py": (
                    "def factorial(n: int) -> int:\n"
                    "    return 1 if n <= 1 else n * factorial(n - 1)\n"
                ),
                "tests/test_factorial.py": (
                    "from mymath.factorial import factorial\n"
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
                "mymath/__init__.py": "",
                "mymath/factorial.py": "def factorial(n): return -1\n",
                "tests/test_factorial.py": (
                    "from mymath.factorial import factorial\n"
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
        assert result["iteration_count"] == 1


# ---------------------------------------------------------------------------
# _prepare_sandbox
# ---------------------------------------------------------------------------

class TestPrepareSandbox:
    def test_adds_src_to_path_in_conftest(self, tmp_path: Path) -> None:
        """When code lives under src/, conftest should add src/ to sys.path."""
        drafts = {
            "src/pkg/__init__.py": "",
            "src/pkg/mod.py": "X = 1\n",
        }
        _prepare_sandbox(tmp_path, drafts)

        conftest = (tmp_path / "conftest.py").read_text()
        assert "src" in conftest

    def test_does_not_overwrite_coder_conftest(self, tmp_path: Path) -> None:
        """If the coder produces a conftest.py, _prepare_sandbox must not replace it."""
        original = "# coder-generated conftest\nimport sys\n"
        drafts = {
            "conftest.py": original,
            "pkg/__init__.py": "",
            "pkg/mod.py": "X = 1\n",
        }
        _prepare_sandbox(tmp_path, drafts)

        assert (tmp_path / "conftest.py").read_text() == original

    def test_flat_layout_conftest_includes_root(self, tmp_path: Path) -> None:
        """Flat layout: conftest should add the root directory."""
        drafts = {
            "pkg/__init__.py": "",
            "pkg/mod.py": "X = 1\n",
        }
        _prepare_sandbox(tmp_path, drafts)

        conftest = (tmp_path / "conftest.py").read_text()
        assert str(tmp_path) in conftest


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


# ---------------------------------------------------------------------------
# Plan phase parsing
# ---------------------------------------------------------------------------

class TestParsePlanPhases:
    """Tests for _parse_plan_phases."""

    def test_parses_multi_phase_plan(self) -> None:
        text = (
            "## Mathematical Specification\nSome math\n\n"
            "## Phase 1: Core solver\n"
            "Build the solver module.\n"
            "Files: solver.py, tests/test_solver.py\n\n"
            "## Phase 2: CLI\n"
            "Add a command-line interface.\n"
            "Files: cli.py, tests/test_cli.py\n"
        )
        phases = _parse_plan_phases(text)
        assert len(phases) == 2
        assert phases[0]["id"] == 1
        assert phases[0]["title"] == "Core solver"
        assert phases[0]["files"] == ["solver.py", "tests/test_solver.py"]
        assert phases[0]["status"] == "pending"
        assert phases[1]["id"] == 2
        assert phases[1]["title"] == "CLI"
        assert "command-line" in phases[1]["description"]

    def test_single_phase_plan(self) -> None:
        text = (
            "## Phase 1: Everything\n"
            "Do it all.\n"
            "Files: main.py\n"
        )
        phases = _parse_plan_phases(text)
        assert len(phases) == 1
        assert phases[0]["title"] == "Everything"

    def test_fallback_when_no_phase_headers(self) -> None:
        text = "Just a plain plan with no phase headers.\nBuild stuff."
        phases = _parse_plan_phases(text)
        assert len(phases) == 1
        assert phases[0]["title"] == "Implementation"
        assert "plain plan" in phases[0]["description"]

    def test_phase_without_files_line(self) -> None:
        text = "## Phase 1: Setup\nJust description, no files line.\n"
        phases = _parse_plan_phases(text)
        assert phases[0]["files"] == []


class TestWritePlanArtifact:
    """Tests for _write_plan_artifact."""

    def test_writes_plan_md(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        phases = [
            {"id": 1, "title": "Core", "description": "Build core.", "status": "completed", "files": ["core.py"]},
            {"id": 2, "title": "CLI", "description": "Build CLI.", "status": "pending", "files": ["cli.py"]},
        ]
        _write_plan_artifact(phases, 1)
        content = (tmp_path / "plan.md").read_text()
        assert "[x] Phase 1: Core" in content
        assert "[~] Phase 2: CLI" in content
        assert "core.py" in content


class TestPlannerPhases:
    """Tests for the planner node producing phased output."""

    def test_planner_returns_plan_phases(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        llm_output = (
            "## Phase 1: Solver\nBuild solver.\nFiles: solver.py\n\n"
            "## Phase 2: Tests\nAdd tests.\nFiles: tests/test_solver.py\n"
        )
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        state = _base_state()
        result = planner(state)

        assert len(result["plan_phases"]) == 2
        assert result["current_phase"] == 0
        assert "Build solver" in result["plan"]
        assert (tmp_path / "plan.md").exists()

    def test_planner_revision_updates_current_phase_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        existing_phases = [
            {"id": 1, "title": "Core", "description": "original", "status": "completed", "files": []},
            {"id": 2, "title": "CLI", "description": "original cli", "status": "pending", "files": []},
        ]
        revised_output = "## Phase 1: Revised CLI\nFixed the CLI.\nFiles: cli.py\n"
        fake = _FakeLLM(revised_output)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        state = _base_state(
            reflection="ImportError in cli.py",
            plan_phases=existing_phases,
            current_phase=1,
        )
        result = planner(state)

        # Phase 1 (Core) should be untouched
        assert result["plan_phases"][0]["description"] == "original"
        # Phase 2 (CLI) should be revised
        assert "Fixed the CLI" in result["plan_phases"][1]["description"]
        assert result["current_phase"] == 1


class TestAdvancePhase:
    """Tests for the advance_phase node."""

    def test_advances_to_next_phase(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        phases = [
            {"id": 1, "title": "Core", "description": "Build core.", "status": "pending", "files": []},
            {"id": 2, "title": "CLI", "description": "Build CLI.", "status": "pending", "files": []},
        ]
        state = _base_state(plan_phases=phases, current_phase=0)
        result = advance_phase(state)

        assert result["current_phase"] == 1
        assert result["plan_phases"][0]["status"] == "completed"
        assert result["plan_phases"][1]["status"] == "in-progress"
        assert "Build CLI" in result["plan"]
        assert result["reflection"] == ""  # cleared for new phase


class TestCoderPhaseContext:
    """Tests for coder receiving phase context."""

    def test_coder_includes_phase_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm_output = "```solver.py\nprint('hello')\n```\n"
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        phases = [
            {"id": 1, "title": "Core", "description": "Build core.", "status": "pending", "files": []},
            {"id": 2, "title": "CLI", "description": "Build CLI.", "status": "pending", "files": []},
        ]
        state = _base_state(plan="Build core.", plan_phases=phases, current_phase=0)
        result = coder(state)

        assert "Phase (1 of 2)" in result["_prompt_summary"]
        assert "Core" in result["_prompt_summary"]

    def test_coder_accumulates_drafts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm_output = "```cli.py\nprint('cli')\n```\n"
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        existing = {"solver.py": "print('solver')"}
        state = _base_state(plan="Build CLI.", code_drafts=existing)
        result = coder(state)

        # Both old and new files should be in code_drafts
        assert "solver.py" in result["code_drafts"]
        assert "cli.py" in result["code_drafts"]

    def test_coder_mentions_existing_files(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm_output = "```cli.py\nprint('cli')\n```\n"
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        existing = {"solver.py": "code"}
        state = _base_state(plan="Build CLI.", code_drafts=existing)
        result = coder(state)

        assert "solver.py" in result["_prompt_summary"]
        assert "Existing files" in result["_prompt_summary"]
