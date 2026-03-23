"""Unit tests for each node in the Scientific Loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from orchestrator.nodes import (
    _check_duplicate_modules,
    _check_import_consistency,
    _check_syntax,
    _ensure_importable,
    _extract_signatures,
    _looks_like_filepath,
    _normalize_pytest_output,
    _parse_code_blocks,
    _parse_plan_phases,
    _prepare_sandbox,
    _warn_stale_files,
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
        # Pre-write code into the output directory (flat layout)
        (tmp_path / "mymath").mkdir()
        (tmp_path / "mymath" / "__init__.py").write_text("")
        (tmp_path / "mymath" / "factorial.py").write_text(
            "def factorial(n: int) -> int:\n"
            "    return 1 if n <= 1 else n * factorial(n - 1)\n"
        )
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_factorial.py").write_text(
            "from mymath.factorial import factorial\n"
            "def test_base():\n"
            "    assert factorial(0) == 1\n"
        )
        # Minimal conftest for the repo
        (tmp_path / "conftest.py").write_text(
            "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parent))\n"
        )

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
                ),
            },
            output_dir=str(tmp_path),
        )
        result = verifier(state)
        assert result["test_logs"] == []
        assert result["iteration_count"] == 1

    def test_src_layout_works_in_write_mode(self, tmp_path: Path) -> None:
        """Write mode with src/ layout: _ensure_importable generates conftest."""
        # Prepare src layout on disk
        (tmp_path / "src" / "mymath").mkdir(parents=True)
        (tmp_path / "src" / "mymath" / "__init__.py").write_text("")
        (tmp_path / "src" / "mymath" / "factorial.py").write_text(
            "def factorial(n: int) -> int:\n"
            "    return 1 if n <= 1 else n * factorial(n - 1)\n"
        )
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_factorial.py").write_text(
            "from mymath.factorial import factorial\n"
            "def test_base():\n"
            "    assert factorial(0) == 1\n"
        )

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
                ),
            },
            output_dir=str(tmp_path),
        )
        result = verifier(state)
        assert result["test_logs"] == []


# ---------------------------------------------------------------------------
# _ensure_importable
# ---------------------------------------------------------------------------

class TestEnsureImportable:
    def test_generates_conftest_for_src_layout(self, tmp_path: Path) -> None:
        """When pip install fails or no pyproject.toml, conftest is generated."""
        (tmp_path / "src" / "pkg").mkdir(parents=True)
        (tmp_path / "src" / "pkg" / "__init__.py").write_text("")
        (tmp_path / "src" / "pkg" / "mod.py").write_text("X = 1\n")

        _ensure_importable(tmp_path, {
            "src/pkg/__init__.py": "",
            "src/pkg/mod.py": "X = 1\n",
        })

        conftest = (tmp_path / "conftest.py").read_text()
        assert "sys.path" in conftest
        assert "src" in conftest

    def test_does_not_overwrite_existing_conftest(self, tmp_path: Path) -> None:
        """If a conftest.py already exists, _ensure_importable leaves it alone."""
        (tmp_path / "conftest.py").write_text("# custom conftest\n")
        (tmp_path / "src" / "pkg").mkdir(parents=True)
        (tmp_path / "src" / "pkg" / "__init__.py").write_text("")

        _ensure_importable(tmp_path, {"src/pkg/__init__.py": ""})

        assert (tmp_path / "conftest.py").read_text() == "# custom conftest\n"

    def test_flat_layout_includes_root(self, tmp_path: Path) -> None:
        """Flat layout: conftest includes the project root."""
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "__init__.py").write_text("")

        _ensure_importable(tmp_path, {"pkg/__init__.py": ""})

        conftest = (tmp_path / "conftest.py").read_text()
        assert str(tmp_path) in conftest

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

    def test_planner_ignores_reflection_context(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """The planner always generates a fresh plan — it no longer handles
        revision loops (the reflector routes directly to the coder)."""
        monkeypatch.chdir(tmp_path)
        plan_output = "## Phase 1: Core\nBuild the core.\nFiles: core.py\n"
        fake = _FakeLLM(plan_output)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        state = _base_state(
            reflection="ImportError in cli.py",
        )
        result = planner(state)

        # Reflection should NOT appear in the planner prompt
        assert "ImportError" not in result["_prompt_summary"]
        assert result["current_phase"] == 0


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

    def test_coder_includes_signatures_of_existing_files(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        llm_output = "```cli.py\nprint('cli')\n```\n"
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        existing = {
            "solver.py": "def solve_square_well(n: int) -> float:\n    pass\n",
        }
        state = _base_state(plan="Build CLI.", code_drafts=existing)
        result = coder(state)

        # Prompt should contain the actual function signature
        assert "solve_square_well" in result["_prompt_summary"]
        assert "exact names" in result["_prompt_summary"].lower()


# ---------------------------------------------------------------------------
# _extract_signatures
# ---------------------------------------------------------------------------

class TestExtractSignatures:
    def test_extracts_functions(self) -> None:
        source = (
            "def solve(n: int, L: float) -> float:\n"
            "    pass\n\n"
            "def helper():\n"
            "    pass\n"
        )
        sigs = _extract_signatures(source)
        assert len(sigs) == 2
        assert "solve" in sigs[0]
        assert "helper" in sigs[1]

    def test_extracts_classes_and_methods(self) -> None:
        source = (
            "class Solver:\n"
            "    def run(self, x: int) -> None:\n"
            "        pass\n"
        )
        sigs = _extract_signatures(source)
        assert any("class Solver" in s for s in sigs)
        assert any("run" in s for s in sigs)

    def test_returns_empty_for_invalid_syntax(self) -> None:
        assert _extract_signatures("def !!!broken") == []

    def test_returns_empty_for_non_python(self) -> None:
        assert _extract_signatures("[project]\nname = 'foo'\n") == []


# ---------------------------------------------------------------------------
# _check_import_consistency
# ---------------------------------------------------------------------------

class TestCheckImportConsistency:
    def test_consistent_imports_pass(self) -> None:
        drafts = {
            "solver.py": "def solve(n: int) -> float:\n    pass\n",
            "tests/test_solver.py": (
                "from solver import solve\n"
                "def test_it():\n    assert solve(1) == 1.0\n"
            ),
        }
        assert _check_import_consistency(drafts) == []

    def test_mismatched_import_detected(self) -> None:
        drafts = {
            "solver.py": "def solve_square_well(n: int) -> float:\n    pass\n",
            "tests/test_solver.py": (
                "from solver import solve_eigenvalue\n"
                "def test_it():\n    assert solve_eigenvalue(1) == 1.0\n"
            ),
        }
        errors = _check_import_consistency(drafts)
        assert len(errors) == 1
        assert "solve_eigenvalue" in errors[0]
        assert "solve_square_well" in errors[0]

    def test_ignores_unknown_modules(self) -> None:
        """Imports from external packages should not trigger errors."""
        drafts = {
            "solver.py": "def solve() -> None:\n    pass\n",
            "tests/test_solver.py": (
                "import numpy as np\n"
                "from solver import solve\n"
                "def test_it():\n    solve()\n"
            ),
        }
        assert _check_import_consistency(drafts) == []

    def test_handles_src_layout(self) -> None:
        drafts = {
            "src/pkg/solver.py": "def solve() -> None:\n    pass\n",
            "tests/test_solver.py": (
                "from pkg.solver import solve\n"
                "def test_it():\n    solve()\n"
            ),
        }
        assert _check_import_consistency(drafts) == []

    def test_class_imports_pass(self) -> None:
        drafts = {
            "engine.py": "class Engine:\n    pass\n",
            "tests/test_engine.py": (
                "from engine import Engine\n"
                "def test_it():\n    Engine()\n"
            ),
        }
        assert _check_import_consistency(drafts) == []

    def test_top_level_assignment_counted(self) -> None:
        drafts = {
            "constants.py": "PI = 3.14159\n",
            "tests/test_constants.py": (
                "from constants import PI\n"
                "def test_pi():\n    assert PI > 3\n"
            ),
        }
        assert _check_import_consistency(drafts) == []

    def test_annotated_assignment_counted(self) -> None:
        """Typed assignments like `X: Final[float] = 1.0` must be recognized."""
        drafts = {
            "constants.py": (
                "from typing import Final\n"
                "HBAR: Final[float] = 1.0545718e-34\n"
                "PI: Final[float] = 3.14159\n"
            ),
            "tests/test_constants.py": (
                "from constants import HBAR, PI\n"
                "def test_hbar():\n    assert HBAR > 0\n"
            ),
        }
        assert _check_import_consistency(drafts) == []

    def test_detects_src_prefix_import(self) -> None:
        """Imports like `from src.pkg.mod import X` should be flagged."""
        drafts = {
            "src/square_well/solver.py": "def find_bound_states() -> list:\n    pass\n",
            "tests/test_solver.py": (
                "from src.square_well.solver import find_bound_states\n"
                "def test_it():\n    find_bound_states()\n"
            ),
        }
        errors = _check_import_consistency(drafts)
        assert len(errors) == 1
        assert "src" in errors[0]
        assert "not a package" in errors[0]
        assert "square_well.solver" in errors[0]

class TestAdvancePhaseGroundTruth:
    def test_clears_ground_truth_on_advance(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        phases = [
            {"id": 1, "title": "Core", "description": "Build core.", "status": "pending", "files": []},
            {"id": 2, "title": "CLI", "description": "Build CLI.", "status": "pending", "files": []},
        ]
        state = _base_state(
            plan_phases=phases,
            current_phase=0,
            ground_truth=["- stale finding from phase 1"],
        )
        result = advance_phase(state)

        assert result["ground_truth"] == []


# ---------------------------------------------------------------------------
# _check_duplicate_modules
# ---------------------------------------------------------------------------

class TestCheckDuplicateModules:
    def test_no_duplicates_flat_layout(self) -> None:
        drafts = {
            "my_pkg/__init__.py": "",
            "my_pkg/solver.py": "def solve(): pass\n",
            "tests/test_solver.py": "def test_it(): pass\n",
        }
        assert _check_duplicate_modules(drafts) == []

    def test_no_duplicates_src_layout(self) -> None:
        drafts = {
            "src/my_pkg/__init__.py": "",
            "src/my_pkg/solver.py": "def solve(): pass\n",
            "tests/test_solver.py": "def test_it(): pass\n",
        }
        assert _check_duplicate_modules(drafts) == []

    def test_detects_duplicate_layout(self) -> None:
        drafts = {
            "square_well/__init__.py": "",
            "square_well/solver.py": "def solve(): pass\n",
            "src/square_well/__init__.py": "",
            "src/square_well/solver.py": "def solve(): pass\n",
            "tests/test_solver.py": "def test_it(): pass\n",
        }
        errors = _check_duplicate_modules(drafts)
        assert len(errors) == 1
        assert "square_well" in errors[0]
        assert "Duplicate layout" in errors[0]

    def test_ignores_tests_directory(self) -> None:
        """tests/ should not be flagged as a flat-layout package."""
        drafts = {
            "src/my_pkg/__init__.py": "",
            "tests/test_solver.py": "def test_it(): pass\n",
        }
        assert _check_duplicate_modules(drafts) == []


# ---------------------------------------------------------------------------
# Coder receives error context
# ---------------------------------------------------------------------------

class TestCoderErrorContext:
    """The coder should include reflection and test_logs in its prompt when
    they are non-empty (i.e. after a failed verify→reflect cycle)."""

    def test_includes_reflection_in_prompt(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        llm_output = "```pkg/mod.py\nx = 1\n```\n"
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        state = _base_state(
            plan="implement constants",
            reflection="Define HBAR at module level, not inside get_constants().",
        )
        result = coder(state)

        prompt_sent = result["_prompt_summary"]
        assert "Previous error analysis" in prompt_sent
        assert "HBAR" in prompt_sent

    def test_includes_test_logs_in_prompt(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        llm_output = "```pkg/mod.py\nx = 1\n```\n"
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        state = _base_state(
            plan="fix imports",
            test_logs=["Import mismatch: tests/test_c.py imports 'HBAR'"],
        )
        result = coder(state)

        prompt_sent = result["_prompt_summary"]
        assert "Test failures to fix" in prompt_sent
        assert "Import mismatch" in prompt_sent

    def test_no_error_context_on_first_iteration(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        llm_output = "```pkg/mod.py\nx = 1\n```\n"
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        state = _base_state(plan="implement something")
        result = coder(state)

        prompt_sent = result["_prompt_summary"]
        assert "Previous error analysis" not in prompt_sent
        assert "Test failures to fix" not in prompt_sent

    def test_critical_escalation_after_3_repeats(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        llm_output = "```pkg/mod.py\nx = 1\n```\n"
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        state = _base_state(
            plan="fix imports",
            reflection="Constants must be at module level.",
            _error_repeat_count=4,
        )
        result = coder(state)

        prompt_sent = result["_prompt_summary"]
        assert "CRITICAL" in prompt_sent
        assert "4 consecutive iterations" in prompt_sent

    def test_revision_protects_existing_test_files(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On revision iterations with phase_error_count >= 2,
        existing test files must not be overwritten even if the coder emits them."""
        # The coder emits both solver.py and test_solver.py
        llm_output = (
            "```solver.py\ndef solve(): return 42\n```\n\n"
            "```tests/test_solver.py\nassert False  # rewritten tests\n```\n"
        )
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        original_test = "from solver import solve\ndef test_it(): assert solve() == 42\n"
        state = _base_state(
            plan="fix solver",
            reflection="Return value is wrong.",
            _phase_error_count=2,
            code_drafts={
                "solver.py": "def solve(): return -1\n",
                "tests/test_solver.py": original_test,
            },
        )
        result = coder(state)

        # Implementation should be updated
        assert "return 42" in result["code_drafts"]["solver.py"]
        # Test file should be PROTECTED (original preserved)
        assert result["code_drafts"]["tests/test_solver.py"] == original_test

    def test_first_revision_allows_test_fixes(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On the first revision (phase_error_count < 2), test files should
        NOT be protected — the coder needs one chance to fix bad tests."""
        rewritten_test = "from solver import solve\ndef test_it(): assert solve() == 42\n"
        llm_output = (
            "```solver.py\ndef solve(): return 42\n```\n\n"
            f"```tests/test_solver.py\n{rewritten_test}```\n"
        )
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        state = _base_state(
            plan="fix solver",
            reflection="Tests are wrong — assert len==0 but always >=1 bound state.",
            _phase_error_count=1,
            code_drafts={
                "solver.py": "def solve(): return -1\n",
                "tests/test_solver.py": "assert False  # wrong tests\n",
            },
        )
        result = coder(state)

        # Test file should be UPDATED (not protected on first revision)
        assert result["code_drafts"]["tests/test_solver.py"] == rewritten_test

    def test_first_iteration_allows_test_files(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On the first iteration (no reflection), test files should be kept."""
        llm_output = (
            "```solver.py\ndef solve(): return 42\n```\n\n"
            "```tests/test_solver.py\nfrom solver import solve\ndef test_it(): assert solve() == 42\n```\n"
        )
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        state = _base_state(plan="implement solver")
        result = coder(state)

        assert "tests/test_solver.py" in result["code_drafts"]
        assert "assert solve() == 42" in result["code_drafts"]["tests/test_solver.py"]


# ---------------------------------------------------------------------------
# Stuck-loop detection in verifier
# ---------------------------------------------------------------------------

class TestStuckLoopDetection:
    """Verifier tracks consecutive identical error fingerprints."""

    def test_first_failure_sets_count_to_one(self) -> None:
        state = _base_state(
            code_drafts={
                "pkg/__init__.py": "",
                "pkg/mod.py": "def solve(): pass\n",
                "tests/test_mod.py": (
                    "from pkg.mod import MISSING\n"
                    "def test_it(): pass\n"
                ),
            },
        )
        result = verifier(state)

        assert result["_error_repeat_count"] == 1
        assert result["_prev_error_fingerprint"] != ""

    def test_same_error_increments_count(self) -> None:
        drafts = {
            "pkg/__init__.py": "",
            "pkg/mod.py": "def solve(): pass\n",
            "tests/test_mod.py": (
                "from pkg.mod import MISSING\n"
                "def test_it(): pass\n"
            ),
        }
        # Run once to get the fingerprint
        state1 = _base_state(code_drafts=drafts)
        r1 = verifier(state1)

        # Run again with the same error and prior fingerprint
        state2 = _base_state(
            code_drafts=drafts,
            _prev_error_fingerprint=r1["_prev_error_fingerprint"],
            _error_repeat_count=r1["_error_repeat_count"],
        )
        r2 = verifier(state2)

        assert r2["_error_repeat_count"] == 2

    def test_different_error_resets_count(self) -> None:
        drafts_a = {
            "pkg/__init__.py": "",
            "pkg/mod.py": "def solve(): pass\n",
            "tests/test_mod.py": (
                "from pkg.mod import MISSING_A\n"
                "def test_it(): pass\n"
            ),
        }
        drafts_b = {
            "pkg/__init__.py": "",
            "pkg/mod.py": "def solve(): pass\n",
            "tests/test_mod.py": (
                "from pkg.mod import MISSING_B\n"
                "def test_it(): pass\n"
            ),
        }
        r1 = verifier(_base_state(code_drafts=drafts_a))

        state2 = _base_state(
            code_drafts=drafts_b,
            _prev_error_fingerprint=r1["_prev_error_fingerprint"],
            _error_repeat_count=r1["_error_repeat_count"],
        )
        r2 = verifier(state2)

        assert r2["_error_repeat_count"] == 1  # reset, not 2

    def test_passing_tests_reset_count(self) -> None:
        state = _base_state(
            code_drafts={
                "pkg/__init__.py": "",
                "pkg/mod.py": "def solve(): pass\n",
                "tests/test_mod.py": (
                    "from pkg.mod import solve\n"
                    "def test_it():\n"
                    "    assert solve() is None\n"
                ),
            },
            _prev_error_fingerprint="old-fp",
            _error_repeat_count=5,
        )
        result = verifier(state)

        assert result["_error_repeat_count"] == 0
        assert result["_prev_error_fingerprint"] == ""

    def test_phase_error_count_increments_on_failure(self) -> None:
        state = _base_state(
            code_drafts={
                "pkg/__init__.py": "",
                "pkg/mod.py": "def solve(): pass\n",
                "tests/test_mod.py": (
                    "from pkg.mod import MISSING\n"
                    "def test_it(): pass\n"
                ),
            },
            _phase_error_count=3,
        )
        result = verifier(state)
        assert result["_phase_error_count"] == 4

    def test_phase_error_count_unchanged_on_pass(self) -> None:
        state = _base_state(
            code_drafts={
                "pkg/__init__.py": "",
                "pkg/mod.py": "def solve(): pass\n",
                "tests/test_mod.py": (
                    "from pkg.mod import solve\n"
                    "def test_it():\n"
                    "    assert solve() is None\n"
                ),
            },
            _phase_error_count=2,
        )
        result = verifier(state)
        assert result["_phase_error_count"] == 2


# ---------------------------------------------------------------------------
# Ground truth deduplication
# ---------------------------------------------------------------------------

class TestGroundTruthDedup:
    """Reflector should not add duplicate findings to ground_truth."""

    def test_no_duplicate_findings(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        existing_finding = "- Define constants at module level"
        responses = [
            "The constant is missing from the namespace.",
            # LLM returns the same finding that already exists
            existing_finding,
        ]
        idx = {"i": 0}

        class _SeqLLM:
            def invoke(self, _msgs: Any) -> Any:
                msg = MagicMock()
                msg.content = responses[idx["i"] % len(responses)]
                idx["i"] += 1
                return msg

        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: _SeqLLM())

        state = _base_state(
            test_logs=["ImportError: cannot import HBAR"],
            ground_truth=[existing_finding],
        )
        result = reflector(state)

        # The finding should appear exactly once, not twice
        assert result["ground_truth"].count(existing_finding) == 1

    def test_new_findings_still_added(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        responses = [
            "Analysis of the error.",
            "- Brand new finding\n- Another finding",
        ]
        idx = {"i": 0}

        class _SeqLLM:
            def invoke(self, _msgs: Any) -> Any:
                msg = MagicMock()
                msg.content = responses[idx["i"] % len(responses)]
                idx["i"] += 1
                return msg

        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: _SeqLLM())

        state = _base_state(
            test_logs=["Some error"],
            ground_truth=["- Existing finding"],
        )
        result = reflector(state)

        assert "- Existing finding" in result["ground_truth"]
        assert "- Brand new finding" in result["ground_truth"]
        assert "- Another finding" in result["ground_truth"]
        assert len(result["ground_truth"]) == 3

    def test_existing_findings_passed_to_llm(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The findings LLM call should receive existing findings so it can
        avoid repeating them."""
        captured_msgs: list[Any] = []
        responses = ["Analysis.", "NONE"]
        idx = {"i": 0}

        class _CaptureLLM:
            def invoke(self, msgs: Any) -> Any:
                captured_msgs.append(msgs)
                msg = MagicMock()
                msg.content = responses[idx["i"] % len(responses)]
                idx["i"] += 1
                return msg

        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: _CaptureLLM())

        state = _base_state(
            test_logs=["assert 1 == 0"],
            ground_truth=["- Solver returns wrong count"],
        )
        reflector(state)

        # Second LLM call is the findings extraction
        assert len(captured_msgs) == 2
        findings_user_msg = captured_msgs[1][1].content
        assert "Existing findings" in findings_user_msg
        assert "Solver returns wrong count" in findings_user_msg


# ---------------------------------------------------------------------------
# Syntax pre-check
# ---------------------------------------------------------------------------


class TestCheckSyntax:
    """_check_syntax should catch syntax errors before pytest runs."""

    def test_valid_code_returns_empty(self) -> None:
        drafts = {"solver.py": "def solve():\n    return 42\n"}
        assert _check_syntax(drafts) == []

    def test_detects_syntax_error(self) -> None:
        drafts = {
            "solver.py": "def solve():\nreturn 42\n",  # IndentationError
        }
        errors = _check_syntax(drafts)
        assert len(errors) == 1
        assert "SyntaxError" in errors[0]
        assert "solver.py" in errors[0]

    def test_multiple_files_multiple_errors(self) -> None:
        drafts = {
            "good.py": "x = 1\n",
            "bad1.py": "def f(\n",
            "bad2.py": "class:\n",
        }
        errors = _check_syntax(drafts)
        assert len(errors) == 2
        combined = "\n".join(errors)
        assert "bad1.py" in combined
        assert "bad2.py" in combined

    def test_skips_non_python_files(self) -> None:
        drafts = {
            "README.md": "# not python {{{",
            "solver.py": "x = 1\n",
        }
        assert _check_syntax(drafts) == []

    def test_unindented_docstring_detected(self) -> None:
        """Reproduce the nemotron bug: method body not indented."""
        drafts = {
            "tests/test_solver.py": (
                "def test_something():\n"
                '"""This docstring is not indented."""\n'
                "    pass\n"
            ),
        }
        errors = _check_syntax(drafts)
        assert len(errors) == 1
        assert "IndentationError" in errors[0] or "SyntaxError" in errors[0]


# ---------------------------------------------------------------------------
# Stale file cleanup
# ---------------------------------------------------------------------------


class TestWarnStaleFiles:
    """_warn_stale_files should remove .py files not in code_drafts."""

    def test_removes_stale_file(self, tmp_path: Path) -> None:
        # Create a stale file from a "previous run"
        stale = tmp_path / "old_module.py"
        stale.write_text("# stale\n")
        # Current drafts don't include old_module.py
        drafts = {"solver.py": "x = 1\n"}
        _warn_stale_files(tmp_path, drafts)
        assert not stale.exists()

    def test_keeps_current_files(self, tmp_path: Path) -> None:
        current = tmp_path / "solver.py"
        current.write_text("x = 1\n")
        drafts = {"solver.py": "x = 1\n"}
        _warn_stale_files(tmp_path, drafts)
        assert current.exists()

    def test_keeps_init_files(self, tmp_path: Path) -> None:
        """__init__.py files should never be removed."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        init = pkg / "__init__.py"
        init.write_text("")
        drafts = {"solver.py": "x = 1\n"}
        _warn_stale_files(tmp_path, drafts)
        assert init.exists()

    def test_removes_stale_in_subdirectory(self, tmp_path: Path) -> None:
        sub = tmp_path / "pkg"
        sub.mkdir()
        stale = sub / "old.py"
        stale.write_text("# old\n")
        drafts = {"pkg/new.py": "x = 1\n"}
        _warn_stale_files(tmp_path, drafts)
        assert not stale.exists()


# ---------------------------------------------------------------------------
# Planner replanning
# ---------------------------------------------------------------------------

class TestPlannerReplan:
    """When plan_phases exist and reflection is set, the planner should revise
    only the current phase instead of creating a fresh plan from scratch."""

    def test_replan_updates_current_phase_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        revised = "## Phase 1: Solver v2\nUse brentq instead of matrix diag.\nFiles: solver.py\n"
        fake = _FakeLLM(revised)
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        phases = [
            {"id": 1, "title": "Solver", "description": "Build matrix solver.", "status": "pending", "files": ["solver.py"]},
            {"id": 2, "title": "CLI", "description": "Build CLI.", "status": "pending", "files": ["cli.py"]},
        ]
        state = _base_state(
            plan_phases=phases,
            current_phase=0,
            reflection="Matrix approach has inherent discretisation error.",
            test_logs=["FAILED test_solver.py::test_energy - AssertionError"],
        )
        result = planner(state)

        # Phase 2 should be untouched
        assert len(result["plan_phases"]) == 2
        assert result["plan_phases"][1]["title"] == "CLI"
        assert result["plan_phases"][1]["description"] == "Build CLI."
        # Phase 1 should be updated
        assert "brentq" in result["plan_phases"][0]["description"]
        assert result["plan_phases"][0]["title"] == "Solver v2"
        # Error repeat count should be reset
        assert result["_error_repeat_count"] == 0

    def test_replan_includes_error_context(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        fake = _FakeLLM("## Phase 1: Fix\nNew approach.\nFiles: solver.py\n")
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        phases = [
            {"id": 1, "title": "Solver", "description": "Build solver.", "status": "pending", "files": []},
        ]
        state = _base_state(
            plan_phases=phases,
            current_phase=0,
            reflection="Off-by-one in bracket computation.",
            test_logs=["FAILED test_solver - assert 1.73 != 1.71"],
        )
        result = planner(state)

        # The prompt should contain the error analysis and test logs
        assert "REPLAN REQUEST" in result["_prompt_summary"]
        assert "Off-by-one" in result["_prompt_summary"]
        assert "assert 1.73 != 1.71" in result["_prompt_summary"]

    def test_initial_plan_not_treated_as_replan(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """When plan_phases is empty (initial call), even with reflection set,
        the planner should produce a fresh plan."""
        monkeypatch.chdir(tmp_path)
        fake = _FakeLLM("## Phase 1: Core\nBuild core.\nFiles: core.py\n")
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        state = _base_state(reflection="Some stale reflection from a previous bug")
        result = planner(state)

        assert result["current_phase"] == 0
        assert "REPLAN" not in result.get("_prompt_summary", "")

    def test_replan_single_phase(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Replan should work even if there's only one phase."""
        monkeypatch.chdir(tmp_path)
        fake = _FakeLLM("## Phase 1: Revised\nNew approach.\nFiles: solver.py\n")
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        phases = [
            {"id": 1, "title": "Solver", "description": "Build.", "status": "pending", "files": []},
        ]
        state = _base_state(
            plan_phases=phases,
            current_phase=0,
            reflection="Error detected.",
        )
        result = planner(state)

        assert len(result["plan_phases"]) == 1
        assert "New approach" in result["plan_phases"][0]["description"]
        assert result["_replan_count"] == 1

    def test_replan_increments_replan_count(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        fake = _FakeLLM("## Phase 1: Try3\nThird try.\nFiles: solver.py\n")
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

        phases = [
            {"id": 1, "title": "Solver", "description": "Build.", "status": "pending", "files": []},
        ]
        state = _base_state(
            plan_phases=phases,
            current_phase=0,
            reflection="Still broken.",
            _replan_count=1,
        )
        result = planner(state)
        assert result["_replan_count"] == 2


# ---------------------------------------------------------------------------
# After-reflector routing
# ---------------------------------------------------------------------------

class TestAfterReflector:
    """The _after_reflector routing should send to planner when total phase
    failures hit the threshold, and to coder otherwise."""

    def test_routes_to_coder_below_threshold(self) -> None:
        from src.cli import _after_reflector

        state = _base_state(_phase_error_count=2)
        assert _after_reflector(state) == "coder"

    def test_routes_to_replan_at_threshold(self) -> None:
        from src.cli import _after_reflector, REPLAN_THRESHOLD

        state = _base_state(_phase_error_count=REPLAN_THRESHOLD)
        assert _after_reflector(state) == "replan"

    def test_routes_to_replan_above_threshold(self) -> None:
        from src.cli import _after_reflector, REPLAN_THRESHOLD

        state = _base_state(_phase_error_count=REPLAN_THRESHOLD + 2)
        assert _after_reflector(state) == "replan"

    def test_routes_to_coder_when_no_error_count(self) -> None:
        from src.cli import _after_reflector

        state = _base_state()
        assert _after_reflector(state) == "coder"

    def test_routes_to_coder_when_replan_exhausted(self) -> None:
        """After MAX_REPLANS, even if error threshold is hit, route to coder."""
        from src.cli import _after_reflector, MAX_REPLANS, REPLAN_THRESHOLD

        state = _base_state(
            _phase_error_count=REPLAN_THRESHOLD,
            _replan_count=MAX_REPLANS,
        )
        assert _after_reflector(state) == "coder"

    def test_oscillating_errors_still_trigger_replan(self) -> None:
        """When errors are different each iteration (repeat count stays at 1),
        but phase error count accumulates, replan should still trigger."""
        from src.cli import _after_reflector, REPLAN_THRESHOLD

        # Simulates: 3 failures with _error_repeat_count=1 (all different)
        # but _phase_error_count=3 (total failures monotonically counted)
        state = _base_state(
            _error_repeat_count=1,
            _phase_error_count=REPLAN_THRESHOLD,
        )
        assert _after_reflector(state) == "replan"


# ---------------------------------------------------------------------------
# Skills MUST-USE directive
# ---------------------------------------------------------------------------

class TestSkillsMustUseDirective:
    """format_skills_context should add a mandatory directive when the skill
    contains a recipe section."""

    def test_recipe_skill_gets_must_use(self, tmp_path: Path) -> None:
        from orchestrator.skills import Skill, format_skills_context

        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("Some intro.\n\n## Recipe: Transcendental equations\nUse brentq.\n")
        skill = Skill(
            name="numerical-optimization",
            description="Optimization recipes",
            path=skill_file,
        )
        result = format_skills_context([skill])
        assert "MANDATORY" in result
        assert "MUST follow" in result

    def test_non_recipe_skill_no_must_use(self, tmp_path: Path) -> None:
        from orchestrator.skills import Skill, format_skills_context

        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("Use pytest.\nArrange-Act-Assert pattern.\n")
        skill = Skill(
            name="testing",
            description="Testing best practices",
            path=skill_file,
        )
        result = format_skills_context([skill])
        assert "MANDATORY" not in result
        assert "Skill: testing" in result

    def test_empty_skills_returns_empty(self) -> None:
        from orchestrator.skills import format_skills_context

        assert format_skills_context([]) == ""


# ---------------------------------------------------------------------------
# Pytest output normalization
# ---------------------------------------------------------------------------

class TestNormalizePytestOutput:
    """_normalize_pytest_output should strip variable timing info."""

    def test_strips_timing(self) -> None:
        raw = "========================= 1 failed, 3 passed in 0.41s =========================="
        result = _normalize_pytest_output(raw)
        assert "0.41s" not in result
        assert "1 failed, 3 passed" in result

    def test_strips_different_timings(self) -> None:
        a = _normalize_pytest_output("=== 1 failed in 0.41s ===")
        b = _normalize_pytest_output("=== 1 failed in 1.23s ===")
        assert a == b

    def test_preserves_rest_of_output(self) -> None:
        raw = "FAILED tests/test_solver.py::test_x - assert 3 == 0\n"
        assert _normalize_pytest_output(raw) == raw

    def test_handles_no_timing(self) -> None:
        raw = "SyntaxError in solver.py line 5: invalid syntax"
        assert _normalize_pytest_output(raw) == raw
