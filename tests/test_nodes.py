"""Unit tests for each node in the Scientific Loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from orchestrator.nodes import (
    MAX_INNER_ITERATIONS,
    _REPLAN_ERROR_REPEAT_THRESHOLD,
    _check_pyproject_toml,
    _check_shadowed_packages,
    _check_syntax,
    _ensure_importable,
    _extract_action,
    _extract_findings,
    _format_code_listing,
    _llm_triage,
    _merge_and_clean,
    _pick_replan_phase,
    _extract_signatures,
    _looks_like_filepath,
    _normalize_pytest_output,
    _parse_code_blocks,
    _parse_plan_phases,
    _parse_unfenced_blocks,
    _prepare_sandbox,
    _quick_verify,
    _resolve_duplicate_layouts,
    _warn_stale_files,
    _write_code_drafts,
    _write_plan_artifact,
    advance_phase,
    auto_reflect,
    coder,
    planner,
    reflector,
    verifier,
)
from orchestrator.state import ScientificState
from orchestrator.transcript import make_entry


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
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state()
        result = planner(state)

        assert "plan" in result
        assert "Plan" in result["plan"]

    def test_appends_to_transcript(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeLLM("## Phase 1: Core\nBuild it.\nFiles: core.py\n")
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state()
        result = planner(state)

        assert len(result["transcript"]) == 1
        assert result["transcript"][0]["node"] == "planner"
        assert result["transcript"][0]["role"] == "ai"

    def test_includes_reflection_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeLLM("revised plan")
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

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
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(plan="implement factorial")
        result = coder(state)

        assert "src/math/factorial.py" in result["code_drafts"]
        assert "def factorial" in result["code_drafts"]["src/math/factorial.py"]

    def test_stores_raw_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm_output = "No code blocks here, just text."
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

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
# Unfenced code block fallback parser
# ---------------------------------------------------------------------------


class TestParseUnfencedBlocks:
    def test_bare_filepath_headers(self) -> None:
        """LLM outputs filenames as standalone lines without ``` markers."""
        text = (
            "pyproject.toml\n"
            "[project]\n"
            'name = "foo"\n'
            "\n"
            "src/foo/__init__.py\n"
            '"""Top-level package."""\n'
            "\n"
            "src/foo/solver.py\n"
            "def solve():\n"
            "    pass\n"
        )
        drafts = _parse_unfenced_blocks(text)
        assert set(drafts.keys()) == {
            "pyproject.toml",
            "src/foo/__init__.py",
            "src/foo/solver.py",
        }
        assert "[project]" in drafts["pyproject.toml"]
        assert "def solve" in drafts["src/foo/solver.py"]

    def test_single_header_is_ignored(self) -> None:
        """A single filepath-like line is not enough to trigger fallback."""
        text = "pyproject.toml\nsome content\n"
        assert _parse_unfenced_blocks(text) == {}

    def test_empty_content_skipped(self) -> None:
        """If a section between two headers is blank, skip it."""
        text = (
            "src/a.py\n"
            "\n"
            "src/b.py\n"
            "print('b')\n"
        )
        drafts = _parse_unfenced_blocks(text)
        assert "src/a.py" not in drafts
        assert "src/b.py" in drafts

    def test_parse_code_blocks_uses_fallback_when_no_fences(self) -> None:
        """_parse_code_blocks should delegate to unfenced fallback."""
        text = (
            "pyproject.toml\n"
            "[project]\n"
            'name = "bar"\n'
            "\n"
            "src/bar.py\n"
            "x = 1\n"
        )
        drafts = _parse_code_blocks(text)
        assert "pyproject.toml" in drafts
        assert "src/bar.py" in drafts

    def test_fenced_blocks_take_priority(self) -> None:
        """When fenced blocks are present, unfenced fallback is not used."""
        text = (
            "```src/a.py\nprint('a')\n```\n"
            "src/b.py\nprint('b')\n"
        )
        drafts = _parse_code_blocks(text)
        assert "src/a.py" in drafts
        assert "src/b.py" not in drafts

    def test_lines_with_spaces_not_treated_as_headers(self) -> None:
        """Lines containing spaces should not be mistaken for file paths."""
        text = (
            "src/a.py\n"
            "def hello():\n"
            "    print('hello world')\n"
            "some explanation text here\n"
            "src/b.py\n"
            "x = 1\n"
        )
        drafts = _parse_unfenced_blocks(text)
        assert "src/a.py" in drafts
        assert "src/b.py" in drafts
        # "some explanation text here" is NOT a filepath header (has spaces),
        # so it stays inside a.py's content block.
        assert "some explanation text here" in drafts["src/a.py"]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class TestVerifier:
    @pytest.fixture(autouse=True)
    def _mock_triage_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM triage returns LGTM so verifier falls through to pytest."""
        monkeypatch.setattr(
            "orchestrator.nodes._shared.get_llm", lambda: _FakeLLM("LGTM")
        )

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
        # Verifier should append to transcript
        assert len(result["transcript"]) == 1
        assert result["transcript"][0]["node"] == "verifier"
        assert "passed" in result["transcript"][0]["content"].lower()

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
        fake = _FakeLLM(
            "The factorial function always returns -1.\n\n"
            "## Key Findings\n"
            "- The function uses a hardcoded return value instead of recursion\n\n"
            "## Action\n"
            "RETRY"
        )
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(test_logs=["AssertionError: assert -1 == 120"])
        result = reflector(state)

        assert "reflection" in result
        assert "-1" in result["reflection"]
        assert "ground_truth" in result
        assert len(result["ground_truth"]) >= 1
        assert result["_reflector_action"] == "retry"


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
# Reflector context enrichment
# ---------------------------------------------------------------------------


class TestReflectorContext:
    """The reflector should receive direction, ground_truth, and prior reflection."""

    def test_includes_anti_test_change_directive(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reflector user message must instruct never to suggest test changes."""
        fake = _FakeLLM("Fix the implementation.")
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(
            test_logs=["TypeError: unexpected keyword argument 'V0'"],
            code_drafts={"solver.py": "def solve(depth): pass\n"},
        )
        result = reflector(state)
        prompt = result["_prompt_summary"]
        assert "NEVER suggest changing test" in prompt
        assert "fix the implementation" in prompt.lower()

    def test_includes_ground_truth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeLLM("Try a different approach.")
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(
            test_logs=["FAILED test - assert 0 == 1"],
            transcript=[
                make_entry(
                    "ai",
                    "Analysis: brentq needs bracket endpoints with opposite signs",
                    node="reflector", step=1, phase=0,
                ),
            ],
        )
        result = reflector(state)
        prompt = result["_prompt_summary"]
        assert "Run History" in prompt
        assert "brentq needs bracket" in prompt
        # Transcript delta contains only the new entry from this node
        assert len(result["transcript"]) == 1
        assert result["transcript"][-1]["node"] == "reflector"

    def test_includes_prior_reflection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeLLM("Different root cause found.")
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(
            test_logs=["FAILED test - assert 0 == 1"],
            transcript=[
                make_entry(
                    "ai",
                    "Previous suggestion: use finer grid. Did not work.",
                    node="reflector", step=1, phase=0,
                ),
            ],
        )
        result = reflector(state)
        prompt = result["_prompt_summary"]
        assert "Run History" in prompt
        assert "finer grid" in prompt


# ---------------------------------------------------------------------------
# Coder ground-truth context
# ---------------------------------------------------------------------------


class TestCoderGroundTruthContext:
    """The coder should see accumulated ground_truth as lessons learned."""

    def test_includes_ground_truth_in_prompt(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _FakeLLM("```solver.py\ndef solve(): pass\n```")
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(
            plan="Implement the solver",
            transcript=[
                make_entry(
                    "ai",
                    "Use brentq not bisect for root finding",
                    node="reflector", step=1, phase=0,
                ),
            ],
        )
        result = coder(state)
        prompt = result["_prompt_summary"]
        assert "Run History" in prompt
        assert "brentq not bisect" in prompt
        # Transcript delta contains only the new entry from this node
        assert len(result["transcript"]) == 1
        assert result["transcript"][-1]["node"] == "coder"

    def test_no_ground_truth_section_when_empty(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _FakeLLM("```solver.py\ndef solve(): pass\n```")
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(plan="Implement the solver")
        result = coder(state)
        prompt = result["_prompt_summary"]
        assert "Run History" not in prompt


# ---------------------------------------------------------------------------
# Verifier – write mode
# ---------------------------------------------------------------------------

class TestVerifierWriteMode:
    @pytest.fixture(autouse=True)
    def _mock_triage_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM triage returns LGTM so verifier falls through to pytest."""
        monkeypatch.setattr(
            "orchestrator.nodes._shared.get_llm", lambda: _FakeLLM("LGTM")
        )

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
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

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
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

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
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

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
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        existing = {"solver.py": "print('solver')"}
        state = _base_state(plan="Build CLI.", code_drafts=existing)
        result = coder(state)

        # Both old and new files should be in code_drafts
        assert "solver.py" in result["code_drafts"]
        assert "cli.py" in result["code_drafts"]

    def test_coder_mentions_existing_files(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm_output = "```cli.py\nprint('cli')\n```\n"
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

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
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

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
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(
            plan="implement constants",
            reflection="Define HBAR at module level, not inside get_constants().",
            transcript=[
                make_entry(
                    "ai",
                    "Define HBAR at module level, not inside get_constants().",
                    node="reflector", step=1, phase=0,
                ),
            ],
        )
        result = coder(state)

        prompt_sent = result["_prompt_summary"]
        assert "HBAR" in prompt_sent

    def test_includes_test_logs_in_prompt(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        llm_output = "```pkg/mod.py\nx = 1\n```\n"
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(
            plan="fix imports",
            transcript=[
                make_entry(
                    "human",
                    "Tests FAILED:\n```\nImport mismatch: tests/test_c.py imports 'HBAR'\n```",
                    node="verifier", step=1, phase=0,
                ),
            ],
        )
        result = coder(state)

        prompt_sent = result["_prompt_summary"]
        assert "Import mismatch" in prompt_sent

    def test_no_error_context_on_first_iteration(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        llm_output = "```pkg/mod.py\nx = 1\n```\n"
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

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
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(
            plan="fix imports",
            reflection="Constants must be at module level.",
            _error_repeat_count=4,
            transcript=[
                make_entry("human", "Tests FAILED:\n```\nNameError: HBAR not defined\n```",
                           node="verifier", step=1, phase=0),
                make_entry("ai", "Constants must be at module level.",
                           node="reflector", step=1, phase=0),
                make_entry("ai", "Generated 1 file(s): solver.py",
                           node="coder", step=1, phase=0),
                make_entry("human", "Tests FAILED:\n```\nNameError: HBAR not defined\n```",
                           node="verifier", step=2, phase=0),
                make_entry("ai", "Constants must be at module level.",
                           node="reflector", step=2, phase=0),
            ],
        )
        result = coder(state)

        prompt_sent = result["_prompt_summary"]
        # Repeated failure pattern is visible in the run history
        assert "Run History" in prompt_sent
        assert "NameError" in prompt_sent

    def test_revision_protects_clean_test_files(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On revision iterations with phase_iteration_count >= 2,
        test files that are in clean_files must not be overwritten."""
        # The coder emits both solver.py and test_solver.py
        llm_output = (
            "```solver.py\ndef solve(): return 42\n```\n\n"
            "```tests/test_solver.py\nassert False  # rewritten tests\n```\n"
        )
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        original_test = "from solver import solve\ndef test_it(): assert solve() == 42\n"
        state = _base_state(
            plan="fix solver",
            reflection="Return value is wrong.",
            _phase_iteration_count=2,
            code_drafts={
                "solver.py": "def solve(): return -1\n",
                "tests/test_solver.py": original_test,
            },
            clean_files=["tests/test_solver.py"],
        )
        result = coder(state)

        # Implementation should be updated
        assert "return 42" in result["code_drafts"]["solver.py"]
        # Test file should be PROTECTED (it's in clean_files)
        assert result["code_drafts"]["tests/test_solver.py"] == original_test

    def test_revision_allows_test_file_fixes_when_not_clean(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test files that are NOT in clean_files should be updatable by the
        coder, even after phase_iteration_count >= 2.  This ensures test infrastructure
        bugs (e.g. wrong subprocess invocation) can be fixed."""
        fixed_test = "import subprocess, sys\ndef run_cli(args):\n    return subprocess.run([sys.executable, '-m', 'pkg.cli'] + args, capture_output=True, text=True)\ndef test_cli(): assert run_cli(['--help']).returncode == 0\n"
        llm_output = (
            "```pkg/cli.py\nprint('hello')\n```\n\n"
            f"```tests/test_cli.py\n{fixed_test}```\n"
        )
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        buggy_test = "import subprocess, sys\ndef run_cli(args):\n    return subprocess.run([sys.executable, 'pkg/cli.py'] + args, capture_output=True, text=True)\ndef test_cli(): assert run_cli(['--help']).returncode == 0\n"
        state = _base_state(
            plan="fix CLI",
            reflection="ImportError: relative import with no known parent package.",
            _phase_iteration_count=3,
            code_drafts={
                "pkg/cli.py": "from .solver import solve\nprint('hello')\n",
                "tests/test_cli.py": buggy_test,
            },
            clean_files=[],  # test_cli.py is NOT clean — all tests fail
        )
        result = coder(state)

        # test_cli.py should be UPDATED (not protected — it's not in clean_files)
        assert result["code_drafts"]["tests/test_cli.py"] == fixed_test

    def test_first_revision_allows_test_fixes(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On the first revision (phase_iteration_count < 2), test files should
        NOT be protected — the coder needs one chance to fix bad tests."""
        rewritten_test = "from solver import solve\ndef test_it(): assert solve() == 42\n"
        llm_output = (
            "```solver.py\ndef solve(): return 42\n```\n\n"
            f"```tests/test_solver.py\n{rewritten_test}```\n"
        )
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(
            plan="fix solver",
            reflection="Tests are wrong — assert len==0 but always >=1 bound state.",
            _phase_iteration_count=1,
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
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(plan="implement solver")
        result = coder(state)

        assert "tests/test_solver.py" in result["code_drafts"]
        assert "assert solve() == 42" in result["code_drafts"]["tests/test_solver.py"]


# ---------------------------------------------------------------------------
# Stuck-loop detection in verifier
# ---------------------------------------------------------------------------

class TestStuckLoopDetection:
    """Verifier tracks consecutive identical error fingerprints."""

    @pytest.fixture(autouse=True)
    def _mock_triage_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM triage detects MISSING imports, returns LGTM for clean code."""
        class _TriageLLM:
            def invoke(self, messages: Any) -> Any:
                # Inspect the user message for MISSING imports
                user_msg = messages[-1].content if messages else ""
                msg = MagicMock()
                if "MISSING_A" in user_msg:
                    msg.content = "- Import mismatch: MISSING_A is not defined"
                elif "MISSING_B" in user_msg:
                    msg.content = "- Import mismatch: MISSING_B is not defined"
                elif "MISSING" in user_msg:
                    msg.content = "- Import mismatch: MISSING is not defined"
                else:
                    msg.content = "LGTM"
                return msg

        monkeypatch.setattr(
            "orchestrator.nodes._shared.get_llm", lambda: _TriageLLM()
        )

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


# ---------------------------------------------------------------------------
# _extract_action
# ---------------------------------------------------------------------------

class TestExtractAction:
    """_extract_action parses the ## Action recommendation from reflector output."""

    def test_returns_retry_by_default(self) -> None:
        assert _extract_action("Some analysis without an action section.") == "retry"

    def test_parses_retry(self) -> None:
        text = "Analysis.\n\n## Key Findings\n- F1\n\n## Action\nRETRY"
        assert _extract_action(text) == "retry"

    def test_parses_replan(self) -> None:
        text = "Analysis.\n\n## Key Findings\n- F1\n\n## Action\nREPLAN"
        assert _extract_action(text) == "replan"

    def test_case_insensitive(self) -> None:
        text = "Analysis.\n\n## Action\nReplan"
        assert _extract_action(text) == "replan"

    def test_empty_action_section_defaults_to_retry(self) -> None:
        text = "Analysis.\n\n## Action\n"
        assert _extract_action(text) == "retry"


# ---------------------------------------------------------------------------
# _format_code_listing
# ---------------------------------------------------------------------------

class TestFormatCodeListing:
    """_format_code_listing formats code_drafts as fenced Markdown blocks."""

    def test_basic_listing(self) -> None:
        drafts = {"a.py": "x = 1\n", "b.py": "y = 2\n"}
        result = _format_code_listing(drafts)
        assert len(result) == 2
        assert "### a.py" in result[0]
        assert "x = 1" in result[0]
        assert "```python" in result[0]

    def test_skips_non_python(self) -> None:
        drafts = {"a.py": "x = 1\n", "readme.md": "# Hello\n"}
        result = _format_code_listing(drafts)
        assert len(result) == 1

    def test_clean_files_annotated(self) -> None:
        drafts = {"a.py": "x = 1\n", "b.py": "y = 2\n"}
        result = _format_code_listing(drafts, clean_files={"a.py"})
        assert "✅" in result[0]
        assert "✅" not in result[1]

    def test_only_failing_skips_clean(self) -> None:
        drafts = {"a.py": "x = 1\n", "b.py": "y = 2\n"}
        result = _format_code_listing(
            drafts, clean_files={"a.py"}, only_failing=True,
        )
        assert len(result) == 1
        assert "b.py" in result[0]

    def test_empty_drafts(self) -> None:
        assert _format_code_listing({}) == []


# ---------------------------------------------------------------------------
# _extract_findings
# ---------------------------------------------------------------------------

class TestExtractFindings:
    """_extract_findings parses analysis + ## Key Findings from reflector output."""

    def test_splits_analysis_and_findings(self) -> None:
        text = "Root cause is X.\n\n## Key Findings\n- Finding one\n- Finding two"
        analysis, findings = _extract_findings(text)
        assert analysis == "Root cause is X."
        assert findings == ["- Finding one", "- Finding two"]

    def test_no_findings_section(self) -> None:
        text = "Root cause is X. No key findings header."
        analysis, findings = _extract_findings(text)
        assert analysis == text.strip()
        assert findings == []

    def test_none_findings(self) -> None:
        text = "Root cause is X.\n\n## Key Findings\nNONE"
        analysis, findings = _extract_findings(text)
        assert analysis == "Root cause is X."
        assert findings == []

    def test_ignores_non_bullet_lines(self) -> None:
        text = "Analysis.\n\n## Key Findings\nSome preamble\n- Actual finding\nMore text"
        analysis, findings = _extract_findings(text)
        assert findings == ["- Actual finding"]


# ---------------------------------------------------------------------------
# Ground truth deduplication
# ---------------------------------------------------------------------------

class TestGroundTruthDedup:
    """Reflector should not add duplicate findings to ground_truth."""

    def test_no_duplicate_findings(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        existing_finding = "- Define constants at module level"
        # LLM returns analysis with the same finding that already exists
        fake = _FakeLLM(
            "The constant is missing from the namespace.\n\n"
            "## Key Findings\n"
            + existing_finding
        )
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

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
        fake = _FakeLLM(
            "Analysis of the error.\n\n"
            "## Key Findings\n"
            "- Brand new finding\n- Another finding"
        )
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(
            test_logs=["Some error"],
            ground_truth=["- Existing finding"],
        )
        result = reflector(state)

        assert "- Existing finding" in result["ground_truth"]
        assert "- Brand new finding" in result["ground_truth"]
        assert "- Another finding" in result["ground_truth"]
        assert len(result["ground_truth"]) == 3

    def test_no_findings_when_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the LLM reports NONE in Key Findings, no new findings added."""
        fake = _FakeLLM(
            "Analysis.\n\n## Key Findings\nNONE"
        )
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(
            test_logs=["assert 1 == 0"],
            ground_truth=["- Solver returns wrong count"],
        )
        result = reflector(state)

        assert result["ground_truth"] == ["- Solver returns wrong count"]


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
# pyproject.toml validation
# ---------------------------------------------------------------------------


class TestCheckPyprojectToml:
    """_check_pyproject_toml should catch common pyproject.toml mistakes."""

    def test_valid_pyproject_passes(self) -> None:
        drafts = {
            "pyproject.toml": (
                "[build-system]\n"
                'requires = ["hatchling"]\n'
                'build-backend = "hatchling.build"\n\n'
                "[project]\n"
                'name = "my-package"\n'
                'version = "0.1.0"\n'
            ),
        }
        assert _check_pyproject_toml(drafts) == []

    def test_missing_build_system(self) -> None:
        drafts = {
            "pyproject.toml": (
                "[project]\n"
                'name = "my-package"\n'
            ),
        }
        errors = _check_pyproject_toml(drafts)
        assert len(errors) == 1
        assert "build-system" in errors[0].lower()

    def test_missing_project_name(self) -> None:
        drafts = {
            "pyproject.toml": (
                "[build-system]\n"
                'requires = ["hatchling"]\n'
                'build-backend = "hatchling.build"\n\n'
                "[project]\n"
                'version = "0.1.0"\n'
            ),
        }
        errors = _check_pyproject_toml(drafts)
        assert len(errors) == 1
        assert "name" in errors[0]

    def test_mixed_backends_detected(self) -> None:
        drafts = {
            "pyproject.toml": (
                "[build-system]\n"
                'requires = ["hatchling"]\n'
                'build-backend = "hatchling.build"\n\n'
                "[project]\n"
                'name = "my-package"\n\n'
                "[tool.setuptools.packages.find]\n"
                'include = ["my_package*"]\n'
            ),
        }
        errors = _check_pyproject_toml(drafts)
        assert len(errors) == 1
        assert "mixed" in errors[0].lower()

    def test_invalid_toml_syntax(self) -> None:
        drafts = {
            "pyproject.toml": "[project\nname = broken",
        }
        errors = _check_pyproject_toml(drafts)
        assert len(errors) == 1
        assert "TOML" in errors[0]

    def test_no_pyproject_returns_empty(self) -> None:
        """If no pyproject.toml exists, no errors reported."""
        drafts = {"solver.py": "x = 1\n"}
        assert _check_pyproject_toml(drafts) == []

    def test_missing_requires_and_backend(self) -> None:
        """build-system table present but missing required keys."""
        drafts = {
            "pyproject.toml": (
                "[build-system]\n"
                'requires = ["hatchling"]\n\n'
                "[project]\n"
                'name = "my-package"\n'
            ),
        }
        errors = _check_pyproject_toml(drafts)
        assert len(errors) == 1
        assert "build-backend" in errors[0]

    def test_empty_build_system_reports_both_fields(self) -> None:
        """Empty [build-system] table should report missing requires AND build-backend."""
        drafts = {
            "pyproject.toml": (
                "[build-system]\n\n"
                "[project]\n"
                'name = "my-package"\n'
            ),
        }
        errors = _check_pyproject_toml(drafts)
        assert len(errors) == 2
        combined = " ".join(errors)
        assert "requires" in combined
        assert "build-backend" in combined


# ---------------------------------------------------------------------------
# _check_shadowed_packages
# ---------------------------------------------------------------------------


class TestCheckShadowedPackages:
    """_check_shadowed_packages should catch directories that shadow installed packages."""

    def test_no_shadow_returns_empty(self) -> None:
        drafts = {
            "square_well/__init__.py": "",
            "square_well/solver.py": "x = 1\n",
            "tests/test_solver.py": "pass\n",
        }
        assert _check_shadowed_packages(drafts) == []

    def test_pytest_directory_detected(self) -> None:
        drafts = {
            "square_well/solver.py": "x = 1\n",
            "pytest/__init__.py": '"""stub pytest"""',
        }
        errors = _check_shadowed_packages(drafts)
        assert len(errors) == 1
        assert "pytest" in errors[0]
        assert "shadows" in errors[0].lower()

    def test_numpy_directory_detected(self) -> None:
        drafts = {
            "numpy/custom.py": "x = 1\n",
        }
        errors = _check_shadowed_packages(drafts)
        assert len(errors) == 1
        assert "numpy" in errors[0]

    def test_multiple_shadows_all_reported(self) -> None:
        drafts = {
            "pytest/__init__.py": "",
            "numpy/foo.py": "",
            "square_well/solver.py": "x = 1\n",
        }
        errors = _check_shadowed_packages(drafts)
        assert len(errors) == 2
        combined = " ".join(errors)
        assert "pytest" in combined
        assert "numpy" in combined

    def test_single_file_not_flagged(self) -> None:
        """A file at the root level (no directory) should not be flagged."""
        drafts = {"pytest.py": "x = 1\n"}
        assert _check_shadowed_packages(drafts) == []


# ---------------------------------------------------------------------------
# _llm_triage
# ---------------------------------------------------------------------------


class TestLlmTriage:
    """LLM triage replaces the old AST-based heuristics with a single LLM call."""

    def test_returns_empty_on_lgtm(self) -> None:
        """LGTM response means no structural issues."""
        llm = _FakeLLM("LGTM")
        drafts = {
            "solver.py": "def solve(): pass\n",
            "tests/test_solver.py": "from solver import solve\ndef test_it(): solve()\n",
        }
        errors, record = _llm_triage(drafts, llm)
        assert errors == []
        assert record is not None

    def test_returns_empty_on_blank(self) -> None:
        """Empty LLM response treated as no issues."""
        llm = _FakeLLM("")
        drafts = {"solver.py": "def solve(): pass\n"}
        errors, record = _llm_triage(drafts, llm)
        assert errors == []
        assert record is not None

    def test_parses_bullet_issues(self) -> None:
        """Bullet-point issues are extracted as separate errors."""
        llm = _FakeLLM(
            "- Import mismatch: tests/test_solver.py imports 'foo' from 'solver'\n"
            "- Contract mismatch in tests/test_solver.py: too few args\n"
        )
        drafts = {
            "solver.py": "def solve(): pass\n",
            "tests/test_solver.py": "from solver import foo\ndef test_it(): foo()\n",
        }
        errors, _ = _llm_triage(drafts, llm)
        assert len(errors) == 2
        assert "Import mismatch" in errors[0]
        assert "Contract mismatch" in errors[1]

    def test_skips_non_py_files(self) -> None:
        """Non-.py files should not be sent to LLM."""
        llm = _FakeLLM("LGTM")
        drafts = {"pyproject.toml": "[project]\nname='x'\n"}
        errors, record = _llm_triage(drafts, llm)
        assert errors == []
        assert record is None  # No LLM call made

    def test_non_bullet_lines_included(self) -> None:
        """Non-header, non-empty lines are captured too."""
        llm = _FakeLLM("Layout conflict detected in solver.py")
        drafts = {"solver.py": "x = 1\n"}
        errors, _ = _llm_triage(drafts, llm)
        assert len(errors) == 1
        assert "Layout conflict" in errors[0]

    def test_header_lines_skipped(self) -> None:
        """Lines starting with # should be ignored."""
        llm = _FakeLLM("# Issues found\n- Bad import in test.py")
        drafts = {"solver.py": "x = 1\n", "tests/test_solver.py": "pass\n"}
        errors, _ = _llm_triage(drafts, llm)
        assert len(errors) == 1
        assert "Bad import" in errors[0]


class TestLlmTriageInVerifier:
    """Verifier integrates LLM triage for code with no syntax/TOML errors."""

    def test_triage_errors_short_circuit_pytest(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When LLM triage finds issues, pytest should not run."""
        triage_response = "- Import mismatch: tests/test_mod.py imports 'missing' from 'pkg.mod'"
        monkeypatch.setattr(
            "orchestrator.nodes._shared.get_llm", lambda: _FakeLLM(triage_response)
        )
        state = _base_state(
            code_drafts={
                "pkg/__init__.py": "",
                "pkg/mod.py": "def solve(): pass\n",
                "tests/test_mod.py": (
                    "from pkg.mod import missing\n"
                    "def test_it(): pass\n"
                ),
            },
        )
        result = verifier(state)
        assert len(result["test_logs"]) == 1
        assert "Import mismatch" in result["test_logs"][0]

    def test_triage_lgtm_falls_through_to_pytest(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When triage says LGTM, pytest runs and tests pass."""
        monkeypatch.setattr(
            "orchestrator.nodes._shared.get_llm", lambda: _FakeLLM("LGTM")
        )
        state = _base_state(
            code_drafts={
                "pkg/__init__.py": "",
                "pkg/mod.py": "def solve(): return 42\n",
                "tests/test_mod.py": (
                    "from pkg.mod import solve\n"
                    "def test_it():\n    assert solve() == 42\n"
                ),
            },
        )
        result = verifier(state)
        assert result["test_logs"] == []

    def test_syntax_errors_skip_triage(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Syntax errors short-circuit before LLM triage even runs."""
        # LLM should NOT be called — raise if it is
        def _should_not_call():
            raise AssertionError("LLM should not be called for syntax errors")
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", _should_not_call)

        state = _base_state(
            code_drafts={
                "pkg/__init__.py": "",
                "pkg/mod.py": "def solve( pass\n",  # syntax error
                "tests/test_mod.py": "def test_it(): pass\n",
            },
        )
        result = verifier(state)
        assert any("SyntaxError" in log for log in result["test_logs"])

    def test_triage_transcript_entry(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verifier transcript includes triage error information."""
        triage_response = "- Duplicate layout: pkg exists in both flat and src"
        monkeypatch.setattr(
            "orchestrator.nodes._shared.get_llm", lambda: _FakeLLM(triage_response)
        )
        state = _base_state(
            code_drafts={
                "pkg/__init__.py": "",
                "pkg/mod.py": "def solve(): pass\n",
                "tests/test_mod.py": "def test_it(): pass\n",
            },
        )
        result = verifier(state)
        assert len(result["transcript"]) == 1
        assert "FAILED" in result["transcript"][0]["content"]


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
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

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
        # Phase 1 should be updated (description changes, title preserved)
        assert "brentq" in result["plan_phases"][0]["description"]
        assert result["plan_phases"][0]["title"] == "Solver"  # title locked to prevent scope regression
        # Error repeat count should be reset
        assert result["_error_repeat_count"] == 0

    def test_replan_includes_error_context(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        fake = _FakeLLM("## Phase 1: Fix\nNew approach.\nFiles: solver.py\n")
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        phases = [
            {"id": 1, "title": "Solver", "description": "Build solver.", "status": "pending", "files": []},
        ]
        state = _base_state(
            plan_phases=phases,
            current_phase=0,
            reflection="Off-by-one in bracket computation.",
            test_logs=["FAILED test_solver - assert 1.73 != 1.71"],
            transcript=[
                make_entry("human", "Tests FAILED:\n```\nFAILED test_solver - assert 1.73 != 1.71\n```",
                           node="verifier", step=1, phase=0),
                make_entry("ai", "Off-by-one in bracket computation.",
                           node="reflector", step=1, phase=0),
            ],
        )
        result = planner(state)

        # The prompt should contain the error analysis and test logs via transcript
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
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

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
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

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
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

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

    def test_replan_skips_scaffolding_when_multi_phase(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """When the LLM returns a full multi-phase plan during replan,
        pick the phase matching the current title, not the scaffolding."""
        monkeypatch.chdir(tmp_path)
        # LLM returns a full plan with scaffolding first
        multi_phase = (
            "## Phase 1: Scaffolding\nStub pass scaffolding files.\n"
            "## Phase 2: Bound-State Solver\nUse brentq root finder.\nFiles: solver.py\n"
            "## Phase 3: Wavefunctions\nCompute psi(x).\n"
        )
        fake = _FakeLLM(multi_phase)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        phases = [
            {"id": 1, "title": "Scaffolding", "description": "Done.", "status": "completed", "files": []},
            {"id": 2, "title": "Bound-State Solver", "description": "Old approach.", "status": "pending", "files": ["solver.py"]},
        ]
        state = _base_state(
            plan_phases=phases,
            current_phase=1,
            reflection="Matrix eigenvalue approach fails.",
            test_logs=["FAILED test_solver.py - wrong eigenvalues"],
        )
        result = planner(state)

        # Should pick the solver phase, not scaffolding
        assert "brentq" in result["plan_phases"][1]["description"]
        assert result["plan_phases"][1]["title"] == "Bound-State Solver"
        # Phase 1 should be untouched
        assert result["plan_phases"][0]["description"] == "Done."

    def test_replan_uses_raw_text_when_no_phase_headers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """When LLM outputs plain text (no ## Phase headers), use it as-is."""
        monkeypatch.chdir(tmp_path)
        plain_text = "Use scipy.optimize.brentq to find roots of the transcendental equation."
        fake = _FakeLLM(plain_text)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        phases = [
            {"id": 1, "title": "Solver", "description": "Old.", "status": "pending", "files": []},
        ]
        state = _base_state(
            plan_phases=phases,
            current_phase=0,
            reflection="Still failing.",
        )
        result = planner(state)

        assert "brentq" in result["plan_phases"][0]["description"]


# ---------------------------------------------------------------------------
# Should-continue routing (verifier → next node)
# ---------------------------------------------------------------------------

class TestShouldContinue:
    """Tests for _make_should_continue: the verifier's conditional edge."""

    def test_tests_passed_last_phase_returns_end(self) -> None:
        from src.cli import _make_should_continue

        fn = _make_should_continue(20)
        state = _base_state(
            test_logs=[],
            plan_phases=[{"id": 1, "title": "Only phase"}],
            current_phase=0,
        )
        assert fn(state) == "end"

    def test_tests_passed_more_phases_returns_advance(self) -> None:
        from src.cli import _make_should_continue

        fn = _make_should_continue(20)
        state = _base_state(
            test_logs=[],
            plan_phases=[{"id": 1, "title": "P1"}, {"id": 2, "title": "P2"}],
            current_phase=0,
        )
        assert fn(state) == "advance_phase"

    def test_tests_failed_returns_reflect(self) -> None:
        from src.cli import _make_should_continue

        fn = _make_should_continue(20)
        state = _base_state(
            test_logs=["FAILED test_a"],
            plan_phases=[{"id": 1, "title": "P1"}, {"id": 2, "title": "P2"}],
            current_phase=0,
        )
        assert fn(state) == "reflect"

    def test_max_iterations_returns_end(self) -> None:
        from src.cli import _make_should_continue

        fn = _make_should_continue(10)
        state = _base_state(
            test_logs=["FAILED test_a"],
            iteration_count=10,
        )
        assert fn(state) == "end"


# ---------------------------------------------------------------------------
# After-reflector routing
# ---------------------------------------------------------------------------

class TestAfterReflector:
    """The _after_reflector routing uses the reflector's ## Action recommendation."""

    def test_routes_to_coder_on_retry(self) -> None:
        from src.cli import _after_reflector

        state = _base_state(_reflector_action="retry")
        assert _after_reflector(state) == "coder"

    def test_routes_to_replan_on_replan(self) -> None:
        from src.cli import _after_reflector

        state = _base_state(_reflector_action="replan", _replan_count=0)
        assert _after_reflector(state) == "replan"

    def test_routes_to_coder_when_no_action(self) -> None:
        from src.cli import _after_reflector

        state = _base_state()
        assert _after_reflector(state) == "coder"

    def test_routes_to_coder_when_replans_exhausted(self) -> None:
        """After MAX_REPLANS, even if reflector says replan, route to coder."""
        from src.cli import _after_reflector, MAX_REPLANS

        state = _base_state(
            _reflector_action="replan",
            _replan_count=MAX_REPLANS,
        )
        assert _after_reflector(state) == "coder"


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


# ---------------------------------------------------------------------------
# _resolve_duplicate_layouts
# ---------------------------------------------------------------------------


class TestResolveDuplicateLayouts:
    """Auto-prune one side of flat-vs-src layout conflicts."""

    def test_prunes_flat_when_new_drafts_use_src(self) -> None:
        merged = {
            "pkg/__init__.py": "",
            "pkg/solver.py": "old flat code",
            "src/pkg/__init__.py": "",
            "src/pkg/solver.py": "new src code",
            "tests/test_solver.py": "test code",
        }
        new_drafts = {"src/pkg/solver.py": "new src code"}
        _resolve_duplicate_layouts(merged, new_drafts)
        assert "pkg/__init__.py" not in merged
        assert "pkg/solver.py" not in merged
        assert "src/pkg/solver.py" in merged

    def test_prunes_src_when_new_drafts_use_flat(self) -> None:
        merged = {
            "pkg/__init__.py": "",
            "pkg/solver.py": "new flat code",
            "src/pkg/__init__.py": "",
            "src/pkg/solver.py": "old src code",
            "tests/test_solver.py": "test code",
        }
        new_drafts = {"pkg/solver.py": "new flat code"}
        _resolve_duplicate_layouts(merged, new_drafts)
        assert "src/pkg/__init__.py" not in merged
        assert "src/pkg/solver.py" not in merged
        assert "pkg/solver.py" in merged

    def test_no_conflict_no_change(self) -> None:
        merged = {
            "src/pkg/__init__.py": "",
            "src/pkg/solver.py": "code",
            "tests/test_solver.py": "test code",
        }
        original = dict(merged)
        _resolve_duplicate_layouts(merged, {})
        assert merged == original

    def test_prefers_src_on_tie(self) -> None:
        """When new_drafts has equal files in both layouts, prefer src/."""
        merged = {
            "pkg/solver.py": "flat",
            "src/pkg/solver.py": "src",
        }
        new_drafts = {
            "pkg/solver.py": "flat",
            "src/pkg/solver.py": "src",
        }
        _resolve_duplicate_layouts(merged, new_drafts)
        assert "src/pkg/solver.py" in merged
        assert "pkg/solver.py" not in merged

    def test_defaults_to_src_when_new_drafts_unrelated(self) -> None:
        """When new_drafts has no files for conflicting pkg, default to src/."""
        merged = {
            "pkg/__init__.py": "",
            "pkg/solver.py": "flat code",
            "src/pkg/__init__.py": "",
            "src/pkg/solver.py": "src code",
            "tests/test_solver.py": "test code",
        }
        # new_drafts only touches tests, not the conflicting package
        new_drafts = {"tests/test_solver.py": "updated test"}
        _resolve_duplicate_layouts(merged, new_drafts)
        assert "src/pkg/solver.py" in merged
        assert "pkg/solver.py" not in merged


# ---------------------------------------------------------------------------
# Coder deletion marker
# ---------------------------------------------------------------------------


class TestCoderDeleteMarker:
    """Test that coder can signal file deletion via ``# DELETE``."""

    def test_delete_marker_removes_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Coder emitting '# DELETE' should remove the file from code_drafts."""
        llm_output = (
            "```old_pkg/solver.py\n"
            "# DELETE\n"
            "```\n"
            "```src/pkg/solver.py\n"
            "def solve(): pass\n"
            "```\n"
        )
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(
            plan="refactor to src layout",
            code_drafts={"old_pkg/solver.py": "old code"},
        )
        result = coder(state)

        # old file deleted, new file present
        assert "old_pkg/solver.py" not in result["code_drafts"]
        assert "src/pkg/solver.py" in result["code_drafts"]

    def test_delete_marker_with_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DELETE marker should work even with surrounding whitespace."""
        llm_output = "```old_file.py\n  # DELETE  \n```\n"
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(
            plan="clean up",
            code_drafts={"old_file.py": "old code"},
        )
        result = coder(state)
        assert "old_file.py" not in result["code_drafts"]


# ---------------------------------------------------------------------------
# Clean-file protection for source files
# ---------------------------------------------------------------------------


class TestCleanFileProtection:
    """Protect all clean files (test and source) from regression during revisions."""

    def test_clean_source_file_protected_on_revision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Clean source files should not be overwritten during revisions."""
        llm_output = (
            "```pkg/solver.py\n"
            "# regressed code\n"
            "```\n"
        )
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(
            plan="fix wavefunctions",
            code_drafts={"pkg/solver.py": "good code"},
            test_logs=["FAILED tests/test_wf.py - assert False"],
            reflection="The wavefunctions module has errors.",
            clean_files=["pkg/solver.py"],
            _phase_iteration_count=3,
        )
        result = coder(state)

        # solver.py was clean — should keep original, not regressed version
        assert result["code_drafts"]["pkg/solver.py"] == "good code"

    def test_dirty_source_file_can_be_overwritten(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-clean source files should still be overwritable."""
        llm_output = (
            "```pkg/solver.py\n"
            "# fixed code\n"
            "```\n"
        )
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(
            plan="fix solver",
            code_drafts={"pkg/solver.py": "broken code"},
            test_logs=["FAILED tests/test_solver.py - assert False"],
            clean_files=[],  # solver is NOT clean
            _phase_iteration_count=3,
        )
        result = coder(state)

        # solver.py was not clean — should be updated
        assert result["code_drafts"]["pkg/solver.py"] == "# fixed code\n"


# ---------------------------------------------------------------------------
# _pick_replan_phase
# ---------------------------------------------------------------------------


class TestPickReplanPhase:
    """Select the best phase from a replan response."""

    def test_single_phase_returned_directly(self) -> None:
        phases = [{"id": 1, "title": "Solver", "description": "Revised approach", "files": []}]
        result = _pick_replan_phase(phases, "Bound-State Eigenvalue Solver")
        assert result is not None
        assert result["description"] == "Revised approach"

    def test_picks_title_match_over_scaffolding(self) -> None:
        phases = [
            {"id": 1, "title": "Project Scaffolding", "description": "stub pass scaffolding", "files": []},
            {"id": 2, "title": "Bound-State Eigenvalue Solver", "description": "Use brentq to find roots", "files": []},
            {"id": 3, "title": "Wavefunctions", "description": "Compute psi(x)", "files": []},
        ]
        result = _pick_replan_phase(phases, "Bound-State Eigenvalue Solver")
        assert result is not None
        assert "brentq" in result["description"]

    def test_skips_scaffolding_when_no_title_match(self) -> None:
        phases = [
            {"id": 1, "title": "Scaffolding", "description": "stub pass scaffolding", "files": []},
            {"id": 2, "title": "Implementation", "description": "Real solver code", "files": []},
        ]
        result = _pick_replan_phase(phases, "Solver")
        assert result is not None
        assert result["description"] == "Real solver code"

    def test_empty_phases_returns_none(self) -> None:
        assert _pick_replan_phase([], "Solver") is None

    def test_all_scaffolding_returns_last(self) -> None:
        phases = [
            {"id": 1, "title": "Scaffolding", "description": "stub pass scaffolding", "files": []},
            {"id": 2, "title": "Scaffolding 2", "description": "more stub pass scaffolding", "files": []},
        ]
        result = _pick_replan_phase(phases, "Solver")
        assert result is not None
        assert result["id"] == 2


# ---------------------------------------------------------------------------
# Helpers for inner-loop tests
# ---------------------------------------------------------------------------

class _SequentialFakeLLM:
    """Returns a different response for each successive ``invoke()`` call.

    After all responses are exhausted, keeps returning the last one.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._call = 0

    def invoke(self, _messages: Any) -> Any:
        idx = min(self._call, len(self._responses) - 1)
        self._call += 1
        msg = MagicMock()
        msg.content = self._responses[idx]
        return msg


# ---------------------------------------------------------------------------
# _quick_verify
# ---------------------------------------------------------------------------

class TestQuickVerify:
    """Tests for the _quick_verify sandbox runner."""

    def test_passing_code_returns_empty(self) -> None:
        drafts = {
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
        }
        errors = _quick_verify(drafts)
        assert errors == []

    def test_syntax_error_caught(self) -> None:
        drafts = {
            "bad.py": "def broken(\n",
        }
        errors = _quick_verify(drafts)
        assert len(errors) > 0
        assert any("SyntaxError" in e for e in errors)

    def test_failing_test_returns_errors(self) -> None:
        drafts = {
            "mymath/__init__.py": "",
            "mymath/factorial.py": "def factorial(n): return -1\n",
            "tests/test_factorial.py": (
                "from mymath.factorial import factorial\n"
                "def test_five():\n"
                "    assert factorial(5) == 120\n"
            ),
        }
        errors = _quick_verify(drafts)
        assert len(errors) > 0

    def test_empty_drafts_returns_empty(self) -> None:
        assert _quick_verify({}) == []


# ---------------------------------------------------------------------------
# _merge_and_clean
# ---------------------------------------------------------------------------

class TestMergeAndClean:
    def test_merges_new_into_existing(self) -> None:
        existing = {"a.py": "old_a\n", "b.py": "old_b\n"}
        new = {"b.py": "new_b\n", "c.py": "new_c\n"}
        merged = _merge_and_clean(new, existing, set())
        assert merged == {"a.py": "old_a\n", "b.py": "new_b\n", "c.py": "new_c\n"}

    def test_applies_deletions(self) -> None:
        existing = {"a.py": "a\n", "b.py": "b\n"}
        new: dict[str, str] = {}
        merged = _merge_and_clean(new, existing, {"a.py"})
        assert "a.py" not in merged
        assert "b.py" in merged


# ---------------------------------------------------------------------------
# Coder inner loop
# ---------------------------------------------------------------------------

class TestCoderInnerLoop:
    """The coder's inner self-correction loop should re-invoke the LLM
    when initial output fails quick verification."""

    def test_no_inner_loop_when_tests_pass(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When code passes tests on first try, no inner-loop iterations."""
        llm_output = (
            "```mymath/__init__.py\n```\n"
            "```mymath/factorial.py\n"
            "def factorial(n: int) -> int:\n"
            "    return 1 if n <= 1 else n * factorial(n - 1)\n"
            "```\n"
            "```tests/test_factorial.py\n"
            "from mymath.factorial import factorial\n"
            "def test_base():\n"
            "    assert factorial(0) == 1\n"
            "```\n"
        )
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(plan="implement factorial")
        result = coder(state)

        assert result["_inner_loop_count"] == 0
        assert "mymath/factorial.py" in result["code_drafts"]

    def test_inner_loop_fixes_broken_code(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When first attempt has a bug, inner loop should self-correct."""
        # First response: broken (always returns -1)
        broken_output = (
            "```mymath/__init__.py\n```\n"
            "```mymath/factorial.py\n"
            "def factorial(n: int) -> int:\n"
            "    return -1\n"
            "```\n"
            "```tests/test_factorial.py\n"
            "from mymath.factorial import factorial\n"
            "def test_five():\n"
            "    assert factorial(5) == 120\n"
            "```\n"
        )
        # Second response: fixed
        fixed_output = (
            "```mymath/__init__.py\n```\n"
            "```mymath/factorial.py\n"
            "def factorial(n: int) -> int:\n"
            "    return 1 if n <= 1 else n * factorial(n - 1)\n"
            "```\n"
            "```tests/test_factorial.py\n"
            "from mymath.factorial import factorial\n"
            "def test_five():\n"
            "    assert factorial(5) == 120\n"
            "```\n"
        )
        fake = _SequentialFakeLLM([broken_output, fixed_output])
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(plan="implement factorial")
        result = coder(state)

        assert result["_inner_loop_count"] == 1
        assert "factorial" in result["code_drafts"]["mymath/factorial.py"]
        # The final code should be the fixed version
        assert "n - 1" in result["code_drafts"]["mymath/factorial.py"]

    def test_inner_loop_capped_at_max(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inner loop should not exceed MAX_INNER_ITERATIONS attempts."""
        # All responses are broken — loop should stop at the cap
        broken_output = (
            "```mymath/__init__.py\n```\n"
            "```mymath/factorial.py\n"
            "def factorial(n: int) -> int:\n"
            "    return -1\n"
            "```\n"
            "```tests/test_factorial.py\n"
            "from mymath.factorial import factorial\n"
            "def test_five():\n"
            "    assert factorial(5) == 120\n"
            "```\n"
        )
        fake = _SequentialFakeLLM([broken_output] * (MAX_INNER_ITERATIONS + 1))
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(plan="implement factorial")
        result = coder(state)

        # Should have done MAX_INNER_ITERATIONS - 1 corrections
        # (first attempt + up to MAX-1 corrections = MAX total attempts)
        assert result["_inner_loop_count"] == MAX_INNER_ITERATIONS - 1
        # LLM should have been called exactly MAX_INNER_ITERATIONS times
        assert fake._call == MAX_INNER_ITERATIONS

    def test_inner_loop_transcript_records_self_corrections(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Transcript entry should mention inner-loop self-corrections."""
        broken_output = (
            "```mymath/__init__.py\n```\n"
            "```mymath/factorial.py\n"
            "def factorial(n: int) -> int:\n"
            "    return -1\n"
            "```\n"
            "```tests/test_factorial.py\n"
            "from mymath.factorial import factorial\n"
            "def test_five():\n"
            "    assert factorial(5) == 120\n"
            "```\n"
        )
        fixed_output = (
            "```mymath/__init__.py\n```\n"
            "```mymath/factorial.py\n"
            "def factorial(n: int) -> int:\n"
            "    return 1 if n <= 1 else n * factorial(n - 1)\n"
            "```\n"
            "```tests/test_factorial.py\n"
            "from mymath.factorial import factorial\n"
            "def test_five():\n"
            "    assert factorial(5) == 120\n"
            "```\n"
        )
        fake = _SequentialFakeLLM([broken_output, fixed_output])
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(plan="implement factorial")
        result = coder(state)

        transcript_entry = result["transcript"][-1]
        assert "Inner-loop self-corrections: 1" in transcript_entry["content"]

    def test_inner_loop_skipped_when_no_code_blocks(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If LLM returns no parsable code, skip inner loop gracefully."""
        fake = _FakeLLM("Here are some thoughts but no code blocks.")
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(plan="implement something")
        result = coder(state)

        assert result["_inner_loop_count"] == 0
        assert result["code_drafts"] == {}

    def test_inner_loop_catches_syntax_errors(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Syntax errors should trigger inner-loop correction."""
        syntax_broken = (
            "```mymath/__init__.py\n```\n"
            "```mymath/calc.py\n"
            "def broken(\n"
            "```\n"
            "```tests/test_calc.py\n"
            "from mymath.calc import broken\n"
            "def test_it():\n"
            "    assert broken() is None\n"
            "```\n"
        )
        fixed = (
            "```mymath/__init__.py\n```\n"
            "```mymath/calc.py\n"
            "def broken():\n"
            "    return None\n"
            "```\n"
            "```tests/test_calc.py\n"
            "from mymath.calc import broken\n"
            "def test_it():\n"
            "    assert broken() is None\n"
            "```\n"
        )
        fake = _SequentialFakeLLM([syntax_broken, fixed])
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        state = _base_state(plan="implement calc")
        result = coder(state)

        assert result["_inner_loop_count"] == 1
        assert "def broken():" in result["code_drafts"]["mymath/calc.py"]


# ---------------------------------------------------------------------------
# Planner tool_calling dispatch (Phase 3)
# ---------------------------------------------------------------------------

class TestPlannerDirectMode:
    """Verify the planner uses planner_direct.md when tool_calling is True."""

    def test_uses_direct_prompt_when_tool_calling(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """When state has tool_calling=True, planner loads planner_direct.md."""
        monkeypatch.chdir(tmp_path)
        llm_output = (
            "## Phase 1: Solver\n"
            "Implement the solver with full tests.\n"
            "Files: solver.py, tests/test_solver.py\n"
        )
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        # Track which prompt file is loaded
        loaded_prompts: list[str] = []

        from orchestrator.nodes._shared import _load_prompt as orig_lp

        def spy_load_prompt(env_var: str, default_filename: str) -> str:
            loaded_prompts.append(default_filename)
            return orig_lp(env_var, default_filename)

        monkeypatch.setattr("orchestrator.nodes._planner._load_prompt", spy_load_prompt)

        state = _base_state(tool_calling=True)
        result = planner(state)

        assert "planner_direct.md" in loaded_prompts
        assert "planner.md" not in loaded_prompts
        assert len(result["plan_phases"]) == 1

    def test_uses_standard_prompt_without_tool_calling(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """When tool_calling is False/missing, planner uses planner.md."""
        monkeypatch.chdir(tmp_path)
        llm_output = (
            "## Phase 1: Scaffolding\nStub it out.\nFiles: solver.py\n\n"
            "## Phase 2: Solver\nImplement solver.\nFiles: solver.py\n"
        )
        fake = _FakeLLM(llm_output)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        loaded_prompts: list[str] = []
        from orchestrator.nodes._shared import _load_prompt as orig_lp

        def spy_load_prompt(env_var: str, default_filename: str) -> str:
            loaded_prompts.append(default_filename)
            return orig_lp(env_var, default_filename)

        monkeypatch.setattr("orchestrator.nodes._planner._load_prompt", spy_load_prompt)

        state = _base_state(tool_calling=False)
        result = planner(state)

        assert "planner.md" in loaded_prompts
        assert "planner_direct.md" not in loaded_prompts
        assert len(result["plan_phases"]) == 2

    def test_replan_ignores_tool_calling(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Replanning always uses planner_replan.md, regardless of tool_calling."""
        monkeypatch.chdir(tmp_path)
        replan_output = "Revised: try algorithm B instead."
        fake = _FakeLLM(replan_output)
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        loaded_prompts: list[str] = []
        from orchestrator.nodes._shared import _load_prompt as orig_lp

        def spy_load_prompt(env_var: str, default_filename: str) -> str:
            loaded_prompts.append(default_filename)
            return orig_lp(env_var, default_filename)

        monkeypatch.setattr("orchestrator.nodes._planner._load_prompt", spy_load_prompt)

        phases = [
            {"id": 1, "title": "Core", "description": "Old desc.",
             "status": "pending", "files": []},
        ]
        state = _base_state(
            tool_calling=True,
            reflection="stuck on error X",
            plan_phases=phases,
            current_phase=0,
        )
        result = planner(state)

        assert "planner_replan.md" in loaded_prompts
        assert "planner_direct.md" not in loaded_prompts


# ---------------------------------------------------------------------------
# auto_reflect (Phase 4)
# ---------------------------------------------------------------------------

class TestAutoReflect:
    """Tests for the lightweight auto_reflect node."""

    def test_retry_on_first_failure(self) -> None:
        """First failure returns retry action."""
        state = _base_state(
            test_logs=["FAILED test_foo - AssertionError"],
            _error_repeat_count=1,
            _phase_iteration_count=1,
        )
        result = auto_reflect(state)

        assert result["_reflector_action"] == "retry"
        assert "reflection" in result
        assert "FAILED" in result["reflection"]

    def test_replan_after_repeated_errors(self) -> None:
        """Repeated same error triggers replan."""
        state = _base_state(
            test_logs=["FAILED test_foo - same error"],
            _error_repeat_count=_REPLAN_ERROR_REPEAT_THRESHOLD,
            _phase_iteration_count=5,
        )
        result = auto_reflect(state)

        assert result["_reflector_action"] == "replan"
        assert "replan" in result["transcript"][-1]["content"].lower()

    def test_replan_just_below_threshold_retries(self) -> None:
        """One below threshold still retries."""
        state = _base_state(
            test_logs=["error output"],
            _error_repeat_count=_REPLAN_ERROR_REPEAT_THRESHOLD - 1,
            _phase_iteration_count=3,
        )
        result = auto_reflect(state)

        assert result["_reflector_action"] == "retry"

    def test_truncates_long_logs(self) -> None:
        """Very long test logs are truncated in reflection."""
        long_log = "x" * 5000
        state = _base_state(
            test_logs=[long_log],
            _error_repeat_count=1,
        )
        result = auto_reflect(state)

        assert len(result["reflection"]) == 3000

    def test_appends_transcript(self) -> None:
        """auto_reflect appends to transcript."""
        existing = [make_entry("ai", "prior", node="coder", step=0, phase=0)]
        state = _base_state(
            test_logs=["test failure"],
            _error_repeat_count=1,
            transcript=existing,
        )
        result = auto_reflect(state)

        # Transcript delta contains only the new entry from this node
        assert len(result["transcript"]) == 1
        assert result["transcript"][-1]["node"] == "auto_reflect"

    def test_no_ground_truth_mutation(self) -> None:
        """auto_reflect does not modify ground_truth (no LLM extraction)."""
        state = _base_state(
            test_logs=["FAILED"],
            ground_truth=["- existing finding"],
            _error_repeat_count=1,
        )
        result = auto_reflect(state)

        assert "ground_truth" not in result
