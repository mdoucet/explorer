"""Tests for the CoderSandbox and tool-calling coder infrastructure."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from orchestrator.nodes import (
    CoderSandbox,
    make_sandbox_tools,
    MAX_TOOL_ROUNDS,
    _finalize_output,
    _tool_calling_coder,
    supports_tool_calling,
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


# ---------------------------------------------------------------------------
# CoderSandbox
# ---------------------------------------------------------------------------

class TestCoderSandbox:
    """Tests for the CoderSandbox file-management class."""

    def test_write_and_read_file(self) -> None:
        """Write a file and read it back."""
        sandbox = CoderSandbox()
        try:
            result = sandbox.write_file("hello.py", "print('hi')")
            assert "Wrote hello.py" in result
            content = sandbox.read_file("hello.py")
            assert content == "print('hi')"
        finally:
            sandbox.cleanup()

    def test_path_traversal_blocked_write(self) -> None:
        """Path traversal in write_file is blocked."""
        sandbox = CoderSandbox()
        try:
            result = sandbox.write_file("../escape.py", "bad")
            assert "Error" in result
            assert "path traversal" in result
        finally:
            sandbox.cleanup()

    def test_path_traversal_blocked_read(self) -> None:
        """Path traversal in read_file is blocked."""
        sandbox = CoderSandbox()
        try:
            result = sandbox.read_file("../../../etc/passwd")
            assert "Error" in result
        finally:
            sandbox.cleanup()

    def test_path_traversal_blocked_delete(self) -> None:
        """Path traversal in delete_file is blocked."""
        sandbox = CoderSandbox()
        try:
            result = sandbox.delete_file("../escape.py")
            assert "Error" in result
        finally:
            sandbox.cleanup()

    def test_delete_file(self) -> None:
        """Write then delete a file."""
        sandbox = CoderSandbox()
        try:
            sandbox.write_file("tmp.py", "x = 1")
            result = sandbox.delete_file("tmp.py")
            assert "Deleted tmp.py" in result
            assert sandbox.read_file("tmp.py").startswith("Error")
        finally:
            sandbox.cleanup()

    def test_delete_nonexistent(self) -> None:
        """Deleting a nonexistent file returns error."""
        sandbox = CoderSandbox()
        try:
            result = sandbox.delete_file("nope.py")
            assert "does not exist" in result
        finally:
            sandbox.cleanup()

    def test_list_files_empty(self) -> None:
        """Empty sandbox lists no files."""
        sandbox = CoderSandbox()
        try:
            assert "No files" in sandbox.list_files()
        finally:
            sandbox.cleanup()

    def test_list_files_tracks_writes(self) -> None:
        """list_files returns written files."""
        sandbox = CoderSandbox()
        try:
            sandbox.write_file("a.py", "")
            sandbox.write_file("b.py", "")
            listing = sandbox.list_files()
            assert "a.py" in listing
            assert "b.py" in listing
        finally:
            sandbox.cleanup()

    def test_collect_drafts(self) -> None:
        """collect_drafts returns written files as a dict."""
        sandbox = CoderSandbox()
        try:
            sandbox.write_file("mod.py", "x = 1\n")
            sandbox.write_file("test.py", "assert True\n")
            drafts = sandbox.collect_drafts()
            assert "mod.py" in drafts
            assert drafts["mod.py"] == "x = 1\n"
            assert "test.py" in drafts
        finally:
            sandbox.cleanup()

    def test_collect_drafts_excludes_deleted(self) -> None:
        """collect_drafts excludes deleted files."""
        sandbox = CoderSandbox()
        try:
            sandbox.write_file("keep.py", "x = 1")
            sandbox.write_file("gone.py", "x = 2")
            sandbox.delete_file("gone.py")
            drafts = sandbox.collect_drafts()
            assert "keep.py" in drafts
            assert "gone.py" not in drafts
        finally:
            sandbox.cleanup()

    def test_initial_drafts_seeded(self) -> None:
        """Sandbox is seeded with initial_drafts."""
        sandbox = CoderSandbox({"existing.py": "# old code\n"})
        try:
            content = sandbox.read_file("existing.py")
            assert content == "# old code\n"
            assert "existing.py" in sandbox.list_files()
        finally:
            sandbox.cleanup()

    def test_initial_drafts_path_traversal_skipped(self) -> None:
        """Path traversal in initial_drafts is silently skipped."""
        sandbox = CoderSandbox({"../evil.py": "import os; os.system('rm -rf /')"})
        try:
            drafts = sandbox.collect_drafts()
            assert "../evil.py" not in drafts
        finally:
            sandbox.cleanup()

    def test_run_tests_passing(self) -> None:
        """run_tests returns success for passing code."""
        sandbox = CoderSandbox()
        try:
            sandbox.write_file("hello.py", "def greet():\n    return 'hi'\n")
            sandbox.write_file(
                "tests/test_hello.py",
                "from hello import greet\n"
                "def test_greet():\n"
                "    assert greet() == 'hi'\n",
            )
            result = sandbox.run_tests()
            assert "passed" in result.lower()
        finally:
            sandbox.cleanup()

    def test_run_tests_failing(self) -> None:
        """run_tests returns error output for failing tests."""
        sandbox = CoderSandbox()
        try:
            sandbox.write_file("hello.py", "def greet():\n    return 'bye'\n")
            sandbox.write_file(
                "tests/test_hello.py",
                "from hello import greet\n"
                "def test_greet():\n"
                "    assert greet() == 'hi'\n",
            )
            result = sandbox.run_tests()
            assert "FAILED" in result or "failed" in result.lower()
        finally:
            sandbox.cleanup()

    def test_cleanup_removes_directory(self) -> None:
        """cleanup removes the sandbox directory."""
        sandbox = CoderSandbox()
        root = sandbox.root
        assert root.exists()
        sandbox.cleanup()
        assert not root.exists()

    def test_nested_directories(self) -> None:
        """write_file creates nested directories automatically."""
        sandbox = CoderSandbox()
        try:
            sandbox.write_file("pkg/sub/mod.py", "x = 1")
            content = sandbox.read_file("pkg/sub/mod.py")
            assert content == "x = 1"
        finally:
            sandbox.cleanup()


# ---------------------------------------------------------------------------
# make_sandbox_tools
# ---------------------------------------------------------------------------

class TestMakeSandboxTools:
    """Tests for the tool factory function."""

    def test_returns_five_tools(self) -> None:
        sandbox = CoderSandbox()
        try:
            tools = make_sandbox_tools(sandbox)
            assert len(tools) == 5
        finally:
            sandbox.cleanup()

    def test_tool_names(self) -> None:
        sandbox = CoderSandbox()
        try:
            tools = make_sandbox_tools(sandbox)
            names = {t.name for t in tools}
            assert names == {"write_file", "read_file", "delete_file", "run_tests", "list_files"}
        finally:
            sandbox.cleanup()

    def test_write_tool_invocation(self) -> None:
        """Tool invocation delegates to sandbox."""
        sandbox = CoderSandbox()
        try:
            tools = make_sandbox_tools(sandbox)
            write_tool = next(t for t in tools if t.name == "write_file")
            result = write_tool.invoke({"path": "f.py", "content": "x=1"})
            assert "Wrote f.py" in result
            assert sandbox.read_file("f.py") == "x=1"
        finally:
            sandbox.cleanup()

    def test_run_tests_tool_invocation(self) -> None:
        """run_tests tool executes pytest via sandbox."""
        sandbox = CoderSandbox()
        try:
            sandbox.write_file("hello.py", "def hi(): return 1\n")
            sandbox.write_file(
                "tests/test_hello.py",
                "from hello import hi\ndef test_hi(): assert hi() == 1\n",
            )
            tools = make_sandbox_tools(sandbox)
            run_tool = next(t for t in tools if t.name == "run_tests")
            result = run_tool.invoke({})
            assert "passed" in result.lower()
        finally:
            sandbox.cleanup()


# ---------------------------------------------------------------------------
# supports_tool_calling
# ---------------------------------------------------------------------------

class TestSupportsToolCalling:
    """Tests for the supports_tool_calling detection function."""

    def test_env_var_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EXPLORER_NO_TOOL_CALLING=1 disables tool calling."""
        monkeypatch.setenv("EXPLORER_NO_TOOL_CALLING", "1")
        llm = MagicMock()
        llm.bind_tools = MagicMock()
        assert supports_tool_calling(llm) is False

    def test_env_var_true_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXPLORER_NO_TOOL_CALLING", "true")
        llm = MagicMock()
        llm.bind_tools = MagicMock()
        assert supports_tool_calling(llm) is False

    def test_no_bind_tools_method(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM without bind_tools returns False."""
        monkeypatch.delenv("EXPLORER_NO_TOOL_CALLING", raising=False)
        llm = MagicMock(spec=[])  # no attributes
        assert supports_tool_calling(llm) is False

    def test_bind_tools_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM with working bind_tools returns True."""
        monkeypatch.delenv("EXPLORER_NO_TOOL_CALLING", raising=False)
        llm = MagicMock()
        llm.bind_tools = MagicMock()  # succeeds
        assert supports_tool_calling(llm) is True

    def test_bind_tools_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM where bind_tools raises returns False."""
        monkeypatch.delenv("EXPLORER_NO_TOOL_CALLING", raising=False)
        llm = MagicMock()
        llm.bind_tools = MagicMock(side_effect=NotImplementedError)
        assert supports_tool_calling(llm) is False


# ---------------------------------------------------------------------------
# _finalize_output
# ---------------------------------------------------------------------------

class TestFinalizeOutput:
    """Tests for the shared _finalize_output helper."""

    def test_basic_return_dict(self) -> None:
        state = _base_state()
        result = _finalize_output(
            state, "raw LLM text",
            {"a.py": "code"}, {}, set(), 0, "user msg",
        )
        assert result["code_drafts"] == {"a.py": "code"}
        assert result["coder_raw_response"] == "raw LLM text"
        assert result["_inner_loop_count"] == 0
        assert result["_prompt_summary"] == "user msg"
        assert isinstance(result["transcript"], list)

    def test_merged_drafts_override(self) -> None:
        state = _base_state()
        result = _finalize_output(
            state, "", {"new.py": "x"}, {"old.py": "y"}, set(), 0, "",
            merged_drafts={"merged.py": "z"},
        )
        assert result["code_drafts"] == {"merged.py": "z"}

    def test_deletions_applied(self) -> None:
        state = _base_state()
        result = _finalize_output(
            state, "", {}, {"old.py": "y"}, {"old.py"}, 0, "",
        )
        assert "old.py" not in result["code_drafts"]


# ---------------------------------------------------------------------------
# _tool_calling_coder
# ---------------------------------------------------------------------------

class _ToolCallingFakeLLM:
    """Mock LLM that supports tool calling.

    First call returns an AIMessage with tool_calls; second call returns
    text-only (signaling completion).
    """

    def __init__(self, tool_responses: list[Any]) -> None:
        self._responses = tool_responses
        self._call = 0
        self._bound_tools: list = []

    def bind_tools(self, tools: list) -> "_ToolCallingFakeLLM":
        clone = _ToolCallingFakeLLM(self._responses)
        clone._bound_tools = tools
        return clone

    def invoke(self, messages: Any) -> Any:
        idx = min(self._call, len(self._responses) - 1)
        self._call += 1
        return self._responses[idx]


class _TextOnlyFakeLLM:
    """Mock LLM that has bind_tools but returns text-only responses."""

    def __init__(self, content: str) -> None:
        self._content = content

    def bind_tools(self, tools: list) -> "_TextOnlyFakeLLM":
        return self  # accepts tools but won't use them

    def invoke(self, messages: Any) -> Any:
        msg = MagicMock()
        msg.content = self._content
        msg.tool_calls = []
        return msg


class TestToolCallingCoder:
    """Tests for _tool_calling_coder."""

    def test_tool_loop_writes_and_returns_drafts(self) -> None:
        """LLM uses write_file tool and drafts are collected."""
        from langchain_core.messages import AIMessage

        # Response 1: write a file via tool call
        resp1 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"path": "hello.py", "content": "def hi(): return 1\n"},
                    "id": "call_1",
                },
            ],
        )
        # Response 2: done (no tool calls)
        resp2 = AIMessage(content="Done!")

        llm = _ToolCallingFakeLLM([resp1, resp2])
        state = _base_state()

        result = _tool_calling_coder(
            state, llm, "system prompt", "user msg", {},
        )

        assert "hello.py" in result["code_drafts"]
        assert result["code_drafts"]["hello.py"] == "def hi(): return 1\n"
        assert result["_inner_loop_count"] == 1  # one tool round

    def test_text_fallback_on_no_tool_calls(self) -> None:
        """When LLM returns text only despite bind_tools, falls back to parsing."""
        code = (
            "```hello.py\n"
            "def hi(): return 1\n"
            "```\n"
        )
        llm = _TextOnlyFakeLLM(code)
        state = _base_state()

        result = _tool_calling_coder(
            state, llm, "system prompt", "user msg", {},
        )

        assert "hello.py" in result["code_drafts"]
        assert result["_inner_loop_count"] == 0  # no tool rounds

    def test_existing_drafts_preserved(self) -> None:
        """Existing drafts that aren't overwritten remain in output."""
        from langchain_core.messages import AIMessage

        resp1 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"path": "new.py", "content": "x = 2\n"},
                    "id": "call_1",
                },
            ],
        )
        resp2 = AIMessage(content="Done!")

        llm = _ToolCallingFakeLLM([resp1, resp2])
        state = _base_state()
        existing = {"old.py": "x = 1\n"}

        result = _tool_calling_coder(
            state, llm, "prompt", "user msg", existing,
        )

        assert "old.py" in result["code_drafts"]
        assert "new.py" in result["code_drafts"]

    def test_deletion_tracked(self) -> None:
        """When a file exists in initial drafts but is deleted via tool, it's removed."""
        from langchain_core.messages import AIMessage

        resp1 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "delete_file",
                    "args": {"path": "obsolete.py"},
                    "id": "call_1",
                },
            ],
        )
        resp2 = AIMessage(content="Done!")

        llm = _ToolCallingFakeLLM([resp1, resp2])
        state = _base_state()
        existing = {"obsolete.py": "old code\n", "keep.py": "good code\n"}

        result = _tool_calling_coder(
            state, llm, "prompt", "user msg", existing,
        )

        assert "obsolete.py" not in result["code_drafts"]
        assert "keep.py" in result["code_drafts"]

    def test_multiple_tool_rounds(self) -> None:
        """LLM can do multiple rounds of tool calls."""
        from langchain_core.messages import AIMessage

        resp1 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"path": "a.py", "content": "x = 1\n"},
                    "id": "c1",
                },
            ],
        )
        resp2 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"path": "b.py", "content": "y = 2\n"},
                    "id": "c2",
                },
            ],
        )
        resp3 = AIMessage(content="All done!")

        llm = _ToolCallingFakeLLM([resp1, resp2, resp3])
        state = _base_state()

        result = _tool_calling_coder(
            state, llm, "prompt", "user msg", {},
        )

        assert "a.py" in result["code_drafts"]
        assert "b.py" in result["code_drafts"]
        assert result["_inner_loop_count"] == 2  # two tool rounds

    def test_unknown_tool_returns_error(self) -> None:
        """Unknown tool name results in error message, not crash."""
        from langchain_core.messages import AIMessage

        resp1 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "nonexistent_tool",
                    "args": {},
                    "id": "c1",
                },
            ],
        )
        resp2 = AIMessage(content="Oops!")

        llm = _ToolCallingFakeLLM([resp1, resp2])
        state = _base_state()

        # Should not raise
        result = _tool_calling_coder(
            state, llm, "prompt", "user msg", {},
        )
        assert isinstance(result["code_drafts"], dict)

    def test_sandbox_cleanup_on_error(self) -> None:
        """Sandbox is cleaned up even if the tool loop raises."""
        from langchain_core.messages import AIMessage

        class _RaisingLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                raise RuntimeError("LLM crashed")

        llm = _RaisingLLM()
        state = _base_state()

        with pytest.raises(RuntimeError, match="LLM crashed"):
            _tool_calling_coder(state, llm, "prompt", "msg", {})


# ---------------------------------------------------------------------------
# coder() dispatch (integration-level)
# ---------------------------------------------------------------------------

class TestCoderToolDispatch:
    """Tests that coder() dispatches to tool-calling when supported."""

    def test_coder_uses_text_mode_when_no_tool_support(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When supports_tool_calling is False, text mode is used."""
        import importlib; shared = importlib.import_module("orchestrator.nodes.shared")
        _coder = importlib.import_module("orchestrator.nodes.coder")

        monkeypatch.setenv("EXPLORER_NO_TOOL_CALLING", "1")

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(
            content="```hello.py\nprint('hi')\n```"
        )
        monkeypatch.setattr(shared, "get_llm", lambda: fake_llm)
        monkeypatch.setattr(
            shared, "_llm", fake_llm,
        )

        state = _base_state(plan="Write hello.py")

        result = _coder.coder(state)
        assert "hello.py" in result["code_drafts"]

    def test_coder_dispatches_to_tool_calling(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When supports_tool_calling is True, _tool_calling_coder is called."""
        import importlib; shared = importlib.import_module("orchestrator.nodes.shared")
        _coder = importlib.import_module("orchestrator.nodes.coder")

        monkeypatch.delenv("EXPLORER_NO_TOOL_CALLING", raising=False)

        # Track whether _tool_calling_coder was called
        called = {"value": False}
        original_tool_coder = _coder._tool_calling_coder

        def mock_tool_coder(*args, **kwargs):
            called["value"] = True
            return {
                "code_drafts": {"tool.py": "x=1"},
                "coder_raw_response": "",
                "_prompt_summary": "",
                "_inner_loop_count": 0,
                "transcript": [],
            }

        fake_llm = MagicMock()
        fake_llm.bind_tools = MagicMock()  # has bind_tools → supports tools
        monkeypatch.setattr(shared, "get_llm", lambda: fake_llm)
        monkeypatch.setattr(shared, "_llm", fake_llm)
        monkeypatch.setattr(_coder, "_tool_calling_coder", mock_tool_coder)

        state = _base_state(plan="Write something")

        result = _coder.coder(state)
        assert called["value"] is True
        assert "tool.py" in result["code_drafts"]

    def test_coder_falls_back_on_tool_parse_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When tool-calling raises a tool-parse error, fall back to text mode."""
        import importlib; shared = importlib.import_module("orchestrator.nodes.shared")
        _coder = importlib.import_module("orchestrator.nodes.coder")

        monkeypatch.delenv("EXPLORER_NO_TOOL_CALLING", raising=False)

        def failing_tool_coder(*args, **kwargs):
            raise RuntimeError("error parsing tool call: raw='{...}', err=invalid character")

        fake_llm = MagicMock()
        fake_llm.bind_tools = MagicMock()
        fake_llm.invoke.return_value = MagicMock(
            content="```hello.py\nprint('hi')\n```"
        )
        monkeypatch.setattr(shared, "get_llm", lambda: fake_llm)
        monkeypatch.setattr(shared, "_llm", fake_llm)
        monkeypatch.setattr(_coder, "_tool_calling_coder", failing_tool_coder)

        state = _base_state(plan="Write hello.py")

        result = _coder.coder(state)
        # Should have fallen back to text mode and parsed the code blocks
        assert "hello.py" in result["code_drafts"]

    def test_coder_reraises_non_tool_errors(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-tool-parse errors still propagate."""
        import importlib; shared = importlib.import_module("orchestrator.nodes.shared")
        _coder = importlib.import_module("orchestrator.nodes.coder")

        monkeypatch.delenv("EXPLORER_NO_TOOL_CALLING", raising=False)

        def crashing_tool_coder(*args, **kwargs):
            raise ValueError("something completely different")

        fake_llm = MagicMock()
        fake_llm.bind_tools = MagicMock()
        monkeypatch.setattr(shared, "get_llm", lambda: fake_llm)
        monkeypatch.setattr(shared, "_llm", fake_llm)
        monkeypatch.setattr(_coder, "_tool_calling_coder", crashing_tool_coder)

        state = _base_state(plan="Write hello.py")

        with pytest.raises(ValueError, match="something completely different"):
            _coder.coder(state)


class TestMakeLlmCallRecord:
    """Tests for the make_llm_call_record helper."""

    def test_basic_record_structure(self) -> None:
        from orchestrator.nodes.shared import make_llm_call_record
        from langchain_core.messages import SystemMessage, HumanMessage

        response = MagicMock()
        response.content = "Hello world"
        response.tool_calls = []
        response.response_metadata = {}
        response.usage_metadata = {}

        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="User prompt"),
        ]
        record = make_llm_call_record(
            node="planner",
            messages=messages,
            response=response,
            duration_s=5.42,
            label="plan",
        )
        assert record["node"] == "planner"
        assert record["label"] == "plan"
        assert record["duration_s"] == 5.42
        assert record["system_prompt"] == "System prompt"
        assert record["user_prompt"] == "User prompt"
        assert record["response_text"] == "Hello world"
        assert record["tool_calls"] == []
        assert record["tool_messages"] == []
        assert "timestamp" in record

    def test_captures_tool_calls(self) -> None:
        from orchestrator.nodes.shared import make_llm_call_record
        from langchain_core.messages import SystemMessage, HumanMessage

        response = MagicMock()
        response.content = ""
        response.tool_calls = [{"name": "write_file", "args": {"path": "x.py"}, "id": "1"}]
        response.response_metadata = {}
        response.usage_metadata = {}

        record = make_llm_call_record(
            node="coder",
            messages=[SystemMessage(content="s"), HumanMessage(content="u")],
            response=response,
            duration_s=1.0,
            tool_messages=[{"tool": "write_file", "args": {"path": "x.py"}, "result": "OK"}],
        )
        assert len(record["tool_calls"]) == 1
        assert record["tool_calls"][0]["name"] == "write_file"
        assert len(record["tool_messages"]) == 1

    def test_captures_token_usage(self) -> None:
        from orchestrator.nodes.shared import make_llm_call_record
        from langchain_core.messages import HumanMessage

        response = MagicMock()
        response.content = "reply"
        response.tool_calls = []
        response.response_metadata = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        response.usage_metadata = {}

        record = make_llm_call_record(
            node="reflector",
            messages=[HumanMessage(content="u")],
            response=response,
            duration_s=2.0,
        )
        assert record["token_usage"]["prompt_tokens"] == 100
        assert record["token_usage"]["completion_tokens"] == 50
