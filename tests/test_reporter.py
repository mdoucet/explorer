"""Unit tests for ChatLogger in reporter.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.reporter import ChatLogger


class TestChatLoggerRunInfo:
    """Tests for the 00_run_info.md file written at init."""

    def test_writes_run_info_with_skills(self, tmp_path: Path) -> None:
        logger = ChatLogger(
            str(tmp_path),
            task="Solve the wave equation",
            skills=["python-testing", "quantum-mechanics"],
            provider="ollama",
            model="qwen2.5-coder:32b",
            max_iterations=10,
        )
        info = (tmp_path / "00_run_info.md").read_text()
        assert "python-testing" in info
        assert "quantum-mechanics" in info
        assert "ollama" in info
        assert "qwen2.5-coder:32b" in info
        assert "Solve the wave equation" in info
        assert "10" in info

    def test_writes_run_info_without_skills(self, tmp_path: Path) -> None:
        logger = ChatLogger(str(tmp_path), task="Hello")
        info = (tmp_path / "00_run_info.md").read_text()
        assert "_(none)_" in info

    def test_step_counter_starts_after_run_info(self, tmp_path: Path) -> None:
        logger = ChatLogger(str(tmp_path))
        logger.log_node("planner", {"plan": "my plan"})
        assert (tmp_path / "01_planner.md").exists()


class TestChatLoggerLogNode:
    """Tests for enriched log_node output."""

    def test_planner_includes_skills_banner(self, tmp_path: Path) -> None:
        logger = ChatLogger(
            str(tmp_path),
            skills=["python-testing"],
        )
        logger.log_node("planner", {"plan": "Step 1: do stuff"})
        content = (tmp_path / "01_planner.md").read_text()
        assert "python-testing" in content
        assert "Plan" in content
        assert "Step 1: do stuff" in content

    def test_planner_trace_links_to_llm_call_file(self, tmp_path: Path) -> None:
        logger = ChatLogger(str(tmp_path))
        logger.log_node("planner", {
            "plan": "The plan",
            "_llm_calls": [{
                "node": "planner",
                "label": "plan",
                "duration_s": 4.0,
                "system_prompt": "sys",
                "user_prompt": "usr",
                "response_text": "resp",
            }],
        })
        content = (tmp_path / "01_planner.md").read_text()
        assert "LLM calls:** 1" in content
        assert "01_planner_llm_call_01.md" in content
        assert (tmp_path / "01_planner_llm_call_01.md").exists()

    def test_coder_shows_file_count(self, tmp_path: Path) -> None:
        logger = ChatLogger(str(tmp_path))
        logger.log_node("coder", {
            "code_drafts": {"main.py": "print('hi')\n", "test_main.py": "pass\n"},
            "coder_raw_response": "...",
        })
        content = (tmp_path / "01_coder.md").read_text()
        assert "Generated Files (2)" in content
        assert "main.py" in content

    def test_verifier_pass(self, tmp_path: Path) -> None:
        logger = ChatLogger(str(tmp_path))
        logger.log_node("verifier", {"test_logs": [], "iteration_count": 1})
        content = (tmp_path / "01_verifier.md").read_text()
        assert "passed" in content.lower()

    def test_verifier_fail(self, tmp_path: Path) -> None:
        logger = ChatLogger(str(tmp_path))
        logger.log_node("verifier", {
            "test_logs": ["FAILED test_foo"],
            "iteration_count": 2,
        })
        content = (tmp_path / "01_verifier.md").read_text()
        assert "failed" in content.lower()
        assert "FAILED test_foo" in content

    def test_reflector_output(self, tmp_path: Path) -> None:
        logger = ChatLogger(str(tmp_path))
        logger.log_node("reflector", {
            "reflection": "The error was a typo",
            "ground_truth": ["- Fixed import"],
        })
        content = (tmp_path / "01_reflector.md").read_text()
        assert "The error was a typo" in content
        assert "Fixed import" in content


class TestChatLoggerSummary:
    """Tests for write_summary with enriched metadata."""

    def test_summary_includes_model_and_skills(self, tmp_path: Path) -> None:
        logger = ChatLogger(
            str(tmp_path),
            task="test task",
            skills=["python-testing"],
            provider="ollama",
            model="qwen2.5:32b",
        )
        logger.write_summary({
            "task_description": "test task",
            "iteration_count": 1,
            "test_logs": [],
            "code_drafts": {"main.py": "pass"},
            "ground_truth": [],
        })
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["model"] == "ollama/qwen2.5:32b"
        assert summary["skills"] == ["python-testing"]
        assert summary["passed"] is True


class TestChatLoggerLlmCalls:
    """Tests for _llm_calls rendering in log_node and write_summary."""

    def test_planner_llm_call_file_has_full_details(self, tmp_path: Path) -> None:
        logger = ChatLogger(str(tmp_path))
        logger.log_node("planner", {
            "plan": "Step 1",
            "_llm_calls": [{
                "node": "planner",
                "label": "plan",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "duration_s": 12.5,
                "system_prompt": "You are a planner.",
                "user_prompt": "Solve the equation.",
                "response_text": "## Phase 1: Solve it",
                "tool_calls": [],
                "tool_messages": [],
                "token_usage": {"prompt_tokens": 500, "completion_tokens": 200},
            }],
        })
        # Trace file links to LLM call file
        trace = (tmp_path / "01_planner.md").read_text()
        assert "LLM calls:** 1" in trace
        assert "12.5s" in trace
        # Separate file has full details
        call_content = (tmp_path / "01_planner_llm_call_01.md").read_text()
        assert "in=500" in call_content
        assert "out=200" in call_content
        assert "You are a planner." in call_content
        assert "Solve the equation." in call_content
        assert "Phase 1: Solve it" in call_content

    def test_coder_llm_call_file_includes_tool_calls(self, tmp_path: Path) -> None:
        logger = ChatLogger(str(tmp_path))
        logger.log_node("coder", {
            "code_drafts": {"main.py": "print('hi')"},
            "coder_raw_response": "...",
            "_llm_calls": [{
                "node": "coder",
                "label": "tool-round-0",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "duration_s": 8.3,
                "system_prompt": "You are a coder.",
                "user_prompt": "Write code.",
                "response_text": "",
                "tool_calls": [{"name": "write_file", "args": {"path": "main.py"}}],
                "tool_messages": [
                    {"tool": "write_file", "args": {"path": "main.py"}, "result": "OK"},
                ],
                "token_usage": {},
            }],
        })
        # Trace file links but has no LLM details
        trace = (tmp_path / "01_coder.md").read_text()
        assert "01_coder_llm_call_01.md" in trace
        assert "You are a coder." not in trace
        # Separate file has tool call details
        call_content = (tmp_path / "01_coder_llm_call_01.md").read_text()
        assert "Tool Calls Issued" in call_content
        assert "write_file" in call_content
        assert "Tool Results" in call_content

    def test_summary_includes_llm_call_stats(self, tmp_path: Path) -> None:
        logger = ChatLogger(
            str(tmp_path),
            provider="ollama",
            model="test",
        )
        logger.write_summary({
            "task_description": "test",
            "iteration_count": 1,
            "test_logs": [],
            "code_drafts": {},
            "ground_truth": [],
            "_llm_calls": [
                {"node": "planner", "duration_s": 5.0},
                {"node": "coder", "duration_s": 10.0},
                {"node": "coder", "duration_s": 3.0},
            ],
        })
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["llm_calls_total"] == 3
        assert summary["llm_calls_by_node"] == {"planner": 1, "coder": 2}
        assert summary["llm_total_duration_s"] == 18.0

    def test_summary_writes_llm_calls_json(self, tmp_path: Path) -> None:
        logger = ChatLogger(str(tmp_path), provider="ollama", model="test")
        calls = [
            {"node": "planner", "duration_s": 5.0, "system_prompt": "sys", "user_prompt": "usr"},
        ]
        logger.write_summary({
            "task_description": "test",
            "iteration_count": 1,
            "test_logs": [],
            "code_drafts": {},
            "ground_truth": [],
            "_llm_calls": calls,
        })
        llm_log = json.loads((tmp_path / "llm_calls.json").read_text())
        assert len(llm_log) == 1
        assert llm_log[0]["node"] == "planner"

    def test_no_llm_calls_skips_section(self, tmp_path: Path) -> None:
        logger = ChatLogger(str(tmp_path))
        logger.log_node("planner", {
            "plan": "Step 1",
            "_llm_calls": [],
        })
        content = (tmp_path / "01_planner.md").read_text()
        assert "LLM calls:" not in content
        # No companion files created
        assert not list(tmp_path.glob("*llm_call*"))

    def test_auto_reflect_output(self, tmp_path: Path) -> None:
        logger = ChatLogger(str(tmp_path))
        logger.log_node("auto_reflect", {
            "_reflector_action": "replan",
            "reflection": "FAILED test_main - assert 1 == 2",
        })
        content = (tmp_path / "01_auto_reflect.md").read_text()
        assert "replan" in content
        assert "FAILED test_main" in content
