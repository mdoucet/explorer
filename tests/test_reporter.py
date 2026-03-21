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

    def test_planner_includes_prompt_summary(self, tmp_path: Path) -> None:
        logger = ChatLogger(str(tmp_path))
        logger.log_node("planner", {
            "plan": "The plan",
            "_prompt_summary": "## Task\nDo science",
        })
        content = (tmp_path / "01_planner.md").read_text()
        assert "Prompt sent to LLM" in content
        assert "Do science" in content

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
