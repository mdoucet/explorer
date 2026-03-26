"""Tests for the conversation transcript module."""

from __future__ import annotations

from orchestrator.transcript import DEFAULT_MAX_RECENT, format_history, make_entry


class TestMakeEntry:
    def test_basic_entry(self) -> None:
        entry = make_entry("ai", "Hello world", node="coder", step=3, phase=1)
        assert entry["role"] == "ai"
        assert entry["content"] == "Hello world"
        assert entry["node"] == "coder"
        assert entry["step"] == 3
        assert entry["phase"] == 1
        assert entry["summary"] == "Hello world"

    def test_custom_summary(self) -> None:
        entry = make_entry("human", "Long content\nwith newlines", summary="Short")
        assert entry["summary"] == "Short"

    def test_auto_summary_from_first_line(self) -> None:
        entry = make_entry("ai", "First line\nSecond line\nThird line")
        assert entry["summary"] == "First line"

    def test_auto_summary_truncated(self) -> None:
        long_first_line = "x" * 300
        entry = make_entry("ai", long_first_line)
        assert len(entry["summary"]) == 200

    def test_defaults(self) -> None:
        entry = make_entry("human", "content")
        assert entry["node"] == ""
        assert entry["step"] == 0
        assert entry["phase"] == 0


class TestFormatHistory:
    def test_empty_transcript(self) -> None:
        assert format_history([]) == ""

    def test_single_entry(self) -> None:
        transcript = [make_entry("ai", "Plan output", node="planner", step=0)]
        result = format_history(transcript)
        assert "## Run History" in result
        assert "Plan output" in result
        assert "planner" in result

    def test_multiple_entries_within_window(self) -> None:
        transcript = [
            make_entry("ai", "Generated plan", node="planner", step=0),
            make_entry("ai", "Generated 3 files", node="coder", step=0),
            make_entry("human", "All tests passed ✓", node="verifier", step=1),
        ]
        result = format_history(transcript)
        assert "Generated plan" in result
        assert "Generated 3 files" in result
        assert "All tests passed" in result
        # No condensed section when within window
        assert "condensed" not in result

    def test_windowing_condenses_old_entries(self) -> None:
        transcript = []
        for i in range(12):
            transcript.append(make_entry(
                "ai", f"Iteration {i} content here" * 10,
                node="coder", step=i,
                summary=f"Step {i} summary",
            ))
        result = format_history(transcript, max_recent=8)
        assert "condensed" in result
        # Old entries (0-3) should appear as summaries
        assert "Step 0 summary" in result
        assert "Step 3 summary" in result
        # Recent entries (4-11) should appear in full
        assert "Iteration 11 content here" in result

    def test_recent_entries_shown_in_full(self) -> None:
        """Recent entries show full content, not just summaries."""
        transcript = [
            make_entry("ai", "Old content line", node="coder", step=0,
                       summary="Old summary"),
        ] * 5 + [
            make_entry("human", "FAILED test_solver:\nassert 7 != 5",
                       node="verifier", step=5, summary="Tests: 1 failed"),
        ]
        result = format_history(transcript, max_recent=3)
        # The verifier entry is in the recent window — full content shown
        assert "assert 7 != 5" in result

    def test_step_and_node_in_header(self) -> None:
        transcript = [make_entry("ai", "content", node="reflector", step=3)]
        result = format_history(transcript)
        assert "step 3" in result
        assert "reflector" in result

    def test_default_max_recent(self) -> None:
        assert DEFAULT_MAX_RECENT == 8


class TestTranscriptIntegration:
    """Test that transcript entries from nodes are well-formed."""

    def test_planner_transcript_entry(self) -> None:
        """Planner entries should include the plan text."""
        entry = make_entry(
            "ai", "## Phase 1: Scaffolding\nBuild project skeleton.",
            node="planner", step=0, phase=0,
            summary="Plan: 2 phases — Scaffolding, Solver",
        )
        assert entry["role"] == "ai"
        assert entry["node"] == "planner"
        assert "Scaffolding" in entry["content"]

    def test_verifier_pass_entry(self) -> None:
        entry = make_entry(
            "human", "All tests passed ✓",
            node="verifier", step=2, phase=1,
            summary="Tests: All passed ✓",
        )
        assert entry["role"] == "human"
        assert entry["node"] == "verifier"

    def test_verifier_fail_entry(self) -> None:
        entry = make_entry(
            "human", "Tests FAILED:\n```\nFAILED test_solver\n```",
            node="verifier", step=3, phase=1,
        )
        assert entry["role"] == "human"
        assert "FAILED" in entry["content"]

    def test_reflector_entry_with_findings(self) -> None:
        entry = make_entry(
            "ai",
            "Root cause: wrong bracket.\n\n## Key findings\n- Use wider bracket",
            node="reflector", step=3, phase=1,
        )
        assert "Key findings" in entry["content"]
        assert "bracket" in entry["summary"]

    def test_advance_phase_entry(self) -> None:
        entry = make_entry(
            "human",
            "Phase 1 'Scaffolding' completed. Moving to Phase 2: 'Solver'.",
            node="advance_phase", step=5, phase=1,
        )
        assert entry["role"] == "human"
        assert "Moving to Phase 2" in entry["content"]
