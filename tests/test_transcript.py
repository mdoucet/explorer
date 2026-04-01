"""Tests for the conversation transcript module."""

from __future__ import annotations

from orchestrator.transcript import (
    DEFAULT_MAX_RECENT,
    _compute_diff_summary,
    _condensed_summary,
    format_history,
    make_entry,
)


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


class TestMakeEntryMetadata:
    """Tests for the metadata parameter of make_entry."""

    def test_metadata_included_when_provided(self) -> None:
        meta = {"type": "plan", "phases": ["A", "B"], "is_replan": False}
        entry = make_entry("ai", "content", metadata=meta)
        assert entry["metadata"] == meta

    def test_metadata_absent_when_none(self) -> None:
        entry = make_entry("ai", "content")
        assert "metadata" not in entry

    def test_metadata_absent_when_empty_dict(self) -> None:
        entry = make_entry("ai", "content", metadata={})
        assert "metadata" not in entry


class TestComputeDiffSummary:
    """Tests for _compute_diff_summary."""

    def test_no_changes(self) -> None:
        drafts = {"a.py": "x = 1\n"}
        diff = _compute_diff_summary(drafts, drafts)
        assert diff["files_added"] == []
        assert diff["files_deleted"] == []
        assert diff["files_modified"] == []
        assert "0 file(s) changed" in diff["summary"]

    def test_file_added(self) -> None:
        old: dict[str, str] = {}
        new = {"solver.py": "def solve():\n    pass\n"}
        diff = _compute_diff_summary(old, new)
        assert diff["files_added"] == ["solver.py"]
        assert diff["files_deleted"] == []
        assert diff["files_modified"] == []
        assert "1 file(s) changed" in diff["summary"]
        assert "+2" in diff["line_deltas"]["solver.py"]

    def test_file_deleted(self) -> None:
        old = {"old.py": "x = 1\n"}
        new: dict[str, str] = {}
        diff = _compute_diff_summary(old, new)
        assert diff["files_added"] == []
        assert diff["files_deleted"] == ["old.py"]
        assert "deleted" in diff["summary"]

    def test_file_modified(self) -> None:
        old = {"a.py": "x = 1\n"}
        new = {"a.py": "x = 2\ny = 3\n"}
        diff = _compute_diff_summary(old, new)
        assert diff["files_modified"] == ["a.py"]
        assert "a.py" in diff["line_deltas"]
        assert "modified" in diff["summary"]

    def test_mixed_changes(self) -> None:
        old = {"keep.py": "x = 1\n", "remove.py": "old\n"}
        new = {"keep.py": "x = 2\n", "add.py": "new\n"}
        diff = _compute_diff_summary(old, new)
        assert diff["files_added"] == ["add.py"]
        assert diff["files_deleted"] == ["remove.py"]
        assert diff["files_modified"] == ["keep.py"]
        assert "3 file(s) changed" in diff["summary"]

    def test_empty_to_empty(self) -> None:
        diff = _compute_diff_summary({}, {})
        assert diff["files_added"] == []
        assert "0 file(s) changed" in diff["summary"]


class TestCondensedSummary:
    """Tests for _condensed_summary with metadata."""

    def test_code_change_uses_diff(self) -> None:
        entry = make_entry("ai", "Generated files", summary="Generated 3 files")
        meta = {"type": "code_change", "diff_summary": "3 files changed (+45/−12)"}
        assert _condensed_summary(entry, meta) == "3 files changed (+45/−12)"

    def test_test_result_passed(self) -> None:
        entry = make_entry("human", "All tests passed")
        meta = {"type": "test_result", "passed": True}
        assert _condensed_summary(entry, meta) == "Tests: All passed ✓"

    def test_test_result_failed_with_repeats(self) -> None:
        entry = make_entry("human", "FAILED")
        meta = {"type": "test_result", "passed": False,
                "error_repeat_count": 3, "collection_error": False}
        result = _condensed_summary(entry, meta)
        assert "FAILED" in result
        assert "×3" in result

    def test_test_result_collection_error(self) -> None:
        entry = make_entry("human", "FAILED")
        meta = {"type": "test_result", "passed": False,
                "collection_error": True, "error_repeat_count": 1}
        assert "collection error" in _condensed_summary(entry, meta)

    def test_plan_initial(self) -> None:
        entry = make_entry("ai", "plan text")
        meta = {"type": "plan", "phases": ["A", "B"], "is_replan": False}
        result = _condensed_summary(entry, meta)
        assert "Plan" in result
        assert "2 phase" in result

    def test_plan_replan(self) -> None:
        entry = make_entry("ai", "replan text")
        meta = {"type": "plan", "phases": ["A"], "is_replan": True}
        assert "Replan" in _condensed_summary(entry, meta)

    def test_analysis_action(self) -> None:
        entry = make_entry("ai", "analysis")
        meta = {"type": "analysis", "action": "retry"}
        assert "retry" in _condensed_summary(entry, meta)

    def test_phase_advance(self) -> None:
        entry = make_entry("human", "advancing")
        meta = {"type": "phase_advance", "from_phase": 0, "to_phase": 1}
        result = _condensed_summary(entry, meta)
        assert "Phase 1" in result
        assert "Phase 2" in result

    def test_fallback_without_metadata(self) -> None:
        entry = make_entry("ai", "some content", summary="my summary")
        assert _condensed_summary(entry, {}) == "my summary"

    def test_format_history_uses_metadata_for_condensed(self) -> None:
        """Integration: verify format_history renders metadata-enriched summaries."""
        transcript = []
        for i in range(10):
            transcript.append(make_entry(
                "ai", f"Iteration {i} content" * 10,
                node="coder", step=i, phase=0,
                metadata={"type": "code_change",
                           "diff_summary": f"2 files changed (+{i*10}/−{i})"},
            ))
        result = format_history(transcript, max_recent=3)
        assert "condensed" in result
        # Old entries should use diff_summary from metadata
        assert "2 files changed (+0/−0)" in result
        assert "2 files changed (+10/−1)" in result
