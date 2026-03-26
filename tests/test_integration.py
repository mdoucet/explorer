"""Integration test: run the full Scientific Loop graph on a toy problem.

The LLM is fully mocked so the test is deterministic and offline.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from orchestrator.state import ScientificState


# ---------------------------------------------------------------------------
# Fake LLM responses for a 1-iteration success scenario
# ---------------------------------------------------------------------------

_PLAN_RESPONSE = """\
## Mathematical Specification
```latex
n! = \\prod_{k=1}^{n} k
```

## Phase 1: Factorial
Implement a recursive factorial function with tests.
Files: factorial.py, tests/test_factorial.py
"""

_CODE_RESPONSE = """\
```factorial.py
def factorial(n: int) -> int:
    \"\"\"Compute $n!$ recursively.\"\"\"
    return 1 if n <= 1 else n * factorial(n - 1)
```

```tests/test_factorial.py
from factorial import factorial

def test_base():
    assert factorial(0) == 1

def test_five():
    assert factorial(5) == 120
```
"""


class _SequenceLLM:
    """Returns pre-canned responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0

    def invoke(self, _messages: Any) -> Any:
        msg = MagicMock()
        msg.content = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return msg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullLoopSuccess:
    """The graph should converge in 1 iteration when code passes tests."""

    def test_single_iteration_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The planner is called once, coder once, triage once. No reflector needed.
        fake = _SequenceLLM([_PLAN_RESPONSE, _CODE_RESPONSE, "LGTM"])
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        from src.cli import build_graph

        graph = build_graph()
        app = graph.compile()

        initial: ScientificState = {
            "task_description": "Implement a factorial function",
            "mathematical_constants": {},
            "plan": "",
            "code_drafts": {},
            "test_logs": [],
            "reflection": "",
            "iteration_count": 0,
            "ground_truth": [],
        }

        final = app.invoke(initial)

        assert final["iteration_count"] == 1
        assert final["test_logs"] == []
        assert "factorial.py" in final["code_drafts"]


class TestLoopReflectsOnFailure:
    """If the first code draft fails, the graph should reflect and retry."""

    def test_reflects_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad_code = """\
```factorial.py
def factorial(n: int) -> int:
    return -1
```

```tests/test_factorial.py
from factorial import factorial

def test_five():
    assert factorial(5) == 120
```
"""
        reflection = (
            "The function returns a hardcoded -1 instead of computing the factorial.\n\n"
            "## Key Findings\n"
            "- The coder produced a stub instead of real logic\n\n"
            "## Action\n"
            "RETRY"
        )

        # Sequence: plan → bad code → triage → reflection (with findings) → good code → triage
        # (reflector routes directly to coder, skipping the planner on revision)
        fake = _SequenceLLM([
            _PLAN_RESPONSE,   # planner #1
            bad_code,          # coder #1 (will fail)
            "LGTM",           # triage #1 (falls through to pytest)
            reflection,        # reflector (single call: analysis + findings)
            _CODE_RESPONSE,   # coder #2 (will pass)
            "LGTM",           # triage #2 (falls through to pytest)
        ])
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        from src.cli import build_graph

        graph = build_graph()
        app = graph.compile()

        initial: ScientificState = {
            "task_description": "Implement a factorial function",
            "mathematical_constants": {},
            "plan": "",
            "code_drafts": {},
            "test_logs": [],
            "reflection": "",
            "iteration_count": 0,
            "ground_truth": [],
        }

        final = app.invoke(initial)

        assert final["iteration_count"] == 2
        assert final["test_logs"] == []


class TestResumeAfterMaxIterations:
    """If a run hits max-iterations with failing tests, --resume should
    detect the failing state, reset the checkpoint, and re-enter the graph
    so the agent can keep iterating."""

    def test_resume_restarts_from_saved_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path,
    ) -> None:
        from orchestrator.state import make_checkpointer
        from src.cli import build_graph

        db_path = str(tmp_path / "test.sqlite")
        thread_id = "resume-test"

        # --- First run: max_iterations=1, code will fail ---
        bad_code = """\
```src/math/factorial.py
def factorial(n: int) -> int:
    return -1
```

```tests/test_factorial.py
from src.math.factorial import factorial

def test_five():
    assert factorial(5) == 120
```
"""
        fake = _SequenceLLM([_PLAN_RESPONSE, bad_code, "LGTM"])
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake)

        checkpointer = make_checkpointer(db_path)
        graph = build_graph(max_iterations=1)
        app = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 2**31}

        initial: ScientificState = {
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

        run1_final: dict = {}
        for event in app.stream(initial, config=config):
            for _node, update in event.items():
                run1_final.update(update)

        # First run should have stopped at iteration 1 with failing tests
        assert run1_final.get("iteration_count") == 1
        assert run1_final.get("test_logs"), "Expected failing test_logs"
        assert run1_final.get("code_drafts"), "Expected code_drafts from first run"

        # --- Second run (resume): read full state, delete checkpoint, restart ---
        fake2 = _SequenceLLM([_PLAN_RESPONSE, _CODE_RESPONSE, "LGTM"])
        monkeypatch.setattr("orchestrator.nodes._shared.get_llm", lambda: fake2)

        checkpointer2 = make_checkpointer(db_path)
        graph2 = build_graph(max_iterations=0)
        app2 = graph2.compile(checkpointer=checkpointer2)

        # Read the saved state (simulates what --resume does)
        saved = app2.get_state(config)
        assert saved and saved.values
        input_state = dict(saved.values)

        # Verify full state was preserved
        assert input_state.get("code_drafts"), "Saved state should have code_drafts"
        assert input_state.get("plan"), "Saved state should have plan"

        # Delete checkpoint and restart with full saved state
        checkpointer2.delete_thread(thread_id)

        run2_final: dict = {}
        for event in app2.stream(input_state, config=config):
            for _node, update in event.items():
                run2_final.update(update)

        # Second run should succeed — tests now pass
        assert run2_final.get("test_logs") == []
        assert run2_final.get("iteration_count", 0) >= 2
