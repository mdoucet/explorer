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

## File-tree
- `src/math/factorial.py` — recursive factorial implementation
- `tests/test_factorial.py` — unit tests
"""

_CODE_RESPONSE = """\
```src/math/factorial.py
def factorial(n: int) -> int:
    \"\"\"Compute $n!$ recursively.\"\"\"
    return 1 if n <= 1 else n * factorial(n - 1)
```

```tests/test_factorial.py
from src.math.factorial import factorial

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
        # The planner is called once, coder once. No reflector needed.
        fake = _SequenceLLM([_PLAN_RESPONSE, _CODE_RESPONSE])
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

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
        assert "src/math/factorial.py" in final["code_drafts"]


class TestLoopReflectsOnFailure:
    """If the first code draft fails, the graph should reflect and retry."""

    def test_reflects_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        reflection = "The function returns a hardcoded -1 instead of computing the factorial."
        findings = "- The coder produced a stub instead of real logic"

        # Sequence: plan → bad code → reflection → findings → revised plan → good code
        fake = _SequenceLLM([
            _PLAN_RESPONSE,   # planner #1
            bad_code,          # coder #1 (will fail)
            reflection,        # reflector (reflection)
            findings,          # reflector (findings extraction)
            _PLAN_RESPONSE,   # planner #2
            _CODE_RESPONSE,   # coder #2 (will pass)
        ])
        monkeypatch.setattr("orchestrator.nodes.get_llm", lambda: fake)

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
