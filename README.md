# Explorer

A LangGraph-based agentic framework for autonomous scientific Python development. Explorer follows a **Scientific Loop** — plan, code, test, reflect — modeled after the scientific method to produce verified, correct code with minimal human intervention.

## Architecture

```
┌──────────┐     ┌────────┐     ┌──────────┐
│ Planner  │────▶│ Coder  │────▶│ Verifier │
└──────────┘     └────────┘     └──────────┘
     ▲                               │
     │         ┌───────────┐         │
     └─────────│ Reflector │◀────────┘
               └───────────┘    (on failure)
```

| Node | Role |
|---|---|
| **Planner** | Analyses the task and produces a LaTeX-formatted mathematical spec + file-tree plan |
| **Coder** | Generates Python source files from the plan (type-hinted, with LaTeX docstrings) |
| **Verifier** | Runs `pytest` against the generated code — in a sandboxed temp dir (default) or directly in the output directory (write mode) |
| **Reflector** | Analyses test failures and feeds error context back to the Planner |

The loop terminates when all tests pass or after a configurable number of iterations (default: 50).

## Requirements

- Python 3.9+
- An LLM endpoint — either:
  - **Ollama** running locally (default), or
  - An **OpenAI** API key

## Installation

```bash
# Clone the repository
git clone <repo-url> && cd explorer

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the project and dev dependencies
pip install -e ".[dev]"
```

## Configuration

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

The `.env` file controls which LLM backend Explorer uses:

| Variable | Description | Default |
|---|---|---|
| `EXPLORER_LLM_PROVIDER` | `ollama` or `openai` | `ollama` |
| `EXPLORER_LLM_MODEL` | Model name for the provider | `qwen2.5-coder:32b` |
| `EXPLORER_LLM_BASE_URL` | API endpoint URL | `http://localhost:11434` |
| `EXPLORER_LLM_TEMPERATURE` | Sampling temperature (0.0 = deterministic) | `0.0` |
| `OPENAI_API_KEY` | Required only when `EXPLORER_LLM_PROVIDER=openai` | — |

All values can also be overridden via CLI flags (see below).

### Verify your LLM connection

```bash
explorer check-llm
```

This sends a test prompt to the configured endpoint and prints the response or the error.

## Usage

### How the agent works

1. You provide a **task description** — either as inline text or a Markdown file.
2. The **Planner** reads the description and produces a mathematical spec and a file-tree plan (e.g. `src/solver.py`, `tests/test_solver.py`).
3. The **Coder** generates Python source files following the plan. Each file is kept in a `code_drafts` dictionary stored in memory as `{filepath: source_code}`. In write mode (`-o`), the files are also written to the output directory.
4. The **Verifier** runs `pytest`. In **sandbox mode** (default) it writes files to a temporary directory and auto-cleans it. In **write mode** it runs `pytest` directly inside the output directory.
5. If tests fail, the **Reflector** analyses the errors and the loop retries from the Planner.
6. When all tests pass (or the iteration cap is reached), Explorer writes a `ground_truth.md` file in your current directory summarising the run.

> **Sandbox mode (default)**: Generated code lives only in a temp directory that is cleaned up after each verification step. The final code is available in the output state's `code_drafts` field. The `ground_truth.md` file is the only artifact written to your working directory.
>
> **Write mode** (`--output-dir`): Generated code is written directly to the specified directory and `pytest` runs there against the real files.

### Quick start — inline task

```bash
explorer run --task "Implement a recursive factorial function with tests"
```

### Using a Markdown file for complex projects

For larger projects, write a detailed task description in a Markdown file and pass it with `--task-file` / `-f`:

```bash
explorer run --task-file examples/square_well_schrodinger.md
```

A good task file should include:

- **Problem statement** — what the code should do
- **Mathematical formulation** — equations, constants, boundary conditions
- **Package structure** — which files to create (`src/`, `tests/`, `pyproject.toml`, CLI, etc.)
- **Acceptance criteria** — what "passing tests" means (tolerances, edge cases, expected outputs)

See [`examples/square_well_schrodinger.md`](examples/square_well_schrodinger.md) for a complete example.

### Working on an existing repository

Use **write mode** (`--output-dir` / `-o`) to have Explorer write generated files directly into a target directory and run `pytest` there:

```bash
# Create a new project from scratch
explorer run -f task.md -o /path/to/my-project

# Work on an existing repo — Explorer can create new files and overwrite existing ones
explorer run -f task.md -o /path/to/existing-repo --max-iterations 10
```

> **Safety tip**: When pointing at an existing git repo, commit or stash your work first — Explorer may overwrite files.

Write a task description that references your existing structure:

```markdown
# Add a caching layer to the data loader

The project already has `src/data/loader.py` with a `load_dataset()` function.
Add an LRU cache using `functools.lru_cache` and create tests in
`tests/test_loader_cache.py`.
```

Path traversal is blocked — the agent cannot write files outside the output directory.

### Skills

Skills are reusable packages of domain knowledge that the agent can draw on
when they match the task.  They follow the
[Agent Skills specification](https://agentskills.io/specification) — each
skill lives in its own directory with a `SKILL.md` file containing YAML
frontmatter and Markdown instructions.

```
skills/
├── quantum-mechanics/
│   └── SKILL.md
└── numerical-optimization/
    └── SKILL.md
```

Point Explorer at your skills directory with `--skills` / `-s`:

```bash
explorer run -f task.md -s skills/
```

The agent uses **progressive disclosure**: it reads only the `description`
from each `SKILL.md` frontmatter at startup, then loads the full content of
skills that match the task.  Matched skills are injected into the Planner and
Coder prompts as additional context.

#### Creating a skill

Create a directory with a `SKILL.md` file:

```markdown
---
name: my-domain
description: Keywords and phrases that describe when this skill should activate
---

# My Domain

## Overview

When to use this skill.

## Instructions

Domain-specific guidance, equations, conventions, and testing tips.
```

Key points:
- The `description` field is used for matching — include relevant keywords.
- The `name` defaults to the directory name if omitted.
- You can pass multiple `--skills` flags to layer skills from different sources
  (last-wins precedence for duplicate names).

See `skills/quantum-mechanics/SKILL.md` for a complete example.

### CLI Reference

```
explorer [OPTIONS] COMMAND [ARGS]...

Commands:
  run        Run the Scientific Loop agent on a task
  check-llm  Check connectivity to the configured LLM endpoint
```

#### `explorer run`

| Flag | Description | Default |
|---|---|---|
| `-t`, `--task` | Inline natural-language task description | — |
| `-f`, `--task-file` | Path to a Markdown file with the task description | — |
| `-o`, `--output-dir` | Write generated files to this directory (write mode) | — (sandbox) |
| `-s`, `--skills` | Directories containing skill folders (repeatable) | — |
| `--thread-id` | Thread ID for checkpoint persistence | `default` |
| `--db` | SQLite file for checkpointing | `checkpoints.sqlite` |
| `--max-iterations` | Maximum plan→code→verify cycles | `50` |
| `--provider` | LLM provider (`ollama` or `openai`) | env / `ollama` |
| `--model` | Model name | env / `qwen2.5-coder:32b` |
| `--base-url` | LLM API base URL | env |
| `--temperature` | Sampling temperature | env / `0.0` |

Provide either `--task` or `--task-file`, not both.

#### `explorer check-llm`

| Flag | Description | Default |
|---|---|---|
| `--provider` | LLM provider (`ollama` or `openai`) | env / `ollama` |
| `--model` | Model name | env / `qwen2.5-coder:32b` |
| `--base-url` | LLM API base URL | env |

### Checkpointing and resumption

Explorer persists graph state to a local SQLite database. If a run is interrupted, re-run the same command with the same `--thread-id` to resume from the last checkpoint:

```bash
explorer run -f task.md --thread-id physics_001
# (interrupted)
explorer run -f task.md --thread-id physics_001   # resumes
```

### Output artifacts

| File | Description |
|---|---|
| `ground_truth.md` | Key findings and learnings from the run (written to cwd) |
| `checkpoints.sqlite` | Persistent graph state for resumption |

## Project Structure

```
explorer/
├── .env.example                       # LLM configuration template
├── .github/copilot-instructions.md    # Coding standards & agent autonomy rules
├── pyproject.toml                     # Project metadata & dependencies
├── docs/plan.md                       # Original design plan
├── examples/
│   └── square_well_schrodinger.md     # Example task: quantum square-well solver
├── prompts/                           # Node system prompts (configurable)
├── skills/                            # Reusable agent skills
│   ├── quantum-mechanics/
│   │   └── SKILL.md
│   └── numerical-optimization/
│       └── SKILL.md
├── src/
│   ├── cli.py                         # StateGraph assembly, conditional edges, CLI
│   └── orchestrator/
│       ├── __init__.py
│       ├── state.py                   # ScientificState TypedDict + SqliteSaver
│       ├── nodes.py                   # Planner, Coder, Verifier, Reflector
│       └── skills.py                  # Skill loader, matcher, formatter
└── tests/
    ├── conftest.py                    # Ollama availability fixture
    ├── test_nodes.py                  # Unit tests per node (mocked LLM)
    ├── test_skills.py                 # Skill loading and matching tests
    ├── test_integration.py            # Full-loop graph tests (mocked LLM)
    └── test_e2e_schrodinger.py        # End-to-end test with real Ollama
```

## Running Tests

```bash
# Unit and integration tests (mocked LLM, runs offline)
PYTHONPATH=src python -m pytest tests/ -v -m "not ollama"

# End-to-end test (requires Ollama running with the configured model)
PYTHONPATH=src python -m pytest tests/ -v -m ollama
```

## Design Decisions

- **JAX + NumPy** as numerical backends — JAX preferred for differentiability in physics engines
- **SqliteSaver** checkpointing — simplest local persistence; allows interrupting and resuming long runs
- **subprocess-based pytest** in the Verifier — sandboxed from the agent process
- **Skills (Agent Skills spec)** — progressive disclosure of domain knowledge; only matched skills are loaded
- **Max 50 iterations** — prevents runaway loops
- **Type hints everywhere** — Python 3.12+ style annotations (via `from __future__ import annotations`)

## License

MIT
