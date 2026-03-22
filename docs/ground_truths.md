# Ground Truths

Key findings and verified facts discovered during development.

## LLM Provider Configuration

- The `get_llm()` factory in `src/orchestrator/nodes.py` supports both `ChatOpenAI` and `ChatOllama` via `configure_llm(provider, model)`.
- Default provider is **Ollama** with model `qwen2.5-coder:32b`. OpenAI remains available via `--provider openai`.
- The lazy singleton pattern is preserved: tests monkeypatch `get_llm()` and never instantiate a real LLM.
- `configure_llm()` resets the cached `_llm` instance so a new provider/model takes effect on the next `get_llm()` call.

## End-to-End Testing with Ollama

- Tests marked `@pytest.mark.ollama` are auto-skipped when Ollama is not reachable (health check: `GET localhost:11434/api/tags`).
- The conftest also verifies the required model is pulled before running.
- E2E test timeout is 30 minutes (`@pytest.mark.timeout(1800)`) to accommodate large model inference.

## Schrödinger Square-Well Example

- The instruction file lives at `examples/square_well_schrodinger.md`.
- It specifies a finite symmetric square-well in atomic units: $\hbar=1$, $m=1$, $V_0=50$, $a=1$.
- Bound states are found by solving transcendental equations (even: $k\tan(ka)=\kappa$, odd: $-k\cot(ka)=\kappa$).
- The expected package structure: `solver.py`, `wavefunctions.py`, `cli.py` + tests.

## Code Block Parsing (`_parse_code_blocks`)

- **Root cause of empty coder output (2025-07):** The parser required `"/" in line` to recognise a fenced block as a file path. This silently dropped root-level files like `pyproject.toml`, `conftest.py`, and `__init__.py`. Local LLMs (e.g. nemotron-3-super) often produce such blocks.
- **Fix:** Replaced the `/` check with `_looks_like_filepath()` which accepts info-strings containing `/` **or** ending with a recognised file extension (`.py`, `.toml`, `.yaml`, etc.). Language tags like `python`, `bash`, `json` are correctly rejected.
- **Raw response logging:** The `coder_raw_response` state field now stores the LLM's full text output. The `ChatLogger` writes it to the chat log when `code_drafts` is empty, enabling post-run diagnosis.
- **Coder prompt:** Updated `prompts/coder.md` with multiple examples (root-level files, nested paths) and explicit "WRONG" format examples to reduce LLM formatting errors.

## Verifier Sandbox Environment

- **Root cause of `ModuleNotFoundError` in sandbox:** The original verifier generated a conftest.py that only added the temp root directory to `sys.path`. When the coder used a `src/` layout (e.g. `src/square_well/solver.py`), tests importing `from square_well.solver import …` failed because `src/` was not on the path.
- **Fix (`_prepare_sandbox`):** The verifier now (1) tries `pip install -e .` if a `pyproject.toml` is present, (2) falls back to generating a smarter conftest that adds both the root and `src/` directories to `sys.path`, and (3) never overwrites a coder-generated `conftest.py`.
- **Package directory detection:** The sandbox walks the file tree for `__init__.py` files, walks up to the top-level package, and adds the parent directory to `sys.path`. This handles nested packages like `src/pkg/sub/module.py`.

## Python Testing & Packaging Skill

- A new skill `skills/python-testing/SKILL.md` teaches the planner and coder to:
  - Prefer **flat layout** (package at project root, not inside `src/`) for sandbox compatibility.
  - Always generate `conftest.py` and `pyproject.toml`.
  - Import from the package name (`from pkg.mod import …`), never from `src.pkg.mod`.
  - Include `__init__.py` in every package directory.
- The skill matches on common keywords: "Python", "pytest", "testing", "packaging", "imports", "conftest", "project", "structure".
- The coder prompt (`prompts/coder.md`) was updated with explicit flat-layout project structure rules and correct/wrong import examples.

## Multi-Phase Planner

- **Why:** Large tasks overwhelm the coder LLM when the full plan is presented at once. Splitting into phases lets the coder focus on one deliverable at a time.
- **Phase parsing:** `_parse_plan_phases()` in `nodes.py` uses a regex (`^## Phase \d+: Title`) to extract phases from the planner's markdown output. Each phase has an `id`, `title`, `description`, `status`, and `files` list. Falls back to a single "Implementation" phase when no headers are found.
- **Planner prompt** (`prompts/planner.md`) instructs the LLM to produce `## Phase N: Title` sections with `Files:` lines listing the files for each phase. Revision mode (when reflection exists) focuses only on the current phase.
- **Plan artifact:** `_write_plan_artifact()` writes `plan.md` to the working directory with checkbox status markers: `[x]` completed, `[~]` in-progress, `[ ]` pending.
- **Graph routing:** `_should_continue()` in `cli.py` now has three-way routing: tests pass + more phases → `advance_phase`, tests pass + all done → `end`, tests fail → `reflect`.
- **`advance_phase` node:** Marks the current phase as completed, increments `current_phase`, sets `plan` to the next phase description, clears `reflection`, and rewrites the plan artifact.
- **Coder awareness:** The coder receives a phase context header ("Current Phase (N of M): Title"), sees existing files from prior phases, and merges new drafts into accumulated `code_drafts` (new files override old).
- **State fields:** `plan_phases: list[dict]` and `current_phase: int` added to `ScientificState`.
- **Test count:** 11 new unit tests covering phase parsing, plan artifact writing, planner phasing, advance_phase node, and coder phase context. Total: 69 tests passing.

## Anti-Loop: Coder Error Feedback & Stuck-Loop Detection

- **Problem observed:** In a live run (schrodinger-qwen, Qwen 2.5-Coder 32B), the agent got stuck for 6+ iterations on the exact same import mismatch error. The reflector correctly identified the fix every time, the planner revised the plan accordingly, but the coder kept making the same mistake because it never saw the error feedback directly.
- **Root cause:** The graph flow is `reflector → planner → coder`, but the coder only received `state['plan']`. It never saw `state['reflection']` or `state['test_logs']`. The error feedback was indirect (through the planner's revised plan), which weaker models like Qwen 32B didn't follow closely enough.
- **Fix 1 — Direct error feedback to coder:** The `coder()` node now includes `state['reflection']` (under "## Previous error analysis") and `state['test_logs']` (under "## Test failures to fix") in its user prompt when they are non-empty. This gives the coder direct awareness of what went wrong.
- **Fix 2 — Stuck-loop detection:** The `verifier()` node now computes a fingerprint of the error output (sorted, joined log strings) and compares it with `_prev_error_fingerprint` in state. If the fingerprint matches, `_error_repeat_count` is incremented; otherwise it resets to 1 (or 0 on passing tests). After 3+ repeated identical errors, the coder receives a "⚠️ CRITICAL" escalation header telling it to make DIFFERENT choices.
- **Fix 3 — Ground truth deduplication:** The `reflector()` node now checks for exact duplicates before appending findings to `ground_truth`. In the stuck run, the same 3 findings were repeated 5+ times, wasting context tokens.
- **State fields added:** `_prev_error_fingerprint: str` and `_error_repeat_count: int` in `ScientificState`.
- **Test count:** 10 new tests (4 coder error context, 4 stuck-loop detection, 2 ground truth dedup). Total: 94 unit tests passing.

## Write Mode: ModuleNotFoundError on First Iteration

- **Problem observed:** Every run of the workflow (in write mode, with `--output-dir`) hit `ModuleNotFoundError: No module named 'square_well'` on the first iteration. The coder generates a `src/` layout but `pytest` can't find the package because nothing sets up `sys.path` or runs `pip install -e .`.
- **Root cause 1 — Write mode skips import setup:** The verifier's `_prepare_sandbox()` function (which handles `pip install -e .` and conftest generation) only runs in **sandbox mode** (temp directory). In **write mode** (`--output-dir`), pytest was invoked directly with no import plumbing. The coder's `src/` layout means `from square_well.solver import …` always fails.
- **Fix 1 — `_ensure_importable()`:** New function called in write mode before running pytest. Tries `pip install -e .` if pyproject.toml exists, falls back to generating a conftest.py with `sys.path` entries for `src/` and package parent directories.
- **Root cause 2 — Skills never loaded:** The `python-testing` skill (which teaches the coder flat layout, correct imports, etc.) was only loaded when the user passed `--skills skills/`. Most runs omitted this flag, so the skill never fired.
- **Fix 2 — Auto-load built-in skills:** The CLI now always includes the project's `skills/` directory (next to `src/`) in the skill search path. User-supplied `--skills` directories are additive. This ensures `python-testing` and other built-in skills always match when relevant.
- **Test count:** 4 new tests (3 `_ensure_importable`, 1 write-mode src-layout verifier). Total: 98 unit tests passing.

## Import Consistency: `ast.AnnAssign` (Typed Assignments)

- **Problem observed:** In a live run (schrodinger-qwen, Qwen 3.5 122B), the agent got stuck for 20+ iterations because `_check_import_consistency()` reported "Import mismatch: only defines: (nothing)" for a `constants.py` file that used `typing.Final` annotations (e.g. `HBAR: Final[float] = 1.0545718e-34`).
- **Root cause:** `_check_import_consistency()` only recognized `ast.Assign` nodes (plain assignments like `PI = 3.14`). Typed/annotated assignments (`HBAR: Final[float] = 1.0`) produce `ast.AnnAssign` nodes in the AST, which were silently ignored. The checker thought the module defined nothing, causing every import from it to fail the consistency check.
- **Fix:** Added `ast.AnnAssign` handling after the existing `ast.Assign` block: `elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name): names.add(node.target.id)`. Note: `ast.AnnAssign.target` is a single `ast.Name` (not a list like `ast.Assign.targets`).
- **Not changed:** `_extract_signatures()` was not modified — it extracts function/class signatures for coder context, and constants don't need signature extraction.
- **Test count:** 1 new test (`test_annotated_assignment_counted`). Total: 99 unit tests passing.

## Skip Planner on Revision Loops (reflector → coder)

- **Problem observed:** In a live run (schrodinger-qwen), the planner's revised output after a reflector analysis was generic and unhelpful. The reflector produced a detailed error analysis with specific code patches (e.g., "clamp k to `[0, sqrt(2*V0)]`"), but the planner replaced the original detailed phase description with a vague rewording ("implement the solver with guards against non-physical k values").
- **Root cause:** The planner in revision mode received only `task_description` + `reflection` — it never saw `state['plan']` (the current phase description), `state['test_logs']`, `state['code_drafts']`, or `state['ground_truth']`. So it regenerated the plan from scratch, producing a generic description that was strictly worse than the original. Meanwhile, the coder already received reflection + test_logs directly (from the anti-loop fix), making the planner's revision redundant.
- **Fix:** Changed the graph edge from `reflector → planner → coder` to `reflector → coder`. The planner now only runs once at the start (and implicitly when `advance_phase` transitions to a new phase). On error loops, the coder gets the original plan (unchanged) plus the reflector's detailed analysis and test logs.
- **Cleanup:** Removed the `is_revision` code path from `planner()`, removed "Previous error analysis" injection into planner prompt, removed revision instructions from `prompts/planner.md`, replaced `test_planner_revision_updates_current_phase_only` with `test_planner_ignores_reflection_context`.
- **Integration test fix:** The `_CODE_RESPONSE` and `bad_code` fixtures used `from src.math.factorial import factorial` (a `src/` layout with `src.` prefix imports). The `_check_import_consistency` function flags `src.` prefix imports as errors, causing the graph to loop infinitely. Fixed by switching to flat layout (`factorial.py`, `from factorial import factorial`).
- **Test count:** 1 replaced test, integration tests fixed (were broken by prior `src.` prefix detection change). Total: 102 tests passing (1 skipped).

## Test File Protection on Revision Iterations

- **Problem observed:** In a live run (schrodinger-qwen, Qwen 3.5 122B), the agent got stuck for 12+ iterations because the coder kept rewriting test files (`test_solver.py`) on every revision iteration. At iteration 5, the coder changed test expectations to physically impossible values (e.g., bound-state energy E = −791.5 for V₀ = 500, violating the constraint −V₀ < E < 0). No solver can satisfy impossible expectations, so all subsequent iterations failed.
- **Root cause:** The `coder()` node's merge logic (`merged_drafts.update(new_drafts)`) allowed new drafts to overwrite *any* existing file, including test files. On revision iterations (reflector → coder), the coder regenerated both implementation AND test files, silently replacing test assertions that served as the specification.
- **Fix 1 — Code-level protection:** On revision iterations (when `state['reflection']` is set and `code_drafts` already exist), the coder now filters out test files from `new_drafts` before merging. Files whose basename starts with `test_` are protected and cannot be overwritten by the coder during error-fix loops.
- **Fix 2 — Prompt-level guidance:** Added instructions to `prompts/coder.md` telling the coder: "When fixing test failures, fix the IMPLEMENTATION code, not the tests. Tests are the specification. Do NOT regenerate test files."
- **Design rationale:** Tests are generated once (iteration 1) as the specification. During error loops, only the implementation should change. If a test expectation is truly wrong, the reflector must flag it explicitly (future enhancement).
- **Test count:** 2 new tests (`test_revision_protects_existing_test_files`, `test_first_iteration_allows_test_files`). Total: 104 tests passing (1 skipped).
