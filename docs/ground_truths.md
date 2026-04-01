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

## Syntax Validation Before Pytest

- **Problem observed:** In a live run (schrodinger-qwen, nemotron-3-super:120b), the LLM generated `test_one_bound_state()` with an unindented docstring. Pytest failed at *collection* time with `IndentationError: expected an indented block`. The reflector saw the raw pytest traceback but the error was buried — the coder kept regenerating the same broken pattern.
- **Root cause:** No syntax validation between code generation and pytest execution. Syntax errors only surfaced as cryptic pytest collection failures.
- **Fix:** Added `_check_syntax()` in `nodes.py` that `ast.parse()`s every `.py` file in code_drafts *before* running pytest. Returns clear `"SyntaxError in {filepath} line {lineno}: {msg}"` messages that the reflector can act on directly.
- **Test count:** 5 new tests in `TestCheckSyntax`. Total: 113 tests passing (1 skipped).

## Stale File Cleanup in Write Mode

- **Problem observed:** In the same live run, stale `.py` files from a previous run persisted in the output directory. When the new run's verifier ran pytest, it tested stale files alongside new ones, causing confusing failures.
- **Fix:** Added `_warn_stale_files()` that scans the output directory for `.py` files not present in the current `code_drafts` and removes them. `__init__.py` files are preserved. Called in verifier's write-mode path before pytest.
- **Test count:** 4 new tests in `TestWarnStaleFiles`.

## Monitor Pattern Improvements

- **Problem observed:** The monitor script (`scripts/monitor_run.sh`) only grepped for `FAILED|Import mismatch|only defines|ModuleNotFoundError|No code drafts|0 passed`. It missed `collected 0 items / 1 error` (pytest collection failures), `SyntaxError`, and `IndentationError` — reporting "0 test(s) failed" with empty error text.
- **Fix:** Extended grep patterns to include `collected 0 items|SyntaxError|IndentationError`. Added specific log categories: `SYNTAX:` for syntax errors, `BROKEN:` for collection failures.

## Reflector Produces Code-Level Patches

- **Problem observed:** In the schrodinger-qwen run (nemotron-3-super:120b), the reflector gave excellent *physics* analysis ("filter spurious eigenvalues", "use analytic branch structure") but the coder couldn't translate abstract advice into working code. The same 4-failure pattern persisted for 5+ iterations despite correct diagnosis.
- **Root cause:** The reflector prompt was minimal ("analyse the root cause and suggest concrete fixes. Be concise.") — it didn't require *code*. The coder LLM needed exact Python patches, not physics prose.
- **Fix:** Enhanced `prompts/reflector.md` to require concrete code snippets with exact variable names from the current code. Added instruction to propose a DIFFERENT algorithmic approach when the same error repeats.

## Ground Truth Finding Deduplication

- **Problem observed:** After 5 iterations on schrodinger-qwen, the reflector accumulated 16 ground_truth findings, many near-duplicates of "spurious shallow negative energy" rephrased in different ways. This noise diluted the signal.
- **Root cause:** The findings prompt didn't see what already existed. The LLM kept re-discovering the same insight.
- **Fix 1 — Prompt-level:** Updated `prompts/findings.md` to say "Do NOT repeat or rephrase findings that already exist. If the existing findings already cover the insight, reply NONE."
- **Fix 2 — Code-level:** Modified `reflector()` in nodes.py to pass existing findings into the findings extraction LLM call as a `## Existing findings (do NOT repeat these)` section.
- **Test count:** 1 new test (`test_existing_findings_passed_to_llm`). Total: 114 tests passing (1 skipped).

## Numerical Recipe Skills for Transcendental Equations

- **Problem observed:** The `numerical-optimization` skill told the coder to "use brentq for bracketed root finding" but didn't warn about the trap: `tan(x)` has poles that create false sign changes. The coder fell into this trap on every iteration of the square-well task.
- **Fix:** Added a "Transcendental equations with periodic singularities" recipe to `skills/numerical-optimization/SKILL.md` with: (a) explicit warning against blind sign-change search, (b) analytic bracket formulas for even/odd parity, (c) complete working code example using `brentq` on individual branches, (d) validation criteria (residual check + pole proximity check).

## Run Comparison: Stubs-First Phase Decomposition (March 2025)

- **Observation:** Two runs of the same Schrödinger finite-well task with the same model (nemotron-3-super:120b) had vastly different outcomes based on how the planner decomposed phases.
- **Successful run (~/git/schrodinger):** Phase 1 = "Project scaffolding" with stub implementations (`pass` bodies, `assert True` tests). Passed after 5 iterations (all failures were import/path issues). Phases 2–4 each passed on the **first try**. Used `brentq` on transcendental equations for the solver — machine-precision accuracy.
- **Stuck run (~/git/schrodinger-qwen):** Phase 1 = "Project setup AND even-parity solver." The hardest physics problem was the first hurdle. 10+ iterations stuck on ~0.9% numerical error because the coder chose finite-difference Hamiltonian diagonalization (`eigh`) instead of `brentq` on transcendental equations. The matrix approach has inherent discretization error that can't reach `rtol=1e-6` without enormous grids.
- **Key insight 1:** "Stubs first" phase decomposition separates infrastructure bugs (imports, paths, test runner) from logic bugs (physics, algorithms). This allows the agent to build a working scaffold before tackling hard problems.
- **Key insight 2:** The skill recipe for transcendental equations was injected into the coder prompt but the model chose a different algorithm anyway. Skills provide guidance but cannot force algorithm choice.
- **Key insight 3:** The reflector kept proposing parameter tweaks (grid size, box extent) instead of suggesting a fundamentally different algorithm, even with the enhanced "propose a DIFFERENT approach" prompt.
- **Key insight 4:** The old run had `reflector → planner → coder` flow (planner in revision loops) while the new run uses `reflector → coder` (planner skipped). The planner's involvement may have helped steer better algorithm choices.
- **Actionable:** Consider adding planner prompt guidance to always make Phase 1 a scaffolding-only phase with stub implementations. Consider re-involving the planner in revision loops when the reflector detects algorithmic (not just parametric) failures.

## Implemented Fixes from Run Comparison (March 2025)

Three changes were implemented based on the run comparison findings above:

### 1. Stubs-First Planner Guidance
- **File:** `prompts/planner.md`
- **Change:** Added explicit rule: "Phase 1 MUST be scaffolding only" — stubs with `pass` bodies, trivial `assert True` tests, `pyproject.toml` with dependencies. Real implementations start in Phase 2+.
- **Rationale:** The successful run's Phase 1 was trivially passable (stubs only), while the stuck run tried to implement the solver in Phase 1.

### 2. Re-engage Planner on Algorithmic Failures
- **Files:** `src/cli.py`, `src/orchestrator/nodes.py`, `src/orchestrator/state.py`
- **Change:** After the reflector, if the same error has repeated ≥3 consecutive times (`REPLAN_THRESHOLD`), route back to the planner instead of the coder. The planner revises only the current phase's description (not the entire plan) and resets the error counter. Replanning is capped at `MAX_REPLANS=2` per phase to prevent infinite loops.
- **New state field:** `_replan_count` tracks replans per phase, reset on `advance_phase`.
- **Graph flow change:** `reflector → (conditional) → coder | planner`
- **Rationale:** The stuck run's reflector kept proposing parameter tweaks; the planner can reason at a higher level about algorithmic approach.

### 3. MUST-USE Skill Directives
- **File:** `src/orchestrator/skills.py`
- **Change:** `format_skills_context()` now detects `## Recipe:` sections in skills (via regex) and prepends a "⚠️ MANDATORY" directive telling the LLM it MUST follow the recipe's algorithm. Uses `re.search(r'#{2}\s+Recipe\b', ..., re.MULTILINE | re.IGNORECASE)` for robust matching.
- **Rationale:** The skill recipe for transcendental equations was injected but the model chose a different algorithm anyway.

### Test count: 13 new tests. Total: 127 tests passing (1 skipped).

## Error Fingerprint Timing Sensitivity (March 2025)

- **Problem observed:** In the schrodinger-qwen run (nemotron-3-super:120b), the replan threshold (`REPLAN_THRESHOLD=3`) was **never reached** despite identical logical errors repeating for 6+ iterations. `_error_repeat_count` stayed at 1 every iteration.
- **Root cause:** The verifier's error fingerprint is computed from sorted pytest output lines. Pytest appends timing information (e.g., `in 0.41s`, `in 0.49s`) that varies slightly each run. Since the raw output was fingerprinted without normalization, semantically identical errors produced different fingerprints, and the repeat counter reset to 1 every time.
- **Fix:** Added `_normalize_pytest_output()` in `nodes.py` that strips timing via `re.compile(r" in \d+\.\d+s")` before fingerprinting. The verifier now normalizes all log lines before computing the fingerprint: `normalised = [_normalize_pytest_output(l) for l in logs]`.
- **Impact:** Without this fix, the replan routing (reflector → planner on repeated errors) could never trigger in practice. All three improvements from the run comparison (stubs-first, replan routing, MUST-USE skills) relied on the replan threshold working correctly.
- **Test count:** 4 new tests in `TestNormalizePytestOutput`.

## Coder Writes Stub Tests in Implementation Phases (March 2025)

- **Problem observed:** In the schrodinger-qwen run, Phase 1 (stubs-first) passed on iteration 1 as expected. But Phase 2's first coder pass wrote `assert len(energies) == 0` — a stub-level assertion — even though it implemented a real solver that returned 3 energies. The test failed with `assert 3 == 0` and the error persisted for 5+ iterations.
- **Root cause:** The coder carried over the Phase 1 mental model of "stub tests" into Phase 2. Since Phase 1 explicitly required `assert True` / trivial tests, the coder continued that pattern even when implementing real logic.
- **Fix:** Added instruction to `prompts/coder.md`: "When implementing a phase AFTER a scaffolding phase, your tests MUST validate the real behaviour you implemented — not stubs. Replace any placeholder assertions (`assert True`, `assert len(...) == 0`) from the scaffolding phase with tests that verify actual computed results."
- **Interaction with test protection:** The existing test-file protection (revision iterations cannot overwrite `test_*` files) means the stub test from iteration 1 becomes locked in on subsequent iterations. Getting the test right on the first pass of each phase is critical.

### Combined test count: 4 new tests. Total: 131 tests passing (1 skipped).

## Test Protection Too Aggressive for New Phases (March 2025)

- **Problem observed:** In the schrodinger-qwen run (nemotron-3-super:120b), Phase 2's first coder pass wrote genuinely wrong tests: `assert len(energies) == 0` for a shallow well (physics guarantees ≥1 bound state in 1D), and two tests evaluating transcendental functions at ξ > C where `sqrt(C²-ξ²)` returns NaN. The reflector correctly diagnosed all three as test bugs and provided patches. But test protection (`is_revision and existing_drafts → protect test_*`) prevented the coder from applying those fixes on the next iteration.
- **Root cause:** Test protection was unconditional on any revision iteration (whenever `reflection` is set). This was designed to prevent the coder from weakening test expectations to match broken code. But it also blocked fixing genuinely wrong tests written on the first pass of a new phase.
- **Fix (v1 — superseded):** Test files were protected only when `_error_repeat_count >= 2`. But this used the *identical-error* repeat counter, which stays at 1 when the coder oscillates (different bugs each iteration). This meant tests were NEVER protected in practice.
- **Fix (v2 — current):** Introduced `_phase_error_count` in state — tracks total failures in the current phase regardless of whether errors are identical. Test files are protected when `_phase_error_count >= 2`. This gives the coder exactly one revision to fix bad tests, then locks them down unconditionally. Reset on `advance_phase`.
- **Key insight:** `_error_repeat_count` (consecutive identical fingerprints) is the wrong metric for test protection. The coder oscillates wildly — fixing one bug while introducing another — so errors are rarely identical. A monotonic counter (`_phase_error_count`) that only increments is the correct signal.
- **Rationale:** After `advance_phase`, the coder writes fresh code+tests. If those tests are wrong, one chance to self-correct is essential. After that, tests must be locked to prevent the coder from weakening them to match broken code.
- **Test count:** 2 new tests (`test_phase_error_count_increments_on_failure`, `test_phase_error_count_unchanged_on_pass`), 2 updated tests. Total: 134 tests passing (1 skipped).

## Quantum Mechanics Skill: Trig Function Pitfalls (March 2025)

- **Problem observed:** In schrodinger-qwen runs, the coder (nemotron-3-super:120b) repeatedly used `np.cot(x)` which doesn't exist in NumPy, causing `AttributeError`. The reflector would fix it to `np.cos(x)/np.sin(x)`, but the coder regressed on the next iteration.
- **Fix:** Added a comprehensive trig-function table to `skills/quantum-mechanics/SKILL.md` listing which functions exist (`np.sin`, `np.cos`, `np.tan`) and which must be composed (`cot`, `sec`, `csc`). Also added domain-validity guidance for `sqrt(C²-ξ²)` in transcendental equations.
- **Key insight:** Proactive guidance in skills prevents recurring LLM mistakes. Smaller models especially benefit from explicit "does NOT exist" warnings.

## Verifier: `sys.executable` Instead of Hard-Coded `"python"` (March 2025)

- **Problem observed:** `_run_pytest` and `_prepare_sandbox` used `["python", "-m", "pytest", ...]` and `["python", "-m", "pip", ...]`. On macOS without a bare `python` symlink (only `python3`), all verifier subprocess calls failed with `FileNotFoundError`.
- **Fix:** Replaced all 3 hard-coded `"python"` strings with `sys.executable`, which always resolves to the running interpreter.
- **Key insight:** Never hard-code `"python"` in subprocess calls — use `sys.executable` for portability.

## Replan Routing: Use `_phase_error_count` Not `_error_repeat_count` (March 2026)

- **Problem observed:** In the schrodinger-qwen run, Phase 5 ("Validation and Edge Cases") failed 11 straight iterations (7→17) with no replan ever triggered. The coder oscillated between different bugs (CLI option naming, numerical precision, edge cases), so `_error_repeat_count` (consecutive identical fingerprints) stayed at 1 — never reaching `REPLAN_THRESHOLD=3`.
- **Fix:** Changed `_after_reflector` to use `_phase_error_count` (total failures in the current phase) instead of `_error_repeat_count`. This is the same class of fix applied to test protection earlier.
- **Key insight:** `_error_repeat_count` is only useful for detecting the **same** bug recurring. For detecting the coder being **stuck** (regardless of whether it's the same bug), `_phase_error_count` is the correct metric. Both counters serve distinct purposes: `_error_repeat_count` → same-bug detection, `_phase_error_count` → total-stuck detection.

## Default `MAX_ITERATIONS` and `EXPLORER_MAX_ITERATIONS` Env Var (March 2026)

- **Problem observed:** `MAX_ITERATIONS=0` (unlimited) allowed the schrodinger-qwen run to loop 17 times with no cap, running for 8+ hours. The process eventually died but could have gone forever.
- **Fix:** Changed default to 20. Added `EXPLORER_MAX_ITERATIONS` env var (and `.env.example` entry) so it's configurable without code changes. CLI `--max-iterations 0` still allows explicit unlimited.
- **Test count:** 1 new test (`test_oscillating_errors_still_trigger_replan`), 5 updated tests. Total: 135 tests passing (1 skipped).

## Anti-Oscillation Improvements: 10-Point Overhaul (Based on GPT-4 Schrödinger Run Analysis)

- **Problem observed:** A GPT-4 run on the Schrödinger finite-well task (`~/git/schrodinger`) ran 62 iterations, got stuck at Phase 2, and never implemented any actual physics code. The coder oscillated between two failure modes: (1) LaTeX backslashes in docstrings causing SyntaxError, and (2) function signature mismatches where the coder changed function names/params each iteration. All 62 iterations produced `pass`-stub functions.
- **Root causes identified:** 10 systemic issues enabling the oscillation loop.

### 1. Full Source on Revision Iterations
- **File:** `src/orchestrator/nodes.py` (`coder()`)
- **Change:** On revision iterations, the coder now receives FULL source code of all existing files (not just function signatures). Clean files are marked with ✅ and "do NOT modify". Files with errors show full source so the coder can see what needs changing.
- **Rationale:** The coder was only seeing signatures, so it re-invented function bodies from scratch each iteration, losing all prior partial fixes.

### 2. Verified Fixes (Cumulative Constraints)
- **Files:** `src/orchestrator/nodes.py` (`coder()`, `verifier()`), `src/orchestrator/state.py`
- **Change:** When a previously-failing syntax check passes (e.g., raw docstring fix), the verifier records it as a permanent "verified fix" constraint. The coder receives these as "MANDATORY constraints" that must not be violated.
- **New state fields:** `verified_fixes: list[str]`, `_prev_syntax_file_count: int`
- **Rationale:** The coder kept re-introducing LaTeX backslashes that had been fixed in prior iterations because it had no memory of validated fixes.

### 3. Clean File Tracking
- **Files:** `src/orchestrator/nodes.py` (`verifier()`, `coder()`), `src/orchestrator/state.py`
- **Change:** The verifier identifies files with zero syntax/import errors and marks them as "clean". The coder is told not to modify files marked ✅.
- **New state field:** `clean_files: list[str]`
- **Rationale:** The coder was modifying working files, re-introducing bugs into previously clean code.

### 4. Replan Scope Locking
- **File:** `src/orchestrator/nodes.py` (`planner()`)
- **Change:** During replans, the original phase title is preserved (not replaced by the LLM's new title). The replan prompt explicitly says "Do NOT regress the scope" and "MUST remain [original title]".
- **Rationale:** Replans could regress from "Implement Solver" back to "Project Scaffolding", losing all implementation work.

### 5. Per-Phase Iteration Cap
- **File:** `src/cli.py`
- **Change:** Added `MAX_PHASE_ITERATIONS=8`. If a single phase exceeds this many iterations, the graph force-advances to the next phase instead of continuing to loop.
- **Rationale:** The GPT-4 run spent all 62 iterations on a single phase. Even if the phase isn't fully solved, moving forward allows progress on other phases that may be easier.

### 6. Exponential Backoff on Replans
- **File:** `src/cli.py` (`_after_reflector()`)
- **Change:** The replan threshold now uses exponential backoff: 3 errors for first replan, 6 for second, etc. (`REPLAN_THRESHOLD * 2^replan_count`).
- **Rationale:** Fixed threshold triggered replans too frequently, not giving the coder enough time to self-correct after each replan.

### 7. LaTeX Backslash Ban in Coder Prompt
- **File:** `prompts/coder.md`
- **Change:** Added explicit "Docstring safety" section: "NEVER use backslash-prefixed LaTeX commands in docstrings" with guidance to use raw docstrings or plain text.
- **Rationale:** LaTeX backslashes (`\psi`, `\hbar`, `\frac`) were the #1 cause of SyntaxError in the GPT-4 run, recurring in every iteration.

### 8. Source Code in Reflector Context
- **Files:** `src/orchestrator/nodes.py` (`reflector()`), `prompts/reflector.md`
- **Change:** The reflector now receives the source code of files with errors (non-clean files), not just test logs. The prompt instructs it to "reference EXACT line numbers and variable names from the code."
- **Rationale:** The reflector was giving generic debugging advice because it couldn't see the actual implementation code.

### 9. Best Iteration Tracking & Rollback
- **Files:** `src/orchestrator/nodes.py` (`verifier()`), `src/orchestrator/state.py`
- **Change:** The verifier tracks the code snapshot with the fewest errors (`best_code_drafts`, `best_error_count`). If error count worsens for 3+ iterations in a row while the coder oscillates, it reverts to the best-known snapshot.
- **Rationale:** The coder's oscillation pattern meant it would fix one bug but introduce two, then fix those but re-introduce the original. Rollback to the best known state prevents regression.

### 10. Streaming LLM Invocation with Retry
- **File:** `src/orchestrator/nodes.py` (`_invoke_llm()`)
- **Change:** Added `_invoke_llm()` helper that uses `llm.stream()` with retry on transient errors (incomplete chunked reads, connection drops, timeouts). All four LLM call sites (planner, coder, reflector x2) now use this helper. Falls back to `invoke()` for test mocks.
- **Rationale:** The server proxy at forerunner.ornl.gov dropped connections that didn't produce a token within ~60s. Large prompts (planner, coder) need streaming to avoid first-byte timeout.

### Test Impact
- 1 test updated (`test_replan_updates_current_phase_only` — title now preserved as "Solver" instead of "Solver v2")
- Total: 135 tests passing (1 skipped).

## GPT-4 Schrödinger Run: Force-Advance + Deprecated API Issues (March 2026)

- **Run:** `~/git/schrodinger`, GPT-4, 20 iterations, 4 phases. Completed ~7.5 minutes (16:13–16:21 UTC).
- **Outcome:** Phases 1–3 completed (1 passed, 2–3 force-advanced), Phase 4 hit MAX_ITERATIONS. Final: tests failing on `ImportError: cannot import name 'trapz' from 'numpy'`.

### Issue 1: Force-advance propagated broken code across phases
- **Problem:** Phase 2 (solver) was force-advanced after 8 iterations with 3 failing tests — wrong eigenvalues (actual: [-46.84, -35.48, -25.0] vs expected: [-43.32, -17.65, -2.09]), phantom bound states, and near-pole handling. Phase 3 (wavefunctions) inherited this broken solver AND introduced its own `trapz` import bug. Phase 3 was also force-advanced after 8 iterations. Phase 4 (CLI) inherited BOTH bugs — couldn't even collect tests.
- **Root cause:** `_should_continue()` force-advanced on `_phase_iteration_count >= MAX_PHASE_ITERATIONS` regardless of error severity. Collection errors (0 tests ran) indicate fundamentally broken code that can't improve by simply moving to the next phase.
- **Fix:** Added `_collection_error` flag to state. The verifier detects collection errors (`"collected 0 items"` or `"error during collection"` in pytest output). The force-advance logic in `_should_continue()` now blocks force-advance when `_collection_error` is True — the agent must fix the import/collection error before progressing.
- **New state field:** `_collection_error: bool`, reset on `advance_phase`.

### Issue 2: `numpy.trapz` and `scipy.integrate.trapz` both removed
- **Problem:** The coder used `from numpy import trapz` (removed in NumPy 2.0). The reflector suggested `from numpy import integrate` (doesn't exist). The planner suggested `np.trapz` (also removed). Then `scipy.integrate.trapz` (removed in SciPy 1.14). None knew the correct API: `scipy.integrate.trapezoid`. This continued for **16 iterations** across Phases 3–4.
- **Root cause:** Neither the coder, reflector, nor planner had any awareness of which library versions were installed (numpy 2.4.1, scipy 1.17.0) or which APIs had been deprecated/removed.
- **Fix 1 — Environment info injection:** Added `_get_environment_info()` helper that runs `pip list --format=freeze` once per run (cached) and extracts versions of key packages (numpy, scipy, matplotlib, click, pytest). This is injected into the coder's prompt as a `## Environment` section so the LLM knows what's available.
- **Fix 2 — Deprecated API table in skill:** Added a "Deprecated / removed APIs" section to `skills/numerical-optimization/SKILL.md` with a table of common removals: `numpy.trapz → scipy.integrate.trapezoid`, `scipy.integrate.trapz → scipy.integrate.trapezoid`, `numpy.bool/int/float/complex → builtins`. Includes explicit "Do NOT use" warnings and correct replacement code.

### Issue 3: Reflector poisoned ground truth
- **Problem:** The reflector recorded the incorrect finding "The ImportError occurs because `trapz` should be imported from `numpy.integrate`, not directly from `numpy`." This is factually wrong — numpy has no `integrate` submodule. The wrong finding was fed back to the coder via ground_truth, reinforcing the incorrect approach.
- **Mitigation:** The environment info injection (Fix 1 above) and skill deprecation table (Fix 2) should prevent this class of error by giving the LLM correct information upfront. A full "validate reflector suggestions" system is deferred.

### Test Impact
- All 135 tests still pass (1 skipped). No test changes needed — the new `_collection_error` field and `_get_environment_info()` are additive.

## State Management: Code Drafts (2025-07)

### Problem: Purely Additive Merge Causes Layout Conflicts
- **Root cause:** The coder's merge logic `merged_drafts = dict(existing); merged_drafts.update(new)` is purely additive — files can enter `code_drafts` but never leave. When the planner switches from flat (`pkg/`) to `src/pkg/` layout, old flat files persist forever.
- **Evidence from schrodinger run:** Both `square_well/solver.py` (167 lines, full implementation) and `src/square_well/solver.py` (24 lines, stub) coexisted. `_check_duplicate_modules()` detected it as an error but the coder couldn't resolve it because it had no deletion mechanism. This caused an unresolvable loop for 16+ iterations.
- **`_warn_stale_files()` was ineffective:** It removes disk files not in `code_drafts`, but since `code_drafts` grows monotonically, it never detected the stale flat-layout files.

### Fix 1: Auto-resolve duplicate layouts (`_resolve_duplicate_layouts`)
- New helper called after the coder merge step. When the same package exists in both flat (`pkg/`) and `src/pkg/` layouts, it determines which layout `new_drafts` prefers (by counting files) and prunes the other from `merged_drafts`. On ties, prefers `src/` (modern convention). Stale files are also removed from disk in write mode.

### Fix 2: Explicit file deletion via `# DELETE` marker
- If the coder emits a code block with just `# DELETE` as content, the file is removed from `code_drafts` and deleted from disk. This gives the LLM a first-class way to remove obsolete files during restructuring.
- Documented in `prompts/coder.md` with example syntax.

### Fix 3: Clean source-file protection during revisions
- Previously only test files were protected from accidental overwrite (after `_phase_error_count >= 2`). Now clean non-test source files (those with no syntax/import errors, marked ✅ in the prompt) are also protected. This prevents the coder from regressing working solver code while fixing unrelated wavefunction errors.
- Protection only applies during revision iterations (non-empty `reflection`) and only after `phase_errors >= 2`, giving the coder initial freedom to make changes.

### Test Impact
- 143 tests pass (was 135). Added 8 new tests covering `_resolve_duplicate_layouts` (4 tests), deletion marker (2 tests), and clean-file protection (2 tests).

## Replan Bug: Scaffolding Regression (2025-07)

### Problem: Replan Replaces Implementation Phase with Scaffolding
- **Root cause:** During replanning, the planner uses the same system prompt that asks for a full `## Phase 1` / `## Phase 2` / etc. multi-phase plan. The LLM generates a fresh plan starting with scaffolding (Phase 1). The code then takes `new_phases[0]` (the scaffolding phase) and uses its description to replace the current phase — overwriting the solver/implementation description with scaffolding instructions.
- **Evidence from schrodinger run (2025-03-24):** Both replans for Phase 2 (iterations 14 and 24) produced "Establish the project structure, package layout, and stub out all modules" instead of a revised solver approach. This wasted both replan attempts and directly contributed to Phase 2 being force-advanced with failing tests.
- **Impact:** Phase 2's description was replaced with scaffolding instructions. The coder received scaffolding instructions where it should have received a revised solver algorithm. Two replan opportunities were wasted.

### Fix 1: Dedicated replan prompt (`prompts/planner_replan.md`)
- New system prompt specifically for replanning that instructs the LLM to output ONLY a single revised phase description, not a full multi-phase plan. Tells it explicitly: "Do NOT output a full multi-phase plan. Do NOT start over with scaffolding."
- Loaded via `EXPLORER_PROMPT_PLANNER_REPLAN` env var, falling back to `planner_replan.md`.

### Fix 2: Smart phase selection (`_pick_replan_phase`)
- New helper that handles the case where the LLM still outputs multi-phase plans during replan. Selection priority:
  1. If only one phase parsed, use it.
  2. If multiple, prefer the one whose title matches the current phase (case-insensitive substring).
  3. Skip scaffolding phases (contain "stub" + "pass" + "scaffolding" in title).
  4. Fall back to first non-scaffolding phase, or last phase if all are scaffolding.

### Additional Finding: Duplicate Layout Still in Plan Artifact
- Even though no `src/` directory was created on disk (layout resolver working), the replanned Phase 2 description still references `src/square_well/` file paths. This is cosmetic (the coder prompt and layout resolver handle it), but the plan.md artifact is confusing.

### Test Impact
- 151 tests pass (was 143). Added 7 new tests: `_pick_replan_phase` (5 tests) and planner replan integration (2 tests).

## Test Protection: Unified Clean-Files Criterion (2025-07)

### Problem: Test Files with Bugs Were Unconditionally Protected
- **Root cause:** The coder's test protection logic unconditionally protected ALL `test_*.py` files when `_phase_error_count >= 2`, regardless of whether those test files had passing or failing tests. Source file protection, by contrast, correctly used `clean_files` — only protecting files with zero errors.
- **Evidence from schrodinger GPT-4 run (2025-03-24):** Phase 4 (CLI) was stuck for 44 iterations (36–80) on `ImportError: attempted relative import with no known parent package`. The test file `tests/test_cli.py` contained a buggy `run_cli()` helper that initialized a correct `python -m square_well.cli` invocation but then overwrote it with `python square_well/cli.py` (direct script execution). The reflector correctly identified the fix every single iteration, but the coder could not apply it because:
  1. The test file was protected from modification (test protection kicked in at `_phase_error_count >= 2`)
  2. The coder kept regenerating `cli.py` with relative imports (`from .solver import ...`), which is correct for module execution but fails with direct script execution
- **Impact:** 44 wasted iterations. The reflector's correct fix ("change to absolute imports in cli.py") was ignored by the coder, and the alternative fix ("change test to use `python -m`") was blocked by test protection.

### Fix: Unified Protection Using `clean_files`
- Changed test protection from unconditional (`test_*.py` → protected) to clean-files-based (`file in clean_files` → protected). Now both test files and source files use the same criterion: only files verified as having zero errors are protected.
- **Behavior change:**
  - Test files with ALL tests passing: Still protected (in `clean_files`) ✅
  - Test files with ANY test failing: Not protected, coder can fix them ✅
  - Source files with no errors: Still protected (in `clean_files`) ✅
  - Source files with errors: Not protected, coder can fix them ✅
- **Risk assessment:** A test file with 4/5 passing tests won't be in `clean_files` (requires zero errors), so the coder could theoretically weaken passing tests. However: (a) the reflector's guidance steers toward fixing the failing test, not weakening others, (b) the `phase_errors >= 2` gate still gives the coder initial freedom before any protection, and (c) the alternative (44-iteration stuck loops) is far worse.

### Test Impact
- Updated `test_revision_protects_existing_test_files` → `test_revision_protects_clean_test_files` (now requires `clean_files` to contain the test file for protection).
- Added `test_revision_allows_test_file_fixes_when_not_clean` (test files NOT in `clean_files` can be updated by the coder even after `_phase_error_count >= 2`).

## Cross-Module Contract Checking (2025-07)

### Problem: Interface Mismatches Between Modules Not Caught Before Pytest
- **Root cause:** The verifier's pre-test checks (`_check_import_consistency`, `_check_duplicate_modules`, `_check_syntax`) only validated that imported **names** exist. They did not check that function **signatures** (argument count, return type) match between definitions and call sites. Each module's unit tests pass individually, but the modules fail when combined.
- **Evidence from schrodinger GPT-4 run (2025-03-24):** Three post-run fixes were needed:
  1. Click `--V0` becomes parameter `v0` (lowercase) — convention mismatch
  2. `find_bound_energies()` returned `energies` but CLI expected `(energies, parities)` — return value contract mismatch
  3. Tests needed updating to unpack the new return tuple — cascading interface change
- **Impact:** All three bugs survived to the end of the run because unit tests passed in isolation.

### Fix 1: `_check_cross_module_contracts()` — AST-based contract validation
- New pre-test check added to the verifier pipeline (runs after import consistency, before pytest).
- Builds a map of all function definitions with their AST nodes across source modules.
- For each call to an imported function, checks:
  1. **Argument count:** Positional arg count at call site vs min/max params in definition (respects defaults, `*args`).
  2. **Return-value unpacking:** If caller does `a, b = func()`, checks that the function's return statements consistently return a tuple of the same length. Detects both over-unpacking (caller expects 2, function returns 1) and under-unpacking (caller assigns to single variable, function returns tuple of 2).
- Helper functions: `_build_module_definitions()` (shared with `_check_import_consistency`), `_count_return_elements()`, `_min_positional_args()`, `_max_positional_args()`, `_check_return_unpacking()`.

### Fix 2: Click CLI guidance in coder prompt and python-testing skill
- Added to `prompts/coder.md`: Warning about Click lowercasing option names, recommendation to use lowercase options or explicit param names, and module invocation for CLI tests.
- Added to `skills/python-testing/SKILL.md`: New "CLI testing with Click" section and "Cross-module interface consistency" section. Extended pitfalls table with Click, relative import, and return value mismatch entries.

### Fix 3: Interface consistency guidance in coder prompt
- Added "Cross-module consistency" section to `prompts/coder.md`: When changing return types, update ALL callers. When adding parameters, update all call sites.

## pyproject.toml Validation (2025-07)

### Problem: Broken pyproject.toml Causes Silent `pip install -e .` Failures
- **Root cause:** The coder prompt said "generate a minimal pyproject.toml with [project] and dependencies" but didn't require `[build-system]`. LLMs frequently omit it, or mix build backends (e.g. `[tool.setuptools.packages.find]` with `build-backend = "hatchling.build"`). Without `[build-system]`, `pip install -e .` falls back to legacy setuptools behavior and fails. The verifier's `_ensure_importable()` silently fell through to conftest-based sys.path fallback, masking the issue.
- **Evidence from schrodinger GPT-4 run (2025-03-24):** The generated pyproject.toml had no `[build-system]`, mixed `[tool.setuptools.packages.find]` with `[tool.hatch.build.targets.wheel]`, and `pip install -e .` failed with a traceback.
- **Additional finding:** The python-testing skill template had `build-backend = "hatchling.backends"` (wrong) instead of `"hatchling.build"` (correct).

### Fix 1: `_check_pyproject_toml()` — pre-test validator
- New verifier pre-check that runs before syntax checks. Validates:
  1. `[build-system]` table exists with `requires` and `build-backend`.
  2. `[project]` table exists with `name`.
  3. No mixed backend configuration (setuptools config with hatchling backend, or vice versa).
  4. Valid TOML syntax.
- Errors are reported as actionable messages that tell the LLM exactly what to add.

### Fix 2: Prescriptive pyproject.toml template in prompts and skills
- Updated `prompts/coder.md`: Replaced "generate a minimal pyproject.toml" with an exact copy-paste template including `[build-system]`, `[project]`, and explicit warnings.
- Updated `skills/python-testing/SKILL.md`: Replaced vague template with mandatory template, added rules section, fixed `hatchling.backends` → `hatchling.build`.
- Updated `prompts/planner.md`: Phase 1 scaffolding requirement now explicitly mentions `[build-system]` (hatchling) alongside `[project]`.

## Context Curation Fixes (March 2025)

Four fixes re-applied on top of `7a8565d` ("update for larger LLMs"), addressing issues observed in run #4 (nemotron-3-super:120b, Schrödinger finite-well task).

### 1. Unfenced Code Block Fallback Parser
- **Problem observed:** Some LLMs (especially nemotron-3-super:120b) occasionally emit file contents without fenced code blocks — just a filepath line followed by code. `_parse_code_blocks()` returned an empty dict, causing "No code drafts" failures.
- **Fix:** `_parse_code_blocks()` now falls back to `_parse_unfenced_blocks()` when no fenced blocks are found. The fallback detects standalone lines that pass `_looks_like_filepath()` and collects content between consecutive filepath headers. Requires ≥2 filepath headers to activate (single match is too ambiguous).
- **Test count:** 6 new tests in `TestParseUnfencedBlocks`.

### 2. Reflector Anti-Test-Change Directive
- **Problem observed:** In runs #3 and #4, the reflector suggested changing test files to match broken code, e.g. "adjust test expectations" or "change the test to accept the actual values." This caused the coder to weaken tests, masking real implementation bugs.
- **Fix (prompt):** Added to `prompts/reflector.md`: "NEVER suggest changing tests. Tests are the specification — they define the correct API signatures, function names, parameter names, return types, and expected values."
- **Fix (code):** The reflector node now injects an "IMPORTANT: direction of fixes" section into the user message, reinforcing that tests are the specification and implementation must change to match.
- **Test count:** 1 new test (`test_includes_anti_test_change_directive`).

### 3. Ground Truth → Coder Context
- **Problem observed:** In run #4, the coder node only received ground_truth indirectly via the planner's revised plan. The coder kept repeating algorithmic mistakes (e.g. using fixed bracketing intervals) that the reflector had already identified as failures, because the ground_truth findings weren't in the coder's prompt.
- **Fix:** The `coder()` function now injects `state['ground_truth']` as a "## Lessons learned from prior iterations" section in the user prompt. Empty ground_truth produces no section.
- **Test count:** 2 new tests in `TestCoderGroundTruthContext`.

### 4. Enriched Reflector Context
- **Problem observed:** The reflector lacked context to produce targeted advice: it didn't see prior reflections (leading to repetition), didn't see accumulated ground_truth findings (re-discovering the same insights), and lacked explicit direction about never changing tests.
- **Fix:** The reflector node now builds a `reflector_parts` list with: (a) "IMPORTANT: direction of fixes" block, (b) "Known lessons (do NOT re-suggest these)" from ground_truth, (c) "Previous reflection" from the prior iteration.
- **Test count:** 2 new tests (`test_includes_ground_truth`, `test_includes_prior_reflection`).

### Combined: 11 new tests. Total: 180 tests passing (1 skipped).

## Scaffolding Phase Generates Wrong Test Expectations (July 2025)

### Problem: Hardcoded Numerical Values in Scaffold Tests Lock In Wrong Specifications
- **Run:** `~/git/schrodinger-qwen`, `gpt-oss:120b` (nemotron-3-super:120b via Ollama), 126 steps, 40 iterations, `passed: false`.
- **Outcome:** 4 phases (Scaffolding, Solver, Wavefunctions, CLI). Phases 1–3 completed (2–3 force-advanced), Phase 4 stuck for 20+ iterations with identical 6-failed/4-passed pattern.
- **Root cause:** Phase 1 (scaffolding) generated tests with hardcoded numerical expectations that were **physically wrong**:
  1. `assert len(energies) == 5` — correct answer is 7 (verified analytically: $C = \sqrt{2 \times 50 \times 1} = 10$, giving 4 even + 3 odd = 7 bound states)
  2. `assert np.all(np.diff(energies) <= 0)` — requires non-increasing, but comment says "sorted ascending"; solver correctly returns ascending
  3. `assert len(energies) == 0` for shallow well ($V_0=0.1$) — in 1D, there is ALWAYS at least 1 bound state
- **Propagation chain:** Scaffolding tests with wrong numbers → test protection locks them in → force-advance propagates broken tests to later phases → solver code is correct but tests reject it → 20+ identical failures → MAX_ITERATIONS hit
- **Evidence:** The solver's `energies.sort()` returns 7 bound states in ascending order (physically correct). The reflector correctly diagnosed the issue every iteration but the coder couldn't change tests (anti-test-change rule).

### Fix 1: Scaffold Test Quality Guidance (Root Cause Fix)
- **Files:** `prompts/coder.md`, `prompts/planner.md`
- **Change (coder.md):** Added explicit rule for scaffolding-phase tests: "MUST use only structural assertions — verify imports, types, shapes, and constraints (e.g. `assert callable(solve)`, `isinstance(result, np.ndarray)`, `len(result) > 0`, `all(x < 0 for x in energies)`). NEVER hardcode specific numerical results."
- **Change (planner.md):** Replaced "trivial assertions (`assert True`)" with "structural assertions only — verify imports work, return types are correct, and basic constraints hold. Do NOT include specific numerical expected values."
- **Rationale:** Stubs have `pass` bodies so correct numerical answers are unknown at scaffolding time. Wrong hardcoded values become locked-in specifications that block all later phases.

### Fix 2: Prefer Replan Over Force-Advance (Structural Fix)
- **File:** `src/cli.py`
- **Change (`_make_should_continue`):** When per-phase iteration cap (8) is exceeded, now checks `_replan_count < MAX_REPLANS` first. If replans remain, routes to `"reflect"` (which forces a replan via `_after_reflector`). Only force-advances as last resort when replans are also exhausted.
- **Change (`_after_reflector`):** When `_phase_iteration_count >= MAX_PHASE_ITERATIONS` and replans remain, forces immediate replan (bypasses normal error-count threshold). This gives the planner a chance to decompose the problem differently before carrying failures to the next phase.
- **Rationale:** The old behavior force-advanced blindly after 8 iterations, carrying wrong tests and broken code forward. The new behavior tries replanning first (giving the planner a shot at a different decomposition) and only force-advances when replanning is exhausted.

### Fix 3: Try/Except Around Graph Runner (Resilience Fix)
- **File:** `src/cli.py`
- **Change:** Wrapped `app.stream()` loop in try/except. On unhandled exception: logs the traceback, writes partial summary and ground truth, exits with code 2.
- **Rationale:** A prior run died silently at step 06 with no error output or summary. The graph runner had no error handling — any exception in LLM invocation, node logic, or streaming killed the process with no diagnostic output.

### Test Impact
- 8 new tests: `TestShouldContinue` (6 tests), `TestAfterReflector` force-replan (2 tests). Total: 188 tests passing (1 skipped).

## Phase 1 Refactor: Conversation Transcript (Context Foundation)

- **Problem observed:** Each LLM call in the Scientific Loop was stateless — a fresh `[SystemMessage, HumanMessage]` pair with no memory of prior iterations. Context was assembled ad-hoc: the coder manually injected `ground_truth`, `reflection`, `test_logs`, and an `_error_repeat_count` CRITICAL warning as separate conditional sections. The reflector similarly injected `ground_truth` and `prev_reflection`. The planner (replan) injected `reflection` and `test_logs`. Each node had its own hand-crafted prompt assembly logic.
- **Root cause (amnesia):** With no structured run history, the LLM on iteration 5 had limited awareness of iterations 1–4. Information was compressed through state fields (`reflection` = latest analysis only, `ground_truth` = deduplicated findings), losing the narrative of what was tried, what failed, and why.

### New module: `src/orchestrator/transcript.py`
- **`make_entry(role, content, *, node, step, phase, summary)`** — Creates a structured transcript entry. Role is `"human"` (environment: verifier, advance_phase) or `"ai"` (agent: planner, coder, reflector). Summary defaults to first 200 chars of first line.
- **`format_history(transcript, max_recent=8)`** — Formats the transcript as a `## Run History` section. Recent entries (last 8) are shown in full. Older entries are condensed to one-line summaries with `[step N, node]` prefix. Returns empty string for empty transcript.

### State change: `transcript: list[dict]` added to `ScientificState`
- Initialized as `[]` in `cli.py`
- Persisted across phases (NOT cleared on `advance_phase`)
- Each node appends entries and returns the updated transcript

### Node wiring (all 5 nodes append to transcript):
- **Planner (initial):** AI entry with full plan text, summary = "Plan: N phases — title1, title2"
- **Planner (replan):** AI entry with revised plan, summary = "Replanned Phase N: title"
- **Coder:** AI entry with file list summary ("Generated N file(s): file1, file2"), not full source
- **Verifier (pass):** Human entry "All tests passed ✓"
- **Verifier (fail):** Human entry with test output (truncated to 3000 chars)
- **Reflector:** AI entry with analysis + any new key findings
- **Advance Phase:** Human entry "Phase N 'title' completed. Moving to Phase N+1: 'title'"

### Context replacement (nodes READ from transcript):
- **Coder:** `format_history()` replaces 4 ad-hoc injections: `ground_truth`, `reflection`, `test_logs`, and `_error_repeat_count` CRITICAL warning. Keeps: phase context, environment info, verified_fixes, existing file source, skills.
- **Reflector:** `format_history()` replaces `ground_truth` and `prev_reflection` injections. Keeps: current test logs, failing source code, anti-test-change directive.
- **Planner (replan):** `format_history()` replaces inline `reflection` and `test_logs` in the replan section. Keeps: task, phase description, skills.

### Key design decisions:
- Transcript is formatted as a narrative section in the HumanMessage (not multi-turn chat). This avoids message alternation issues across different LLM providers.
- Each node still uses its own system prompt. The transcript provides history; the current message provides structured data (code, environment, skills).
- Windowing: recent 8 entries shown in full, older entries condensed to summaries. This prevents prompt bloat on long runs while preserving recent context.

### Test Impact
- 17 new tests in `tests/test_transcript.py` (5 `TestMakeEntry`, 7 `TestFormatHistory`, 5 `TestTranscriptIntegration`)
- 7 existing tests updated to provide transcript entries instead of direct state field assertions
- Total: 205 tests passing (1 skipped).

## Phase 2 Refactor: LLM-Powered Output Validation (Heuristic Replacement)

- **Problem observed:** The verifier contained 7 AST-based heuristic check functions (~250 lines of complex code) that were added reactively to catch specific failure modes: import mismatches, cross-module contract errors (arg count, return unpacking), and layout conflicts (flat vs src/). Each heuristic used `ast.parse` + `ast.walk` to analyze code structure. While effective, they were brittle, hard to maintain, and could only catch the specific patterns they were coded for.
- **Root cause (heuristic overload):** Each new failure mode observed in production runs led to a new hand-coded AST check. The checks grew organically without a unifying framework, resulting in ~250 lines of tightly coupled code that was hard to extend and test.

### Solution: Replace 3 complex AST heuristics with a single LLM triage call

**Kept (cheap, deterministic):**
- `_check_syntax()` — 18 lines, `ast.parse()` each `.py` file. Catches syntax errors instantly.
- `_check_pyproject_toml()` — 78 lines, TOML parse + validate. Prevents `pip install -e .` failures.

**Replaced with `_llm_triage()`:**
- `_check_import_consistency()` (49 lines) — cross-checked test imports vs source definitions
- `_check_cross_module_contracts()` (76 lines) — arg count + return unpacking across modules
- `_check_duplicate_modules()` (31 lines) — flat vs src/ layout conflict detection
- 5 helper functions: `_build_module_definitions`, `_count_return_elements`, `_min_positional_args`, `_max_positional_args`, `_check_return_unpacking` (~90 lines)
- **Total removed: ~250 lines**

### New: `_llm_triage(code_drafts, llm)` function
- Loads `prompts/triage.md` system prompt via `_load_prompt()`
- Builds a code listing of all `.py` files
- Calls the LLM with `_invoke_llm()` (same streaming/retry as other nodes)
- Parses response: "LGTM" → empty list, bullet points → error list
- Returns `list[str]` in the same format as the old heuristics

### New: `prompts/triage.md`
- Focused prompt for structural code review before pytest
- Checks: import consistency, function signature mismatches, return-value unpacking, layout conflicts
- Output format: bullet-point errors or "LGTM"
- Rules: only check internal code relationships, tests are specification, be precise

### Verifier flow change
- Before: syntax → TOML → duplicates → imports → contracts → (pytest OR pre_errors)
- After: syntax → TOML → LLM triage → (pytest OR pre_errors)
- The verifier now calls `get_llm()` when syntax + TOML checks pass
- LLM triage result follows the same `logs = pre_errors or _run_pytest(root)` pattern
- Triage errors short-circuit pytest, just like the old heuristics

### Impact
- nodes.py: 1624 → 1395 lines (229 lines removed net)
- Tests: 206 → 195 (21 heuristic tests removed, 10 triage tests added)
- New test classes: `TestLlmTriage` (6 tests), `TestLlmTriageInVerifier` (4 tests)
- Existing verifier tests updated with autouse `_mock_triage_llm` fixture
- Integration tests updated with "LGTM" triage response in LLM sequences

## Phase 3: Merge Reflector Into Conversation Flow

- **Problem:** The reflector made 2 separate LLM calls per reflection cycle: (1) analysis with `reflector.md` prompt, (2) findings extraction with `findings.md` prompt. The second call was a simple structured task (extract bullet-point findings) that didn't need a dedicated LLM invocation.
- **Fix:** Merged the two prompts into a single `reflector.md` that instructs the LLM to output analysis **and** a `## Key Findings` section at the end. The new `_extract_findings()` helper parses the combined response locally, splitting analysis from bullet-point findings. Deduplication against existing `ground_truth` is preserved.
- **Deleted:** `prompts/findings.md` (merged into `reflector.md`).
- **New helper:** `_extract_findings(text) -> (analysis, findings_list)` in `nodes.py` — splits on `## Key Findings` header, extracts `"- …"` bullet lines, handles `NONE` sentinel.
- **Reflector now:** 1 LLM call (was 2). Same outputs: `reflection`, `ground_truth`, `transcript`, `_prompt_summary`. No changes to the `reflection` state field contract — it's still used by coder (`is_revision`), planner (`is_replan`), and verifier (`verified_fixes`).

### Impact
- nodes.py: 1395 → 1416 lines (net +21 from `_extract_findings` helper; -1 LLM call removed)
- Tests: 195 → 199 (4 new `_extract_findings` tests, 1 replaced `test_existing_findings_passed_to_llm` → `test_no_findings_when_none`)
- Integration test LLM sequence reduced from 7 to 6 calls (removed separate findings extraction call)

## Phase 4: Simplify Graph Routing (LLM-Informed Decisions)

- **Problem:** Post-reflector routing used counter-based heuristics: `_phase_error_count >= REPLAN_THRESHOLD * 2^_replan_count` to decide when to replan, `MAX_PHASE_ITERATIONS` to force-advance phases, and `verified_fixes` keyword matching to inject constraints into the coder. These were brittle, hard to tune, and couldn't adapt to context.
- **Fix:** The reflector now outputs a `## Action` section at the end of its response with exactly `RETRY` or `REPLAN`. A new `_extract_action()` helper parses this, and `_after_reflector()` uses the LLM's recommendation (capped by `MAX_REPLANS`) instead of counter thresholds.
- **Removed constants:** `REPLAN_THRESHOLD`, `MAX_PHASE_ITERATIONS` from `cli.py`.
- **Removed state fields:** `_phase_error_count`, `_prev_syntax_file_count`, `verified_fixes`, `best_code_drafts`, `best_error_count` (5 fields removed from `ScientificState`).
- **Added state field:** `_reflector_action: str` — stores `"retry"` or `"replan"` from the reflector's `## Action` section.
- **New helper:** `_extract_action(text) -> str` in `nodes.py` — parses `## Action` header, returns `"replan"` or `"retry"` (default).
- **Updated helper:** `_extract_findings()` now strips the `## Action` section before parsing findings.
- **Simplified `_should_continue()`:** 3 outcomes only — `end` (tests pass, all done), `advance_phase` (tests pass, more phases), `reflect` (tests fail). Removed force-advance and phase-cap logic.
- **Simplified `_after_reflector()`:** Reads `_reflector_action` and `_replan_count`. If `replan` and replans remaining → replan; otherwise → coder.
- **Removed from verifier:** `verified_fixes` keyword matching, rollback/best-draft tracking (`best_code_drafts`, `best_error_count`), `_phase_error_count` tracking, `_prev_syntax_file_count` tracking.
- **Removed from coder:** `verified_fixes` injection ("⚠️ MANDATORY CONSTRAINTS" section).
- **Coder clean-file protection:** Now uses `_phase_iteration_count >= 2` instead of removed `_phase_error_count >= 2`.

### Impact
- State fields: 25 → 21 (removed 5, added 1)
- cli.py: Removed 2 constants, simplified 2 routing functions
- nodes.py: Removed ~50 lines of heuristic code, added ~15 lines for `_extract_action`
- Tests: 199 → 196 (removed 13 counter-based tests, added 10 LLM-routing tests)

## Phase 5: Compress Prompt Assembly

- **Problem:** Three nodes (coder, triage, reflector) each had inline loops to format `code_drafts` as fenced Markdown code blocks. The formatting logic was duplicated with minor variations (clean-file marks, failing-only filter).
- **Fix:** Extracted a shared `_format_code_listing()` helper that handles all three use cases via keyword arguments (`clean_files`, `only_failing`). Each call site reduced from 5–12 lines to 2–3 lines.
- **New helper:** `_format_code_listing(code_drafts, *, clean_files=None, only_failing=False) -> list[str]` — filters to `.py` files, optionally annotates clean files with ✅, optionally skips clean files.
- **Assessment:** The remaining "duplicated" patterns (run-history injection, skills-context injection, SystemMessage/HumanMessage invocation) are each 2–3 lines and don't benefit from further abstraction — extracting them would add indirection without reducing complexity.

### Impact
- nodes.py: 1377 → ~1370 lines (small net reduction — gained helper, simplified 3 call sites)
- Tests: 196 → 201 (5 new `_format_code_listing` tests)

## Phase 1 Upgrade: Inner Coder Self-Correction Loop

- **Motivation:** Analysis of a VS Code Copilot agent trace (JSONL) showed it solved the Schrödinger problem in ~1 LLM turn by immediately running code and fixing errors in the same session. Our agent required 5–20 outer graph cycles (3–4 LLM calls each). The inner loop closes this feedback gap.
- **What changed:** The `coder()` node now runs `_quick_verify()` (syntax check + pytest in a temp sandbox) after each LLM generation. If tests fail, the error output is appended to the conversation and the LLM is re-invoked — up to `MAX_INNER_ITERATIONS` (3) total attempts — before returning to the outer graph.
- **Key design decisions:**
  - Clean-file protection from the outer state applies on **every** inner-loop iteration, not just the first — prevents the LLM from regressing verified-clean files during self-correction.
  - The inner loop skips verification on the last attempt (lets the outer verifier handle it) to avoid wasting time on unrecoverable failures.
  - `_quick_verify()` is lightweight: syntax check + pytest, no LLM triage. Keeps inner-loop latency low.
  - `_merge_and_clean()` extracted as a helper since the merge+delete+layout-resolution sequence is needed on every inner iteration.
- **New state field:** `_inner_loop_count: int` — tracks how many self-correction iterations the coder used.
- **Reporter:** Shows `⟳ N inner-loop self-correction(s)` in the coder's output when `_inner_loop_count > 0`.
- **Transcript:** Coder transcript entry includes `Inner-loop self-corrections: N` when N > 0.
- **Integration test impact:** `TestLoopReflectsOnFailure` now expects `iteration_count == 1` (was 2) because the inner loop fixes the bad code before the outer verifier runs, eliminating the reflector round-trip.
- **Tests:** 201 → 213 (added 10 new tests: `TestQuickVerify` (4), `TestMergeAndClean` (2), `TestCoderInnerLoop` (6)).

## Planner Hardcodes Wrong Eigenstate Count (April 2026)

### Problem: Solver Output Truncated by Wrong Formula from Plan Tests
- **Run:** `~/git/schrodinger-qwen`, Qwen 2.5-Coder 32B (tool-calling mode)
- **Symptom:** `square_well.cli solve` returns 6 bound states; the correct answer is 7. The shallowest even-parity state (E ≈ -3.16) is missing.
- **Root cause chain:**
  1. **Planner** wrote test `test_number_of_bound_states_default` with formula `expected = int(np.floor(C/np.pi)) * 2` (= 6 for C = 10). This formula is **wrong** — it undercounts by 1 because even-parity states start at n=0, giving `floor(C/π) + 1` even states, not `floor(C/π)`.
  2. **Coder** added truncation `max_states = floor(C/π) * 2; energies = energies[:max_states]` to match the test.
  3. **Verifier** ran tests → all passed (wrong test was satisfied).
  4. **Reflector** never engaged (no failures).
- **Correct formula:** Total bound states = `ceil(2C/π)`. For C = 10: `ceil(20/π) = 7`.
- **Key insight 1:** The quantum-mechanics skill already said "NEVER assume you know the exact number of eigenvalues" but the planner ignored this guidance and hardcoded a formula. Skill guidance cannot force LLM behavior — critical constraints must also be reinforced in the planner prompt itself.
- **Key insight 2:** When the planner writes tests with specific numerical expectations derived from a formula, and the formula is wrong, the entire system has no mechanism to catch the error. Tests pass → verifier succeeds → no reflector → shipped with bug.
- **Key insight 3:** Quantities that emerge from computation (number of roots, number of eigenvalues) should NOT be hardcoded in tests written before implementation. The correct pattern is: (a) test each result individually (residual checks), (b) use lower-bound count tests during initial implementation, (c) re-derive exact count from working code as a post-implementation validation step.

### Fixes Applied
1. **`prompts/planner_direct.md`** — Added "Emergent quantities" rule: don't hardcode solver-derived counts. Use property-based assertions and lower bounds. Re-derive exact counts after implementation.
2. **`prompts/planner.md`** — Same rule for scaffold mode.
3. **`skills/quantum-mechanics/SKILL.md`** — Added "Counting bound states" section with correct formula `ceil(2C/π)`, explicit warning against `floor(C/π)*2`, and good/bad test patterns.
