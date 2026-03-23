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
