# Ground Truth

> Task: # Quantum Square-Well Solver

## Problem Statement

Create a small Python package with a CLI that solves the **time-independent
one-dimensional Schrödinger equation** for a **finite symmetric square-well
potential**.

The potential is defined as:

$$
V(x) = \begin{cases} -V_0 & \text{if } |x| < a \\ 0 & \text{otherwise} \end{cases}
$$

where $V_0 > 0$ is the well depth and $a$ is the half-width.

The stationary Schrödinger equation to solve is:

$$
-\frac{\hbar^2}{2m}\frac{d^2\psi(x)}{dx^2} + V(x)\,\psi(x) = E\,\psi(x)
$$

## Physics Background

Bound states ($-V_0 < E < 0$) satisfy transcendental equations obtained by
matching the wavefunction and its derivative at the well boundaries $x = \pm a$.

Define:

$$
k = \sqrt{\frac{2m(E + V_0)}{\hbar^2}}, \qquad
\kappa = \sqrt{\frac{-2mE}{\hbar^2}}
$$

- **Even-parity states**: $k \tan(ka) = \kappa$
- **Odd-parity states**: $-k \cot(ka) = \kappa$

with the constraint $k^2 + \kappa^2 = \frac{2mV_0}{\hbar^2}$.

## Requirements

### Package structure

```
src/
    square_well/
        __init__.py
        solver.py        — find bound-state eigenvalues by numerically
                           solving the transcendental equations above
        wavefunctions.py — compute the piecewise wavefunction ψ(x) for
                           each bound state
        cli.py           — Click CLI: print eigenvalues, optionally plot
                           wavefunctions with matplotlib
tests/
    test_solver.py       — unit tests for the eigenvalue finder
    test_wavefunctions.py — unit tests for the wavefunction builder
pyproject.toml           — minimal build metadata
```

### Solver (`solver.py`)

- Accept parameters: `m` (particle mass), `V0` (well depth), `a` (half-width).
- Default to atomic units: $\hbar = 1$, $m = 1$, $V_0 = 50$, $a = 1$.
- Return a sorted list of bound-state energies.
- Use `scipy.optimize.brentq` to find roots of the transcendental equations.

### Wavefunctions (`wavefunctions.py`)

- Given an energy eigenvalue, compute $\psi(x)$ on a grid.
- Normalise the wavefunction so that $\int |\psi|^2 dx = 1$.

### CLI (`cli.py`)

- Command: `square-well solve`
- Options: `--v0`, `--a`, `--mass`, `--num-states`
- Print the eigenvalues to stdout.
- Optional `--plot` flag to display wavefunctions via matplotlib.

### Constants

Use atomic units throughout:

| Symbol    | Value | Description          |
|-----------|-------|----------------------|
| $\hbar$   | 1     | Reduced Planck const |
| $m$       | 1     | Particle mass        |
| $V_0$     | 50    | Well depth           |
| $a$       | 1     | Half-width           |

## Acceptance Criteria

1. `pytest` must pass with all tests green.
2. The solver must find **all** bound states for the default parameters.
3. Eigenvalues must agree with reference values from the transcendental
   equations to within $10^{-6}$ relative error.
4. Wavefunctions must be normalised: $\left|\int |\psi|^2 dx - 1\right| < 10^{-4}$.
5. The CLI must print eigenvalues when invoked with `--v0 50 --a 1`.

> Iterations: 16

## Key Findings

- The test `test_wavefunctions.py` fails to import `solve_energy_levels` from `square_well.solver` because that function is not defined in the module (it may be renamed, missing, or not exported).
- The test `test_wavefunctions.py` fails with an ImportError because the function `solve_energy_levels` is not defined or exported in `square_well/solver.py`.
- The test `test_wavefunctions.py` fails with ImportError because `square_well/solver.py` does not define a top-level function named `solve_energy_levels`.
- The function may be missing, renamed, or defined inside a class/private scope, requiring either adding/exporting the function or updating the test import accordingly.
- The solver incorrectly includes the trivial E = 0 root, causing bound‑state checks to fail and the infinite‑well limit to return zero instead of the expected first level.
- The public function `solve_energy_levels` does not accept a `num_states` keyword argument, leading to a TypeError in the wavefunction test.
- To avoid non‑physical roots, the root‑finding should either start at a small positive energy or explicitly filter out any solutions with E ≤ 0 (or E ≥ V0) before returning results.
- The CLI test fails with AttributeError because `square_well.solver` does not expose a `solve_square_well` function.
- Adding the missing function (or an alias) to `src/square_well/solver.py` and exporting it will allow the CLI to succeed and the test to pass.
- Deprecation warnings about invalid escape sequences in docstrings can be resolved by using raw docstrings or properly escaping backslashes.
- The ImportError occurs because `square_well/cli.py` does not define a top‑level variable named `cli`; the Click command/group is defined under a different name (e.g., `main`) or inside a function, making it unavailable for import.
- The test `tests/test_cli.py` fails with `ImportError: cannot import name 'cli'` because the module `square_well.cli` does not expose an attribute named `cli`.
- Resolve the issue by either renaming the CLI object to `cli` in `square_well/cli.py` (or adding an alias `cli = <actual_name>`) or by updating the test to import the actual CLI object (e.g., `main` or `app`) and alias it as `cli`.
- The test collection fails because `square_well/cli.py` tries to import `solve_square_well` from `square_well.solver`, but that function is not defined or exposed in the solver module.
- The solver module's docstring contains an invalid escape sequence `\h`, which triggers a DeprecationWarning that can be fixed by using a raw string or escaping the backslash.
- Defining (or exposing) `solve_square_well` in `solver.py` (or adjusting the import to match the actual function name) and correcting the docstring escape sequence will resolve the import error and allow the tests to collect and run.
- The test `tests/test_cli.py` fails with an ImportError because it tries to import a name `cli` from `square_well.cli` that is not defined in the module.
- A DeprecationWarning is triggered by an invalid escape sequence `\h` present in a docstring within `square_well/solver.py`.
- Fixing the import (by exposing a `cli` object or adjusting the test) and correcting the escape sequence (using a raw string or escaping the backslash) will resolve both the error and the warning.
- The solver module does not expose a function named `solve_square_well`, causing an ImportError during test collection that prevents pytest from collecting any tests.
- The test suite fails to collect because `cli.py` attempts to import `solve_eigenvalue_equation` from `solver.py`, which does not define that function.
- A deprecation warning is raised due to an invalid escape sequence `\s` in a docstring in `solver.py`.
- The solver module lacks the expected exported symbols (`solve_eigenvalue`, `solve_eigenvalues`, and `solve_eigenvalue_equation`), causing ImportError during test collection.
- The module's docstring contains an invalid escape sequence `\s`, producing a DeprecationWarning that interferes with clean test runs.
