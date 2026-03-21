# Quantum Square-Well Solver

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
