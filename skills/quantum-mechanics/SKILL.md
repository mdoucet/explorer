---
name: quantum-mechanics
description: Quantum mechanics, Schrödinger equation, wavefunctions, eigenvalue problems, bound states, potential wells, atomic units
---

# Quantum Mechanics

## Overview

Use this skill when the task involves quantum mechanics, solving the Schrödinger
equation, computing wavefunctions, finding energy eigenvalues, or working with
quantum potential wells.

## Instructions

### IMPORTANT BACKGROUND INFORMATION

- The user may be using this skill for a problem that has no known solution. Do not assume that you know the exact numerical values or the number of energy eigenvalues or wavefunctions in advance. Instead, write code that can compute these from the mathematical properties of the system, and write tests that verify these properties rather than specific numbers.

### Physical conventions

- Default to **atomic units** ($\hbar = 1$, $m_e = 1$) unless the task
  specifies otherwise.
- Always state the unit system in docstrings and CLI output.

### Schrödinger equation

The time-independent Schrödinger equation is:

$$
-\frac{\hbar^2}{2m}\frac{d^2\psi(x)}{dx^2} + V(x)\,\psi(x) = E\,\psi(x)
$$

When implementing:
- Represent $V(x)$ as a callable accepting a NumPy array.
- Use `scipy.optimize.brentq` for transcendental eigenvalue equations.
- Use `scipy.integrate.solve_ivp` for numerical shooting methods.

### Bound states

For piecewise-constant potentials (square wells, step potentials):
- Match the wavefunction **and** its derivative at every boundary.
- Bound states satisfy $E < 0$ (with $V(\pm\infty) = 0$ convention).
- Define:

$$
k = \sqrt{\frac{2m(E + V_0)}{\hbar^2}}, \qquad
\kappa = \sqrt{\frac{-2mE}{\hbar^2}}
$$

### Wavefunction normalisation

Always normalise so that $\int_{-\infty}^{\infty} |\psi(x)|^2 \, dx = 1$.
Use `numpy.trapz` (or `numpy.trapezoid` on NumPy ≥ 2) for numerical
integration on a grid.

### Trigonometric functions in NumPy / SciPy

When implementing transcendental equations for bound states you will need
`tan`, `cot`, `sec`, `csc`, etc.  **Be aware of what NumPy provides:**

| Function | NumPy name | Notes |
|----------|-----------|-------|
| sin | `np.sin` | ✅ built-in |
| cos | `np.cos` | ✅ built-in |
| tan | `np.tan` | ✅ built-in |
| cot | — | ❌ **does NOT exist** as `np.cot`. Use `np.cos(x) / np.sin(x)` or `1 / np.tan(x)` |
| sec | — | ❌ Use `1 / np.cos(x)` |
| csc | — | ❌ Use `1 / np.sin(x)` |
| arctan | `np.arctan` | ✅ built-in |
| arctan2 | `np.arctan2` | ✅ built-in |

**Common mistake:** Writing `np.cot(x)` — this raises `AttributeError:
module 'numpy' has no attribute 'cot'`.  Always use `np.cos(x) / np.sin(x)`.

Guard against division by zero near poles:
```python
# Safe cotangent with NaN at poles instead of ±inf
def safe_cot(x):
    s = np.sin(x)
    return np.where(np.abs(s) > 1e-15, np.cos(x) / s, np.nan)
```

### Domain validity for transcendental equations

When solving the finite square-well transcendental equations, the
dimensionless variable $\xi$ must satisfy $0 < \xi < C$ where
$C = a\sqrt{2mV_0}/\hbar$.  Outside this range, $\sqrt{C^2 - \xi^2}$
produces `NaN`.

- **Never** evaluate the transcendental functions at $\xi \geq C$.
- **Never** write tests that call these functions with $\xi > C$ and
  compare the result with `==`, `<`, or `>` — the result is `NaN` and
  all comparisons return `False`.
- Clamp search brackets: `hi = min(hi, C - epsilon)`.
- If writing tests for the raw transcendental function, only use
  $\xi$ values strictly inside $[0, C)$.

### Testing guidance

- NEVER assume that you know the exact numerical values or the number of energy eigenvalues or wavefunctions in advance.  Instead, write tests that verify the mathematical properties of the solutions (e.g. normalisation, boundary conditions, asymptotic behaviour) rather than specific numbers.
- When computing wavefunctions, verify normalisation: `np.trapz(np.abs(psi)**2, x)` ≈ 1.

### Counting bound states — correct formula

For a symmetric finite square well with dimensionless parameter
$C = a\sqrt{2mV_0}/\hbar$, the total number of bound states is:

$$
N = \left\lceil \frac{2C}{\pi} \right\rceil
$$

**Common mistake:** using `floor(C/π) * 2`.  This is WRONG because
even-parity states start at $n = 0$, giving `floor(C/π) + 1` even states
(one more than odd), so multiplying by 2 undercounts by one when an extra
even state exists.  For C = 10: `floor(10/π) * 2 = 6` but the correct
answer is `ceil(20/π) = 7`.

**NEVER hardcode the state count in the solver or use it to truncate results.**
Let the root-finder discover all roots and return them all.

### Test patterns

**Good** — verify properties of each returned eigenvalue:
```python
def test_transcendental_residuals():
    energies = find_bound_energies()
    assert len(energies) >= 1  # at least one bound state
    for E in energies:
        k = np.sqrt(2 * m * (E + V0))
        kappa = np.sqrt(-2 * m * E)
        # Check even or odd transcendental equation is satisfied
        res_even = k * np.tan(k * a) - kappa
        res_odd = -k / np.tan(k * a) - kappa
        assert min(abs(res_even), abs(res_odd)) < 1e-8
```

**Good** — verify count AFTER implementation using `ceil(2C/π)`:
```python
def test_correct_number_of_bound_states():
    energies = find_bound_energies()
    C = a * np.sqrt(2 * m * V0) / hbar
    expected_count = int(np.ceil(2 * C / np.pi))
    assert len(energies) == expected_count
```

**BAD** — hardcoding a wrong formula as "upper bound":
```python
# DO NOT DO THIS — floor(C/π)*2 is incorrect!
expected = int(np.floor(C / np.pi)) * 2
assert len(energies) <= expected
```
