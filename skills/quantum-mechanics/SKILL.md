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

- Compare eigenvalues against analytically known limits (e.g. infinite-well
  limit: $E_n = \frac{n^2\pi^2\hbar^2}{2m(2a)^2}$).
- Use tolerances appropriate for the numerical method (typically `rtol=1e-6`).
- Test both even-parity and odd-parity states separately.
- Verify normalisation: `np.trapz(np.abs(psi)**2, x)` ≈ 1.
