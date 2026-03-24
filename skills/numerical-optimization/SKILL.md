---
name: numerical-optimization
description: Numerical optimization, root finding, minimization, curve fitting, scipy.optimize, gradient descent, Newton method
---

# Numerical Optimization

## Overview

Use this skill when the task involves finding roots of equations, minimizing
or maximizing functions, curve fitting, or other numerical optimization problems.

## Instructions

### Root finding

- Prefer `scipy.optimize.brentq` for bracketed scalar root finding — it is
  guaranteed to converge.
- Use `scipy.optimize.fsolve` for systems of nonlinear equations.
- Always validate that the bracket `[a, b]` contains a sign change before
  calling `brentq`.

### Minimization

- For smooth, unconstrained problems: `scipy.optimize.minimize` with
  `method="L-BFGS-B"`.
- For bounded scalar problems: `scipy.optimize.minimize_scalar`.
- For global optimization: `scipy.optimize.differential_evolution`.

### Curve fitting

- Use `scipy.optimize.curve_fit` for nonlinear least-squares.
- Always provide reasonable initial guesses (`p0`) to avoid local minima.
- Return uncertainties via the covariance matrix: `perr = np.sqrt(np.diag(pcov))`.

### Transcendental equations with periodic singularities

When solving equations involving `tan(x)`, `cot(x)`, or other periodic
functions with poles:

- **Never** do a blind sign-change search across the full domain — every
  pole creates a false sign change that `brentq` will converge to.
- Instead, **bracket roots analytically** using the branch structure:
  - For `f(x) = x·tan(x) - g(x)` (even parity): roots live in
    `(nπ, (n+½)π)` for n = 0, 1, 2, …
  - For `f(x) = -x·cot(x) - g(x)` (odd parity): roots live in
    `((n-½)π, nπ)` for n = 1, 2, 3, …
- Use `brentq` on **each analytic bracket individually**, with endpoints
  nudged inward by a small epsilon (e.g. `1e-12`) to avoid the poles.
- **Validate** each root: check `|f(root)| < tol` AND that the root is not
  near a pole (`|cos(root)| > epsilon` for tan-based equations).
- Terminate the search when the bracket is empty: `left >= right` or when
  `g(xi)` becomes imaginary (i.e. `xi > C` for finite-well problems).

Example (finite square well, even parity):
```python
import numpy as np
from scipy.optimize import brentq

def _even_roots(C: float) -> list[float]:
    """Find xi values satisfying xi·tan(xi) = sqrt(C² - xi²)."""
    roots = []
    n = 0
    while True:
        lo = n * np.pi + 1e-12
        hi = (n + 0.5) * np.pi - 1e-12
        if lo >= C:
            break
        hi = min(hi, C - 1e-12)
        if lo >= hi:
            n += 1
            continue
        f = lambda xi: xi * np.tan(xi) - np.sqrt(C**2 - xi**2)
        if f(lo) * f(hi) < 0:
            roots.append(brentq(f, lo, hi))
        n += 1
    return roots
```

### Testing guidance

- Test with known analytic solutions (e.g. quadratics, trig functions).
- Verify convergence by checking residuals: `|f(x*)| < tol`.
- Test edge cases: functions with multiple roots, flat regions, discontinuities.

### Deprecated / removed APIs

**CRITICAL: These common functions have been removed in recent versions.**
Check the Environment section in the prompt for installed versions.

| Removed function | Replacement | Removed in |
|-----------------|-------------|------------|
| `numpy.trapz()` | `scipy.integrate.trapezoid()` | NumPy 2.0 |
| `scipy.integrate.trapz()` | `scipy.integrate.trapezoid()` | SciPy 1.14 |
| `numpy.bool` / `numpy.int` / `numpy.float` | `bool` / `int` / `float` | NumPy 1.24 |
| `numpy.complex` | `complex` | NumPy 1.24 |
| `numpy.object` | `object` | NumPy 1.24 |
| `numpy.str` | `str` | NumPy 1.24 |

For numerical integration, **always** use:
```python
from scipy.integrate import trapezoid
# Usage:
result = trapezoid(y, x)
```

Do **NOT** use `np.trapz`, `numpy.trapz`, or `scipy.integrate.trapz` —
they no longer exist in current versions.
