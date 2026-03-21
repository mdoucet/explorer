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

### Testing guidance

- Test with known analytic solutions (e.g. quadratics, trig functions).
- Verify convergence by checking residuals: `|f(x*)| < tol`.
- Test edge cases: functions with multiple roots, flat regions, discontinuities.
