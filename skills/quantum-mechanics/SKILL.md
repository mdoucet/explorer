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

### Testing guidance

- Compare eigenvalues against analytically known limits (e.g. infinite-well
  limit: $E_n = \frac{n^2\pi^2\hbar^2}{2m(2a)^2}$).
- Use tolerances appropriate for the numerical method (typically `rtol=1e-6`).
- Test both even-parity and odd-parity states separately.
- Verify normalisation: `np.trapz(np.abs(psi)**2, x)` ≈ 1.
