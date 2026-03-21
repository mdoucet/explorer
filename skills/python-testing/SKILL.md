---
name: python-testing
description: Python project structure, packaging, imports, pytest, testing, conftest, pyproject.toml, src layout, flat layout
---

# Python Testing & Packaging

## Overview

Use this skill for **every task** that produces Python code and tests.
It ensures the generated project can be tested in an isolated sandbox
where only `sys.path` manipulation (no `pip install`) may be available.

## Project layout

The verifier runs `pytest` in a **temporary directory** containing only the
files the coder produces.  Imports must work without installation.

### Preferred: flat layout (recommended for single-package projects)

Place the package directory at the project **root** — not inside `src/`.

```
square_well/
    __init__.py
    solver.py
    wavefunctions.py
tests/
    test_solver.py
    test_wavefunctions.py
conftest.py
pyproject.toml
```

With this layout, `from square_well.solver import …` works when the project
root is on `sys.path`.

### Avoid `src/` layout

If you place the package under `src/`, the import path is still the package
name (e.g. `from square_well import …`), but `src/` must be on `sys.path`.
The verifier handles this automatically, but **flat layout is strongly
preferred** because it avoids path issues entirely.

## conftest.py

Do **not** generate `conftest.py` — the test runner (verifier) creates it
automatically with the correct `sys.path` entries for whatever layout was
used.  If you generate your own conftest, the verifier will respect it, but
you risk incorrect path setup.

## pyproject.toml

Always generate a minimal `pyproject.toml` so the project is installable:

```toml
[project]
name = "my-package"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = ["numpy", "scipy"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.backends"
```

For flat layout, **no extra config** is needed — Hatch auto-discovers
package directories at the root.

For `src/` layout, add:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/my_package"]
```

## Import rules

- **Test files** must import from the package name, never from `src.`:
  - ✅ `from square_well.solver import find_bound_energies`
  - ❌ `from src.square_well.solver import find_bound_energies`
- Every package directory must have an `__init__.py`.
- Use relative imports inside a package: `from .solver import solve`.

## Testing rules

- Use `pytest` (not `unittest`).
- Name test files `test_*.py` inside a `tests/` directory.
- Tests must be runnable with `python -m pytest` from the project root.
- Do not depend on any fixtures or conftest from the *host project* —
  the sandbox is isolated.
- Pin tolerances for numerical tests: `np.testing.assert_allclose(result, expected, rtol=1e-6)`.

## Common pitfalls

| Pitfall | Fix |
|---------|-----|
| `ModuleNotFoundError` in sandbox | Use flat layout or generate conftest with `sys.path` |
| Tests import from `src.pkg` | Import from `pkg` — conftest adds `src/` to path |
| Missing `__init__.py` | Always generate `__init__.py` for every package dir |
| `conftest.py` conflicts | Do not generate conftest — let the verifier handle it |
| Tests rely on host-installed packages | Only depend on stdlib + numpy + scipy |
