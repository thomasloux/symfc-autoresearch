# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Symfc computes force constants from displacement-force datasets in the supercell approach for phonon calculations, enforcing crystal symmetry constraints via projector-based estimation. Pure Python, built on numpy/scipy/spglib.

**Citation:** Seko & Togo, Phys. Rev. B, 110, 214302 (2024)

## Common Commands

Use `uv` to run all commands (e.g. `uv run python`, `uv run pytest`).

```bash
# Install (editable)
uv pip install -e . -vvv

# Run all tests
uv run pytest -v

# Run tests with coverage (as CI does)
uv run pytest -v --cov=./ --cov-report=xml

# Run a single test file
uv run pytest tests/test_api.py -v

# Run a single test
uv run pytest tests/test_api.py::test_function_name -v

# Include large/slow tests
uv run pytest --runbig -v

# Lint
uv run ruff check src/ tests/

# Lint with auto-fix
uv run ruff check --fix --show-fixes src/ tests/

# Format
uv run ruff format src/ tests/

# Build docs
uv run sphinx-build doc docs_build
```

## Linting & Formatting

Ruff is the sole linter/formatter (line-length 88, numpy docstring convention). Pre-commit hooks run trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, ruff, and ruff-format.

## Architecture

The package lives in `src/symfc/` with four core modules:

- **`api_symfc.py`** — `Symfc` class: the main user-facing API. Takes a supercell (`SymfcAtoms`), displacements, and forces; orchestrates basis sets and solvers via `.run(orders=[2])`.

- **`basis_sets/`** — Symmetry-adapted basis sets for force constants at each order (O1–O4). `FCBasisSetBase` defines the interface; `FCBasisSetO2`, `O3`, `O4` implement order-specific projectors.

- **`solvers/`** — Force constant solvers. Single-order (`FCSolverO2`, `O3`, `O4`), combined-order (`FCSolverO2O3`, `O3O4`, `O2O3O4`), and sparse (`FCSparseSolverO2`). All extend `FCSolverBase`.

- **`spg_reps/`** — Space group representation matrices for N-th order force constants (`SpgRepsO1`–`O4`).

- **`utils/`** — ~30 modules covering eigenvalue tools, permutation/translation/rotation symmetry operations, sparse matrix utilities, cutoff handling, and graph operations. Order-specific utilities follow the `*_O{n}.py` naming pattern.

The "O" suffix throughout the codebase refers to the Taylor expansion **order** (O1 = first-order, O2 = second-order/harmonic, etc.).

## Testing

Tests mirror the source structure under `tests/`. Test fixtures in `conftest.py` provide crystal structures (NaCl, GaN, Si, SiO2, SnO2). Compressed `.xz` files supply reference force constants and datasets. Tests marked `@pytest.mark.big` are skipped unless `--runbig` is passed.

## Key Dependencies

- **numpy/scipy** — core numerics and sparse linear algebra
- **spglib** (>=2.5) — crystallographic symmetry operations
- **phonopy** — used in examples (not a runtime dependency of symfc itself)
