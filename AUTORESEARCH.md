# autoresearch — symfc force constant symmetrization

This is an experiment to have the LLM autonomously research and implement optimizations to accelerate symfc's force constant symmetrization.

## Goal

**Minimize the wall-clock time** of `ph.symmetrize_force_constants(use_symfc_projector=True)` for a Si diamond 4×4×4 supercell (128 atoms, order=2), while preserving numerical correctness (atol=1e-6 vs. baseline).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar26`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files** for full context. Read ALL of these before starting:
   - `CLAUDE.md` — repository context and common commands.
   - `benchmark.py` — the benchmark script. **Read-only. Do not modify.**
   - `src/symfc/api_symfc.py` — main Symfc class.
   - `src/symfc/basis_sets/basis_sets_O2.py` — O2 basis set construction (where most time is spent).
   - `src/symfc/solvers/solver_O2.py` — O2 solver.
   - `src/symfc/spg_reps/spg_reps_O2.py` — space group representations for O2.
   - `src/symfc/utils/eig_tools.py` — eigenvalue solver interface.
   - `src/symfc/utils/eig_tools_sparse.py` — sparse eigenvalue solver (bottleneck).
   - `src/symfc/utils/eig_tools_division.py` — submatrix division algorithm (bottleneck).
   - `src/symfc/utils/eig_tools_core.py` — core eigenvalue decomposition.
   - `src/symfc/utils/utils_O2.py` — O2 utilities including coset projector.
   - `src/symfc/utils/permutation_tools_O2.py` — permutation compression.
   - `src/symfc/utils/translation_tools_O2.py` — translational sum rule projector.
   - `src/symfc/utils/matrix.py` — BlockMatrixNode, sparse dot product utilities.
   - The phonopy interface (to understand the projection): run `uv run python -c "import phonopy.interface.symfc; print(phonopy.interface.symfc.__file__)"` and read the `symmetrize_by_projector()` function.
4. **Create `run_benchmark.py`**: This wrapper instruments `benchmark.py` with timing, memory tracking, and correctness validation. See the template below.
5. **Run baseline**: Execute `uv run python run_benchmark.py > run.log 2>&1` and record results.
6. **Initialize `results.tsv`** with header and baseline row.
7. **Confirm and go.**

Once you get confirmation, kick off the experimentation.

### `run_benchmark.py` template

Create this file during setup. It is **also read-only** after creation — do not modify it during experiments.

```python
"""Benchmark wrapper with timing, memory tracking, and correctness validation."""

import gc
import os
import time
import tracemalloc

import numpy as np
import phonopy
import pymatgen
import pymatgen.io.phonopy
from ase.build import bulk
from phonopy import Phonopy
from pymatgen.io.ase import AseAtomsAdaptor

BASELINE_FC_PATH = "baseline_fc.npy"

# --- Setup (same as benchmark.py) ---
np.random.seed(42)  # Fixed seed for reproducibility
struct = bulk("Si", "diamond", a=5.43) * 1
struct.positions = struct.positions + 0.01 * np.random.randn(*struct.positions.shape)
struct = AseAtomsAdaptor.get_structure(struct)
size = 4
symprec = 0.01
supercell_matrix = [size, size, size]

ph = Phonopy(
    pymatgen.io.phonopy.get_phonopy_structure(struct),
    supercell_matrix=np.diag(supercell_matrix),
    primitive_matrix="auto",
    symprec=symprec,
)
ph.generate_displacements(distance=0.03)
supercells = ph.supercells_with_displacements
if supercells is None:
    raise ValueError("No supercells generated")
force_sets = [np.random.randn(len(supercells[0]), 3)] * len(supercells)
ph.forces = force_sets
ph.produce_force_constants()
fc_before = ph.force_constants.copy()

# --- Benchmark: symmetrize_force_constants ---
gc.collect()
tracemalloc.start()
t0 = time.perf_counter()

ph.symmetrize_force_constants(show_drift=False, use_symfc_projector=True)

wall_clock_s = time.perf_counter() - t0
_, peak_memory_bytes = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_memory_mb = peak_memory_bytes / 1024 / 1024

fc_after = ph.force_constants

# --- Correctness check ---
if os.path.exists(BASELINE_FC_PATH):
    fc_baseline = np.load(BASELINE_FC_PATH)
    max_diff = float(np.max(np.abs(fc_after - fc_baseline)))
    fc_correct = max_diff < 1e-6
else:
    # First run: save baseline
    np.save(BASELINE_FC_PATH, fc_after)
    max_diff = 0.0
    fc_correct = True

# --- Output ---
print("---")
print(f"wall_clock_s:     {wall_clock_s:.4f}")
print(f"peak_memory_mb:   {peak_memory_mb:.1f}")
print(f"fc_correct:       {fc_correct}")
print(f"fc_max_diff:      {max_diff:.2e}")
print(f"fc_shape:         {fc_after.shape}")
```

## Code path overview

When `ph.symmetrize_force_constants(use_symfc_projector=True)` runs, the call chain is:

```
phonopy.symmetrize_force_constants(use_symfc_projector=True)
  → phonopy.interface.symfc.symmetrize_by_projector()
      → Symfc.__init__()  →  SpgRepsO2 (space group reps)
      → Symfc.compute_basis_set(orders=[2])
          → FCBasisSetO2.run():
              1. compr_permutation_lat_trans_O2()    — permutation+translation compression
              2. get_compr_coset_projector_O2()       — rotational symmetry projector
              3. eigsh_projector(proj_rpt)            — eigendecomposition of rotation projector  ★ BOTTLENECK
              4. dot_product_sparse(c_pt, c_rpt)      — composite compression matrix
              5. compressed_projector_sum_rules_O2()   — translational sum rule projector
              6. eigsh_projector_sumrule(proj)         — final eigendecomposition              ★ BOTTLENECK
      → Projection (4 sparse matrix operations):
          fc_sym = fc.ravel() @ compmat
          fc_sym = blocked_basis_set.transpose_dot(fc_sym.T)
          fc_sym = blocked_basis_set.dot(fc_sym.T)
          fc_sym = fc_sym.T @ compmat.T
```

The two **★ BOTTLENECK** steps dominate runtime. They involve:
- Finding block-diagonal structure in sparse matrices
- Solving independent eigenvalue problems per block (scipy.sparse.linalg.eigsh / scipy.linalg.eigh)
- For large blocks (>500×500): a submatrix division algorithm that splits, solves sub-problems, then recombines

## Experimentation

**What you CAN do:**
- Modify any file under `src/symfc/` — architecture, algorithms, data structures, parallelization, everything.
- Add dependencies to `pyproject.toml` (install with `uv sync --extra benchmark`). Stick to well-known packages: numba, joblib, threadpoolctl, scikit-learn, etc.
- Add new files under `src/symfc/`.

**What you CANNOT do:**
- Modify `benchmark.py` or `run_benchmark.py`. They are read-only.
- Break the existing test suite. If in doubt, run `uv run pytest tests/test_api.py -v` to verify.
- Exceed **18 GB peak memory** during the benchmark.
- Break the public API of the `Symfc` class or `FCBasisSetO2`.

**Correctness constraint**: The symmetrized force constants must match the baseline within `atol=1e-6` (`fc_correct: True` in output). An experiment that produces incorrect results is treated as a crash — revert it.

**Simplicity criterion**: All else being equal, simpler is better. A small speedup that adds ugly complexity is not worth it. Removing code and getting equal or better performance is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the speedup magnitude. A 2% speedup from 50 lines of hacky code? Probably not worth it. A 2% speedup from deleting code? Definitely keep.

## Output format

After running `run_benchmark.py`, extract results:

```bash
grep "^wall_clock_s:\|^peak_memory_mb:\|^fc_correct:\|^fc_max_diff:" run.log
```

## Logging results

Log every experiment to `results.tsv` (tab-separated, NOT comma-separated).

Header and 6 columns:

```
commit	wall_clock_s	memory_gb	fc_correct	status	description
```

1. git commit hash (short, 7 chars)
2. wall_clock_s (e.g. 12.3456) — use 0.0000 for crashes
3. peak memory in GB, round to .1f (divide peak_memory_mb by 1024) — use 0.0 for crashes
4. fc_correct: true/false — use false for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	wall_clock_s	memory_gb	fc_correct	status	description
a1b2c3d	14.2300	2.1	true	keep	baseline
b2c3d4e	11.8500	2.3	true	keep	parallelize eigsh block solves with joblib
c3d4e5f	10.1200	19.5	false	discard	approximate eigendecomposition (correctness failed)
d4e5f6g	0.0000	0.0	false	crash	numba JIT on permutation loop (compilation error)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar26`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Make changes to `src/symfc/` with an experimental optimization idea.
3. git commit (commit the code change, NOT results.tsv or run.log).
4. Run the experiment: `uv run python run_benchmark.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
5. Read out the results: `grep "^wall_clock_s:\|^peak_memory_mb:\|^fc_correct:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't fix it after a few attempts, give up on this idea.
7. **Correctness gate**: If `fc_correct: False`, treat this as a failure — revert.
8. **Memory gate**: If `peak_memory_mb > 18432` (18 GB), treat this as a failure — revert.
9. Record the results in `results.tsv` (do NOT commit results.tsv — leave it untracked).
10. If wall_clock_s improved (lower) AND correctness + memory pass → **keep** the commit, advance the branch.
11. If wall_clock_s is equal or worse, or correctness/memory failed → **git reset** back to where you started.

**Timeout**: The benchmark should complete in under 5 minutes. If a run exceeds 5 minutes, kill it (`kill %1` or similar) and treat it as a crash.

**Crashes**: Use your judgment. Typos and missing imports → fix and re-run. Fundamentally broken idea → skip, log "crash", move on.

**After adding a dependency**: Run `uv sync --extra benchmark` before running the benchmark. If the dependency fails to install, revert and move on.

**Periodic test suite check**: Every 3-5 successful experiments, run `uv run pytest tests/test_api.py -v` to make sure nothing is broken beyond the benchmark. If tests fail, revert to the last passing state.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the bottleneck code, try combining previous near-misses, try more radical algorithmic changes. The loop runs until the human interrupts you, period.

## Research directions (starting points)

These are hints, not a prescribed order. Use your judgment.

1. **Parallel block eigensolves**: `eigsh_projector` in `eig_tools_sparse.py` solves independent blocks sequentially. These can be parallelized with `joblib` or `multiprocessing` (up to 20 cores).

2. **Parallel submatrix division**: `eigsh_projector_sumrule` in `eig_tools_division.py` has independent sub-problems that could run in parallel.

3. **Sparse matrix format tuning**: Profile whether CSR vs CSC vs COO matters for the specific operations in the projection chain.

4. **BLAS thread control**: Use `threadpoolctl` to ensure numpy/scipy use all available cores for dense BLAS operations (eigh, matrix multiply).

5. **Algorithmic improvements to eigendecomposition**: The code solves `eigsh` then filters eigenvalues ≈ 1.0. Could a randomized SVD or Lanczos with shift-invert be faster?

6. **Reduce redundant computation**: The code checks for duplicate blocks in `eigsh_projector`. Are there more redundancies to exploit?

7. **Numba JIT**: Hot loops in permutation construction or sparse matrix assembly might benefit from JIT compilation.

8. **Batch sparse operations**: Some operations build sparse matrices element-by-element. Batching COO construction can be much faster.

9. **Memory-layout optimization**: Ensure arrays are contiguous and avoid unnecessary copies in the projection chain.

10. **Caching / precomputation**: If intermediate results can be reused across the basis set construction steps, cache them.

11. **Alternative linear algebra backends**: scipy can use different LAPACK implementations. Check if switching routines helps.

12. **Approximate sum rules**: Could iterative constraint enforcement replace the exact (expensive) projector eigendecomposition?
