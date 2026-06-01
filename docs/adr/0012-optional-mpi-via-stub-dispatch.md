# ADR-0012: `mpi4py` is optional; production code imports through `cosmocore._mpi`

## Status

Accepted — landed in PR #27 (2026-05-13). CI matrix split per package (mpi / nompi) added in PR #29 to keep both branches exercised.

## Context

Before this decision, every Python process that imported `cosmocore` transitively pulled `from mpi4py import MPI` from `cosmocore.mpi_utils`, which eagerly initialises the MPI runtime at module import. Three forcing functions made that untenable:

- **Broken MPI runtimes on fresh installs.** OpenMPI 5.0.x on stock Ubuntu produced minutes-long import hangs on `unix_wait_for_peer` sockets. The hang surfaced even for users who never intended to run under `mpirun`.
- **Single-process cost.** Notebooks, the docs build, and every single-process pytest invocation paid an MPI init they did not use.
- **Install friction.** `mpi4py` is the most fragile dependency in the stack on systems without a working MPI development toolchain. Requiring it just to import the library raised the floor for casual users (students, paper readers reproducing a figure, CI on minimal images).

Two obvious alternatives were considered:

1. **Hard-require `mpi4py` and document workarounds.** Status quo; rejected on the symptoms above.
2. **Lazy import inside MPI-using functions.** Cleaner in spirit but doesn't compose with the codebase's `MPISharedMemoryMixin`, which references `MPI.Win`, `MPI.COMM_TYPE_SHARED`, and `MPI.UNDEFINED` at class-definition time, plus dozens of `comm.bcast` / `Bcast` / `Allreduce` call sites that would each need their own guard.
3. **Vendor `mpi4py-stubs` from PyPI for typing only.** Doesn't help — those stubs are type-only and don't provide runtime no-op semantics, so they don't solve the import-hang or the install-friction problem.

## Decision

`mpi4py` is moved out of every package's runtime requirements and into an opt-in `mpi` extra. A dispatch layer at `cosmocore/_mpi.py` is the **single import boundary** between production code and `mpi4py`:

```python
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    from . import _mpi_stub as MPI
    HAS_MPI = False
```

All production modules (`cosmocore.mpi_utils`, `qube.fisher`, `qube.spectra`, `picslike.picslike`, …) import `from cosmocore._mpi import MPI`. Nothing under `src/` imports `mpi4py` directly.

Concretely, the contract has four parts:

1. **Single import boundary.** Every `from mpi4py import MPI` in production code becomes `from cosmocore._mpi import MPI`. Direct imports of `mpi4py` are reserved for *non-library* code: cluster benchmark scripts under `benchmarks/` and `outputs/` that are only ever run under `mpirun`.
2. **Stub mirrors the *used* surface, no more.** `_mpi_stub.py` provides exactly the `mpi4py.MPI` symbols the library calls — currently `COMM_WORLD`, `COMM_NULL`, `COMM_TYPE_SHARED`, `SUM`, `UNDEFINED`, `Comm`, `Win`. Adding a new MPI primitive to library code means adding a no-op shim in the stub in the same PR. The stub is not a parallel MPI implementation; it is a single-rank semantic identity (collectives return their input, `Allocate_shared` returns a local heap buffer).
3. **Real-MPI behaviour is unchanged.** When `mpi4py` is installed, the dispatch falls through to the C library and collectives behave bit-for-bit as before. This ADR does not touch the `MPISharedMemoryMixin` algorithm — only its import source.
4. **`HAS_MPI` is the exported runtime flag.** Downstream code that wants to gate MPI-only printouts or guard against accidental collective ops in single-process mode reads `HAS_MPI`; it does not introspect `MPI.__name__` or feature-test for `Allocate_shared`.

Packaging:

- `cosmoforge.cosmocore` declares `mpi4py` under `[project.optional-dependencies].mpi`.
- `qube`, `picslike`, and the `cosmoforge` umbrella metapackage each expose a passthrough `mpi = ["cosmocore[mpi]"]` extra so opt-in works at any installation level.
- `uv sync` → no `mpi4py`, stub path active. `uv sync --all-packages --all-extras --dev` → real `mpi4py`.

CI exercises both branches: a per-package matrix (mpi / nompi) runs the suite under both the real and stub paths so a stub-surface drift breaks visibly.

## Consequences

- **Reproducing a paper figure no longer requires a working MPI toolchain.** Casual users, readers, students, and minimal CI images can install and run.
- **Cluster behaviour is unchanged.** Production runs under `mpirun -n N` import the real `mpi4py` and execute the same collectives as before — there is no parallel-implementation branch to drift.
- **Stub surface is a maintenance contract.** A library PR that adds `MPI.LAND` (for example) without a matching stub entry will pass tests under real-MPI and fail under the stub branch. The CI matrix is the enforcement mechanism.
- **Two rank-mock tests** in `qube/tests/test_fisher.py` and `test_spectra.py` need the real `MPI.LAND` sentinel for their mocking (a symbol unused in production). They gate themselves behind `pytest.importorskip("mpi4py.MPI")`. Adding more such tests is fine; they should follow the same pattern rather than being added to the stub.
- **The `HAS_MPI` flag is the only sanctioned way to ask "am I running under real MPI?"** at runtime. Library code that needs MPI-aware behaviour (e.g. printing only on rank 0 *and only when there are multiple ranks*) checks `HAS_MPI` plus `comm.size`.

## References

- PR #27 (the rewrite), PR #29 (CI matrix mpi/nompi).
- `cosmocore/_mpi.py`, `cosmocore/_mpi_stub.py`, `cosmocore/tests/test_mpi_stub.py`.
- `feedback_mpi4py_mpich_thinkpad.md` / `project_mpi4py_mpich_thinkpad.md` — the immediate forcing function on the developer machine.
- ADR-0008 — the analogous single-source-of-truth principle for dense linear algebra; this ADR applies the same pattern to the MPI surface.
