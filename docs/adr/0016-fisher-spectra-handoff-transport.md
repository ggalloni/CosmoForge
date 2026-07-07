# ADR-0016: Fisher→Spectra handoff transport

## Status

Accepted (2026-07-07; merge target `feat/opt-in-writes`). Sibling of ADR-0015.

## Context

`Spectra` reuses a completed `Fisher`. On the no-basis traditional path
(`basis_manager is None`) it needs two pixel-space matrices from that Fisher:
the reduced noise covariance **N** and the inverted total covariance **C⁻¹**
(used together to form the noise-bias term `C⁻¹ N C⁻¹`). It also reuses, from
the *live* Fisher object, the field collection, geometry, signal matrix, bins,
beam, basis manager, the Fisher matrix, and it calls behaviour-bearing methods
(`get_fisher_matrix`, `get_window_matrix`, `get_bandpower_window_function`) —
the last two *recompute* from live state, re-invoking `Fisher.compute()`.

Historically N and C⁻¹ were transported Fisher→Spectra through disk files
(`outnoisecovmat1/2`, `outinvcovmatfile1/2`): Fisher wrote them, Spectra read
them back with `np.fromfile`. But every `Spectra` construction already holds a
live, fully-run Fisher in-process — either passed via `fisher=` (validated as
run) or self-built via `_get_fisher()` (which calls `run()`). The disk read was
therefore a redundant round-trip into the same process that had just computed
the data. After ADR-0015 made `out*` default to `None`, the default in-memory
run hit `np.fromfile(None)` and crashed — the wall the clean-workdir acceptance
test stopped at.

Two facts constrain the fix:

1. `Fisher.prepare_covariance_matrices` overwrites `noise_cov1` **in place** with
   C⁻¹ during inversion, so the original reduced **N** is dropped from memory
   after `run()`. An in-memory handoff first requires Fisher to *retain* N.
2. A genuine two-job HPC split (job 1 = Fisher, job 2 = Spectra, job 2 skipping
   recomputation) cannot be served by reading four covariance files: Spectra's
   window-function methods re-invoke `Fisher.compute()` over near-total live
   state, and `Fisher` holds an MPI communicator and a file-handle logger that
   do not pickle. Skipping recomputation is whole-Fisher (de)serialization, a
   separate substantial feature — not a covariance-file read.

## Decision

**1. In-memory handoff is the primary transport.** `Fisher` retains the reduced
noise as `reduced_noise_cov1` / `reduced_noise_cov2` — a reference kept just
before the in-place inversion, so the array already existed and no new
allocation is made. On the traditional path `Spectra` **aliases** the live
Fisher's arrays: `inv_cov1 ← fisher.noise_cov1` (C⁻¹) and
`noise_cov1 ← fisher.reduced_noise_cov1` (N), plus the `…2` pair for cross.
These arrays are read-only downstream (only `matrix_mult`, which returns fresh
arrays), so aliasing is safe and no defensive copy is made. Net memory is **one
fewer pix² array** than the disk round-trip, which materialised a second copy of
C⁻¹ on read.

**2. The disk files remain a dormant read adapter.** `Spectra` resolves the
covariances in priority order: (i) the live Fisher's retained arrays; (ii) the
`out*` paths via `np.fromfile` (the opt-in two-job transport of ADR-0015);
(iii) otherwise raise, naming both options. Today (i) always wins for a run
Fisher, so (ii) is reachable only for a future Fisher-like object that lacks the
in-memory arrays. The seam and priority exist and are tested; the branch is
kept, not exercised by the default pipeline.

**3. Whole-Fisher serialization is deferred, not built.** Neither pickling the
`Fisher` object nor saving-and-reconstructing its pieces to let job 2 skip
recomputation is implemented, because: (a) `Fisher` holds an unpicklable MPI
communicator and file-handle logger; (b) the window-function methods re-invoke
`compute()` over near-total live state, so a resumed Fisher must carry that state
anyway — a piecemeal covariance load would save only the O(n³) inversion and the
Fisher-matrix trace loop while recomputing geometry, signal, basis, and
derivatives; (c) pickled scientific state is brittle across numpy/Python/cluster
stacks. No current caller needs it. This ADR records the analysis so future
reviews do not re-propose it without new motivation.

## Consequences

- A default run with no `out*` paths completes without touching disk; the
  clean-workdir acceptance test (`test_fisher_spectra_leave_workdir_untouched`)
  passes and is un-xfailed. This is B2's definition of done.
- Only the no-basis traditional path is affected. The harmonic and
  pixel-direct-*basis* paths never read these files (`basis_manager is not
  None`); they obtain N through the basis manager. `qube/memory_budget.py`
  models only those basis paths, so no budget stage changes — a clarifying
  comment is the only budget-side edit.
- The gated write sites in `Fisher.prepare_covariance_matrices` (ADR-0015) still
  emit the files when paths are set, feeding adapter (ii) for the two-job split.
- A two-job resume that skips recomputation stays a future feature; the disk
  files today let job 2 *validate against* or *inspect* the covariances, not
  avoid recomputing them.

## References

- ADR-0015 — Opt-in persistence (sibling; makes these files opt-in artifacts).
- `.claude/plans/2026-07-06-in-memory-pipeline-master-plan.md` — Phase B / Slice B2.
- Code: `qube/fisher.py` (`prepare_covariance_matrices` retains N),
  `qube/spectra.py` (`_reuse_fisher_components` resolution priority,
  `_load_covariance_matrices` = disk adapter).

## See also

- `project_smw_stability_fix` — the SMW form producing C⁻¹.
- `project_memory_budget_calculator` — why only the basis paths are modelled.
