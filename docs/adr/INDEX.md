# ADR Index

Architectural decisions for CosmoForge. The **Decision** of an accepted
ADR is immutable: changes to what was decided require a new ADR that
supersedes or extends the earlier one. Implementation amendments
(documenting how the decision was realised, dated follow-up work, or
clarifications that don't alter the original decision) are recorded
in an "Update (YYYY-MM-DD)" section appended to the original ADR.

See [`CONTEXT.md`](../../CONTEXT.md) for the domain language used in these
ADRs, and [`docs/agents/domain.md`](../agents/domain.md) for the
agent-facing convention that governs when an ADR should be written.

## By topic

### Basis architecture
- [ADR-0002 — Computation basis abstraction](0002-computation-basis-abstraction.md). `HarmonicBasis` and `PixelBasis` behind a `ComputationBasis` ABC; replaces the legacy "compression" framing.
- [ADR-0003 — Pixel basis direct mode and `method="auto"`](0003-pixel-basis-direct-mode.md). Direct pixel-space path bypassing V; cost-based selector compares `n_modes³` vs `(n_bins+1)·n_pix³`. fsky ≈ 0.35 crossover.
- [ADR-0018 — Auto basis is the true default; `compression=` → `basis=`](0018-auto-basis-default-and-kwarg-rename.md). Realises ADR-0003's auto default at the orchestration layer; renames the constructor kwarg; establishes the post-1.0 deprecation-shim policy (retires ADR-0002/0013 no-shim precedent).

### Numerical stability and linear algebra
- [ADR-0001 — Stable SMW form](0001-stable-smw-form.md). `M (I + ΛM)⁻¹` replaces the unstable subtractive form `M − M K⁻¹ M` for the projected inverse at high SNR.
- [ADR-0008 — Linalg single source of truth](0008-linalg-single-source-of-truth.md). All dense linear algebra routes through `cosmocore/basics/linalg.py` wrappers (`cholesky_factor`, `cholesky_solve`, `matrix_inverse_symm`, etc.).

### Performance
- [ADR-0004 — Sparse-COO derivatives](0004-sparse-coo-derivatives.md). Sparse-COO storage for harmonic-basis derivative matrices; reduces Fisher trace evaluation to O(lmax⁴).

### Bandpower / binning / inference
- [ADR-0005 — Bandpower window matrix](0005-bandpower-window-matrix.md). Window matrix exposed for parameter-level inference; downstream likelihoods can convolve theory consistently.
- [ADR-0007 — `Bins` class attribution](0007-bins-attribution.md). Implementation adapted from xQML (Vanneste+ 2018, GPLv3). Includes the Bond/Jaffe/Knox initial-binning vs rebinning distinction.

### Conventions and APIs
- [ADR-0006 — Physical C_ℓ inputs](0006-physical-cl-convention.md). Standard CAMB/CLASS-style C_ℓ; the `(2ℓ+1)/(4π)` factor is absorbed into the Legendre basis functions, not pre-multiplied onto inputs.
- [ADR-0009 — Multipole-range API](0009-multipole-range-api.md). Multipole-range type; accepted 2026-05-07 (implementation tracked as backlog item #4).
- [ADR-0010 — Per-spectrum multipole windows](0010-per-spectrum-multipole-windows.md). **Proposed (deferred).** Per-spectrum-key multipole bands for heterogeneous joint analyses.
- [ADR-0011 — SymmetryMode for cross-EB](0011-symmetry-mode-cross-eb.md). `SYMMETRIC` default (GC averaged) vs `DIRECTIONAL` (GC/CG kept separate) for spin-2 × spin-2 cross-component spectra.
- [ADR-0013 — `SpectrumKey` is the single public identity for spectra](0013-spectrum-key-public-identity.md). Frozen-dataclass `SpectrumKey(comp_i, comp_j, kind, spins=...)` replaces the legacy `(comp_i, comp_j, mode)` triple across all keyed APIs; Numba kernels keep int-mode signatures.

### Inputs, outputs, and persistence
- [ADR-0015 — Opt-in persistence](0015-opt-in-persistence.md). No computed quantity is written unless the caller provides an output path; `out*` defaults become `None` and the four write helpers no-op on a falsy path. Hard cut from the old write-by-default behaviour.
- [ADR-0016 — Fisher→Spectra handoff transport](0016-fisher-spectra-handoff-transport.md). In-memory alias of the live Fisher's covariances is the primary handoff; the `out*` files remain a dormant read adapter (two-job transport). Whole-Fisher serialization deferred (unpicklable comm/logger; window-fns re-invoke `compute()`).
- [ADR-0017 — File-or-array loading seams](0017-file-or-array-loading-seams.md). Two adapters per input seam (params path + injected in-memory object); dispatch and semantic validation live in per-input `Core._resolve_<input>()` methods; readers stay pure parsers; fixed injection-kwarg vocabulary (`mask`, `noise_cov1/2`, `maps1/2`, `cls_data`, `fiducial_cls`, `beam`). Adopts the two-layer (high/low) interface split.

### Packaging and runtime
- [ADR-0012 — Optional `mpi4py` via stub dispatch](0012-optional-mpi-via-stub-dispatch.md). Single import boundary at `cosmocore._mpi`; `mpi4py` becomes an opt-in `mpi` extra; CI matrix exercises both branches.
- [ADR-0014 — MPI broadcast and shared-memory conventions](0014-mpi-broadcast-and-shared-memory.md). `_shared_array` (read-only, intra-node zero-copy) vs `_bcast_array` (writable or > 2 GB buffer) vs `comm.bcast` (Python objects, ≤ 2 GB); window lifecycle owned by `MPISharedMemoryMixin`.

## Status snapshot

| Status | ADRs |
|---|---|
| Accepted | 0001, 0002, 0003, 0004, 0005, 0006, 0007, 0008, 0009, 0011, 0012, 0013, 0014, 0015, 0016, 0017, 0018 |
| Proposed / deferred | 0010 |
| Superseded | — |

No supersession chains in place; all current ADRs are additive.

## Adding a new ADR

Next free number is **0019**. Use the pattern `NNNN-kebab-title.md`.

1. First line: `# ADR-NNNN: {title}`.
2. Sections: **Status** (Accepted / Proposed / Superseded; date and merge target if known), **Context**, **Decision**, **Consequences**. Optional: **References**, **Validation**, **See also** (related memory files and code paths).
3. Write a new ADR when the decision is **hard to reverse and surprising without context** — not for routine choices that are obvious from the code.
4. Update this index when adding (topic placement + status table).
5. If the new ADR supersedes or modifies an earlier one, link both directions: the new ADR's Status references the old; the old ADR's Status appends a "Superseded by ADR-NNNN" note. Add a row to the supersession chain table above when this happens.

## Amending an existing ADR

For implementation amendments that don't change the Decision (dated
follow-up notes, references to the PR that realised the decision,
clarifications, deprecation notices on now-removed flags), append an
``## Update (YYYY-MM-DD)`` section at the bottom of the original ADR.
The Decision section itself stays untouched; readers see the
chronology by reading top-to-bottom.

Anything that changes *what was decided* — flag semantics flipped,
the cost model replaced, a different basis chosen — requires a new
ADR per the supersession protocol above.
