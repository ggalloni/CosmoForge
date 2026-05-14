# ADR Index

Architectural decisions for CosmoForge. Each ADR is immutable once accepted;
revisit by writing a new ADR that supersedes or extends an earlier one.

See [`CONTEXT.md`](../../CONTEXT.md) for the domain language used in these
ADRs, and [`docs/agents/domain.md`](../agents/domain.md) for the
agent-facing convention that governs when an ADR should be written.

## By topic

### Basis architecture
- [ADR-0002 — Computation basis abstraction](0002-computation-basis-abstraction.md). `HarmonicBasis` and `PixelBasis` behind a `ComputationBasis` ABC; replaces the legacy "compression" framing.
- [ADR-0003 — Pixel basis direct mode and `method="auto"`](0003-pixel-basis-direct-mode.md). Direct pixel-space path bypassing V; cost-based selector compares `n_modes³` vs `(n_bins+1)·n_pix³`. fsky ≈ 0.35 crossover.

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

## Status snapshot

| Status | ADRs |
|---|---|
| Accepted | 0001, 0002, 0003, 0004, 0005, 0006, 0007, 0008, 0009, 0011 |
| Proposed / deferred | 0010 |
| Superseded | — |

No supersession chains in place; all current ADRs are additive.

## Adding a new ADR

Next number is **0012**. Use the pattern `NNNN-kebab-title.md`.

1. First line: `# ADR-NNNN: {title}`.
2. Sections: **Status** (Accepted / Proposed / Superseded; date and merge target if known), **Context**, **Decision**, **Consequences**. Optional: **References**, **Validation**, **See also** (related memory files and code paths).
3. Write a new ADR when the decision is **hard to reverse and surprising without context** — not for routine choices that are obvious from the code.
4. Update this index when adding (topic placement + status table).
5. If the new ADR supersedes or modifies an earlier one, link both directions: the new ADR's Status references the old; the old ADR's Status appends a "Superseded by ADR-NNNN" note. Add a row to the supersession chain table above when this happens.
