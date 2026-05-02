# ADR-0002: Computation basis abstraction; rename from "compression"

## Status

Accepted — implemented and merged to master 2026-04-19.

## Context

The original code modelled the harmonic transform `V` as a
"compression": user-facing flag `compression="harmonic"`, directory
`cosmocore/compression/`, manager class `compression_manager`. This
naming was wrong on two counts:

1. The harmonic transform from pixels to a/b harmonic modes is a
   **basis change**, not a compression. It is exact when n_modes ≥
   n_pix, and it can in fact *expand* the problem at small fsky
   (n_pix < n_modes) — the opposite of compression.
2. Real compression (lossy approximations like m-block, eigenmode
   truncation) is a property *within* a basis, not the same thing as
   the basis itself. Conflating the two made it impossible to talk
   about "harmonic basis with no compression" or "pixel basis with
   eigenmode truncation".

## Decision

Restructure the abstraction:

- `ComputationBasis` (ABC) with concrete subclasses `HarmonicBasis`
  and `PixelBasis`. Each basis is responsible for the basis change
  itself (V operator construction, Λ block layout).
- Each basis has *optional* compression: m-block (harmonic) or
  eigenmode truncation (pixel). Compression flags live on the basis,
  not on the manager.
- Directory rename `compression/` → `basis/`. Module rename
  `compression_manager` → `basis_manager`. Method rename
  `setup_compression(...)` → `setup_computation_basis(...)`.
- New API:
  `setup_computation_basis(basis="auto"|"harmonic"|"pixel",
   compress=False, delta_m=0, threshold=None)`.

The V operator is built m-ordered (modes grouped by `|m|`, not by ℓ).
This is a prerequisite for m-block compression and is harmless for
the uncompressed path.

## Consequences

- **Public API break**: `compression="harmonic"` and the old
  `setup_compression` signature are gone. Users on older snapshots
  must rename. Acceptable because the project is pre-1.0 and there
  was no published external API surface yet.
- **Conceptual clarity**: "basis" and "compression" are now two
  orthogonal axes. New compressions can be added per-basis without
  growing the top-level flag space.
- **Enables follow-on work**: pixel basis direct mode (see ADR-0003),
  m-block compression (single-field spin-0), field block-diagonal K
  detection (auto-detected, no flag).
- **Documentation**: any reference to "harmonic compression" in
  comments, docstrings, or paper drafts must be updated to "harmonic
  basis". Compression now refers only to *lossy* approximations
  within a basis.
