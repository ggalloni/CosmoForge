# ADR-0006: Physical C_ℓ inputs; (2ℓ+1)/(4π) absorbed into Legendre basis

## Status

Accepted — implemented on the `change_normalization` branch and
merged.

## Context

Earlier versions of CosmoForge required users to pre-multiply C_ℓ
by `(2ℓ+1)/(4π)` before passing into the pipeline. The factor was
applied post-hoc by an `apply_normalization()` step that touched
the signal arrays after the fact. Two problems:

1. **User-facing convention mismatch.** CAMB and CLASS produce
   *physical* C_ℓ (no `(2ℓ+1)/(4π)` factor). Every CosmoForge user
   had to remember to multiply, which is exactly the kind of silent
   convention error that produces ~30 dB of wrongness in plots and
   is hard to track back.
2. **Internal complexity.** The derivative chain rule had to track
   the pre-multiplication. `dΛ/dC_ℓ` carried `factor2` / `factor`
   coefficients distinguishing spin types, which propagated into the
   E matrices and made spin-2 sign conventions fragile.

There was also a latent bug: early returns in the Legendre
implementations (`if lmax == 2: return`) bypassed the post-hoc
normalisation step at ℓ=2, silently producing wrong values for the
quadrupole.

## Decision

Absorb the `(2ℓ+1)/(4π)` normalisation into the Legendre basis
functions themselves. C_ℓ inputs are physical (standard CAMB/CLASS
values). `apply_normalization()` becomes a no-op.

Specifically:

- `legendre_plm` returns `sqrt((2ℓ+1)/(4π)) · N_ℓm · P_ℓm`.
- `legendre_00` returns `(2ℓ+1)/(4π) · P_ℓ`.
- `legendre_22` includes `(2ℓ+1)/(4π) · factor2`.
- `legendre_02` includes `(2ℓ+1)/(4π) · factor`.
- The V operator: spin-0 uses normalised `legendre_plm`; spin-2
  uses `scale_ell = sqrt((2ℓ+1)/(4π))` (the spin-weighted Wigner
  d-matrix carries the rest).
- Derivative E matrices use **±1.0** for *all* spin types (not
  `factor2` / `factor`). Λ now stores physical C_ℓ, so
  `dΛ/dC_physical = identity` up to sign.
- The minus sign for spin-0 × spin-2 derivatives (`E = −(₂Y + ₋₂Y)/2`)
  is preserved.

## Consequences

- **API simplification**: users pass standard CAMB/CLASS C_ℓ
  directly. No more `apply_normalization` confusion. Existing user
  scripts that pre-multiplied must be updated to remove the
  multiplication — silent breakage if not.
- **Derivative simplicity**: dΛ/dC = ±1 across all spin types makes
  spin-2 sign-convention errors much harder to introduce.
- **Internal invariant**: Λ blocks store *physical* C_ℓ. The
  `(2ℓ+1)/(4π)` shows up exclusively in the basis functions (V
  operator), not in Λ. Code touching Λ must respect this — anything
  that re-introduces the factor in Λ would silently double-count.
- **Quadrupole bug fix**: by moving normalisation into the Legendre
  functions themselves, the early-return-at-ℓ=2 path is now
  correctly normalised. ℓ=2 results are now consistent with the
  rest of the spectrum.
- **`apply_normalization()` retained as no-op**: kept for API
  compatibility while downstream callers are audited. Future ADR
  will remove it.
