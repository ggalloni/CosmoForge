# ADR-0011 — SymmetryMode flag for cross-component spin-2 EB spectra

**Status:** Accepted  
**Date:** 2026-05-11

---

## Context

For two spin-2 components (i, j) both with spin=2, four EB-type slot-pair combinations exist in the
Fisher/Spectra output: GG (EE), CC (BB), GC (E_i×B_j), and CG (B_i×E_j). GC and CG are genuinely
independent observables — they involve different alm products from different fields — and are not
related by any algebraic identity for real-valued fields on the sphere.

All other cross-spectrum cases resolve the direction unambiguously:
- Spin-0 × spin-0: one slot per component, no ambiguity.
- Spin-0 × spin-2: direction fixed by which component is spin-0 vs spin-2 (component declaration).
- Single spin-2 auto: C_ℓ^{E₀B₀} = C_ℓ^{B₀E₀} by Hermiticity.

HEALPix `alm2cl*` defaults to unsymmetrized output (`symmetric=.false.`), emitting separate TE, ET,
EB, BE spectra. Symmetrized output is opt-in (`symmetric=.true.`). However, HEALPix operates at the
raw alm level without physics assumptions. CosmoForge is a CMB power-spectrum estimator with
physics-informed defaults.

## Decision

Add a `SymmetryMode` enum to `cosmocore.spectrum_key` with two values:

- `SYMMETRIC` (default): emits one combined GC spectrum per cross-component spin-2 pair. The model
  covariance (Lambda) uses a single `C_EB` in both off-diagonal blocks. Reproduces the behaviour of
  all code predating this ADR bit-for-bit.
- `DIRECTIONAL` (opt-in): emits both `GC` and `CG` as separate spectra. Lambda uses `C_GC` for the
  (E_i, B_j) block and `C_CG` for the (B_i, E_j) block independently.

The flag lives on `Fisher` (set at construction time). `Spectra` inherits it from its Fisher
instance; it does not accept an independent `symmetry_mode` argument.

## Rationale

**Why SYMMETRIC is the default:**  
Standard cosmology conserves parity — C_ℓ^{EB} = 0 for every map and frequency pair. With GC = CG
= 0, the two modes are numerically identical; DIRECTIONAL adds one extra spectrum, one extra E
matrix, and one extra Fisher row/column for no information gain in the common case. Users who need
the distinction know they need it.

**Why DIRECTIONAL exists:**  
At the map level, GC and CG become different when polarisation angle calibration differs between two
bands. A miscalibration angle θ on band 1 (band 0 perfectly calibrated) produces:

    C_ℓ^{E₀B₁'} ≈ −sin(2θ) · C_ℓ^{EE}
    C_ℓ^{B₀E₁'} ≈ +sin(2θ) · C_ℓ^{EE}   (opposite sign)

The symmetric average (GC + CG)/2 ≈ 0 — the miscalibration cancels. GC − CG ≈ −2 sin(2θ)·C_ℓ^{EE}
is the diagnostic. DIRECTIONAL mode exposes this; SYMMETRIC hides it.

**Two layers must change for DIRECTIONAL:**  
Both the derivative E matrix (`_build_derivative_matrix_with_spins`) and the Lambda matrix
(`_build_lambda_matrix_keyed` / `_build_lambda_block_spin2`) encode the signal model. Fixing only
the E matrix leaves Lambda using a single C_EB for both off-diagonal blocks, producing a misspecified
covariance and biased estimates.

**SYMMETRIC is not a misspecified model in standard cosmology:**  
With C_EB = 0, the SYMMETRIC covariance is exact. "Misspecification" only arises when the true
universe has GC ≠ CG (birefringence, systematics), which is the opt-in case.

**Literature precedent:**  
The directional treatment is standard in CMB cosmic-birefringence analyses, where per-band
miscalibration angles α_i make C_ℓ^{E_i B_j} ≠ C_ℓ^{E_j B_i}. The framework was introduced by
Minami et al. 2019 (PTEP 2019, 083E02; arXiv:1904.12440) and Minami & Komatsu 2020
(PRL 125, 221301; arXiv:2011.11254), which states explicitly: *"We have 32 independent equations
from 16 combinations of maps, as we have two different equations for C_ℓ^{E_i B_j,o} and
C_ℓ^{E_j B_i,o}."* The asymmetric rotation matrix D(α_i, α_j) couples {EE, BB} into {EB, BE}
differently under (i ↔ j) swap. The same machinery is inherited by Diego-Palazuelos et al. 2022
(arXiv:2201.07682) — "28 unique pairs for EE and BB, 56 unique pairs for EB" reflects the ordered-
vs-unordered distinction — and Eskilt 2022 (arXiv:2201.13347), Eskilt & Komatsu 2022
(arXiv:2205.13962). These papers operate on Gaussian-on-bandpower cross-spectrum likelihoods at
ℓ ≥ 51; DIRECTIONAL mode here exposes the same physics in the pixel-level QML / exact-likelihood
regime where the bandpower approximation fails.

## Scope

DIRECTIONAL adds N(N-1)/2 extra spectra for N spin-2 components (one CG per cross-pair). For the
typical two-band case N=2 this is one extra spectrum. All spin-0 paths, spin-0 × spin-2 paths, and
single-field spin-2 auto-spectra are identical in both modes.

## Consequences

- `SpectrumKind.CG` becomes reachable in `SpectraManager.build_inputs` only when
  `symmetry_mode=SymmetryMode.DIRECTIONAL` and `comp_i != comp_j`.
- `kind_to_legacy_mode(SpectrumKind.CG, is_cross=True)` returns 2 (cross-pair ordering
  `[GG=0, GC=1, CG=2, CC=3]`); the auto-pair encoding still has no slot for CG, so the
  default call `kind_to_legacy_mode(SpectrumKind.CG)` continues to raise `NotImplementedError`.
- `_build_lambda_block_spin2` gains an optional `C_CG` parameter; callers passing a single
  positional `C_EB` continue to work unchanged (SYMMETRIC behaviour: C_CG defaults to C_GC).
- Setting `symmetry_mode` on `Spectra` independently of its `Fisher` is an error; the flag must
  be consistent (Spectra inherits from Fisher).
- Implementation: Slice 5 of `.claude/plans/2026-05-10-spectrum-key-types.md`.
