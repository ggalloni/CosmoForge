# ADR-0001: Stable SMW form for projected inverse at high SNR

## Status

Accepted — implemented in `cosmocore/basis/harmonic.py`.

## Context

The Sherman–Morrison–Woodbury (SMW) identity is used to compute the
projected inverse `V C⁻¹ Vᵀ` without forming the full pixel-space
covariance, where `K = Λ⁻¹ + M` and `M = Vᵀ N⁻¹ V`. The textbook form

```
V C⁻¹ Vᵀ = M − M K⁻¹ M
```

suffers catastrophic cancellation in the cosmic-variance-limited
regime (signal ≫ noise, `Λ ≫ M⁻¹`): both `M` and `M K⁻¹ M` become
large and nearly equal, the difference loses 9–14 digits in float64,
and produces spuriously **negative** entries in `V C⁻¹ Vᵀ`.

Empirically: at nside=16 multi-spectrum TQU validation with
2.5 µK·arcmin polarisation noise, T at low ℓ has SNR ~10⁹–10¹². The
naive form gave diagonal entries of `V C⁻¹ Vᵀ` for T at −0.09 and
made per-ℓ 6×6 blocks for ℓ ∈ {2, 4, 6} non-PD; QUBE Fisher had four
negative eigenvalues; inversion failed via `dpotrf info=7`.

## Decision

Use the algebraically equivalent stable identities everywhere SMW
algebra is needed:

- Projected inverse: `V C⁻¹ Vᵀ = M (I + Λ M)⁻¹`
- Data weighting:    `V C⁻¹ d   = (I + M Λ)⁻¹ Vᵀ N⁻¹ d`
- Noise-bias matrix: `A = (I + M Λ)⁻¹`

These come from the rewrite `M − M K⁻¹ M = M K⁻¹ Λ⁻¹` followed by
factoring `K = Λ⁻¹ (I + Λ M)`. The matrix `(I + ΛM)` has eigenvalues
`1 + eig(Λ^{1/2} M Λ^{1/2}) ≥ 1`, so it is always well-conditioned and
invertible without regularisation.

Helper: `HarmonicBasis.prepare_stable_inner_inv(C_ell, lambda_matrix=None)`
returns `(I + Λ M)⁻¹`. `_smw_projected_inverse` accepts
`lambda_matrix=` to trigger the stable form.
`Spectra._compute_qml_spectra_compressed` passes `stable_inner_inv`
to data weighting and `_compute_noise_cov_compressed`.

## Consequences

- **Numerical stability**: V C⁻¹ Vᵀ stays PD across the SNR range
  needed for sub-µK polarisation noise. Multi-spectrum TQU Fisher is
  invertible at low ℓ.
- **Cost**: Comparable to the legacy form (1 LU vs 1 Cholesky + dense
  matmul). Do **not** claim a speedup attributable to this change in
  commit messages or paper text.
- **Reference**: Higham, *Accuracy and Stability of Numerical Algorithms*
  (SIAM, 2002), §11 on Woodbury stability.

## Validation

`plot_multi_fsky_teb.py` at fsky ~0.5, nside=16, 2.5 µK·arcmin in P,
r=0.01. Recovers all six TEB spectra (auto + cross) with
`σ_MC/σ_Fisher` within 1σ of unity (1/√2N ≈ 0.022 for N=1000 sims).
