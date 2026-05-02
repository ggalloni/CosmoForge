# ADR-0004: Sparse-COO derivative storage and O(lmax⁴) Fisher trace

## Status

Accepted — implemented 2026-04-22 in `qube/fisher.py`.

## Context

The Fisher matrix entry `F_{ij} = (1/2) Tr[A E_i A E_j]` (with
`A = V C⁻¹ Vᵀ`) was computed as a dense O(n_modes²) operation per
trace, giving overall `O(lmax⁶)` scaling. At nside=32 / lmax=64 this
trace stage dominated wall time (~4400s of a 4800s QU run on the
g100 cluster). ECLIPSE (Bilbao-Ahedo+ 2021) reports a "factor 1000"
reduction in this stage via symbolic trace manipulation, but the
construction there assumes diagonal noise — not viable for our
multi-component / correlated-noise targets.

Each derivative `E_ℓ` has only O(2ℓ+1) nonzero entries: it touches
only the harmonic-mode indices belonging to multipole ℓ. Two patterns:

- Diagonal spectra (TT/EE/BB): triplets `(k, k, weight)`.
- Off-diagonal spectra (EB/TE/TB): triplets `(k, k+n_base, w)` and
  `(k+n_base, k, w)`.

A trace `Tr[A E_i A E_j]` then collapses to a sub-block extraction
from `A`:

```
Tr[A E_i A E_j] = Σ_{αβ}  v_α v_β  A[c_β, r_α]  A[c_α, r_β]
```

where `(r_α, c_α, v_α)` are the COO triplets of `E_i`. Cost per
entry becomes `O(N_i × N_j)` with `N` the nonzero count — i.e.
`O(lmax²)` per trace, `O(lmax⁴)` total.

## Decision

Store all per-ℓ harmonic derivatives as sparse COO triplets and
compute Fisher traces by direct sub-block extraction.

- All derivatives stored as `(rows, cols, vals)` triplets in a
  `sparse_coo_data` dict keyed by spectrum index.
- Diagonal spectra emit `(k, k, weight)` triplets per mode.
- Off-diagonal spectra emit symmetric pairs `(k, k+n_base, w)` and
  `(k+n_base, k, w)`.
- Unified trace evaluation:
  `np.einsum("ji,ij,i,j->", M1, M2, v_i, v_j)`
  where `M1 = A[c_β, r_α]` and `M2 = A[c_α, r_β]` are extracted from
  the dense `V_Cinv_VT` block.
- Implemented in both `_compute_single_spectrum` and
  `_compute_multi_spectrum`. The trace loop checks `sparse_coo_data`
  and falls back to dense only if no triplets are present.

The optimisation is mathematically exact — it is a different way of
evaluating the same trace, not an approximation.

## Consequences

- **Algorithmic complexity**: Fisher trace stage drops from O(lmax⁶)
  to O(lmax⁴). Matches ECLIPSE's algorithmic efficiency.
- **No diagonal-noise restriction**: unlike ECLIPSE's symbolic
  manipulation, sub-block extraction works with any pre-computed
  `A = V C⁻¹ Vᵀ`. Multi-component analyses with correlated noise
  benefit equally.
- **Measured speedups** (g100 cluster, 48 cores, single rank):
  - QU nside=16: traces 71s → 0.3s (237×); total 92s → 22s (4.2×).
  - QU nside=32: traces ~4400s → 5.2s (~850×); total ~4800s → 455s
    (~10×).
  - T  nside=32: traces 125s → 0.3s (418×); total 227s → 99s (2.3×).
- **Bottleneck shift**: setup (signal-matrix construction, covariance
  I/O, V construction, K Cholesky) now dominates. Future
  optimisations should target setup, not traces.
- **Cost model dependency**: the harmonic-side cost in
  `_auto_pick_method` (ADR-0003) is `n_modes³` *because* sparse
  traces made the trace stage essentially free. Reverting this
  optimisation would change the cost expression and shift the
  auto-selection threshold.

## Reference

Bilbao-Ahedo et al. 2021 (ECLIPSE), "Speeding up QML…" — the same
O(lmax⁴) target via symbolic trace manipulation under a
diagonal-noise assumption. CosmoForge reaches the same complexity
without that assumption.
