# ADR-0003: Pixel basis direct mode and `method="auto"` selection

## Status

Accepted — merged to master via PR #15 (`pixel-basis-auto`).

## Context

ADR-0002 split the computation into `HarmonicBasis` and `PixelBasis`
abstractions. Both initially routed all algebra through a V operator
(`E_ℓ_basis = V E_ℓ Vᵀ`, `C_basis = V C Vᵀ`) whose dominant cost
scales with `n_modes`. At small sky fractions `n_pix < n_modes` and
the V operator *expands* the problem — running the basis in V form
is then strictly worse than working in pixel space directly.

The "right" basis is not a function of `n_pix` vs `n_modes` alone:

- The harmonic path's bottleneck is the SMW kernel Cholesky, scaling
  as `n_modes³`. Trace evaluation is O(lmax⁴) once sparse-COO
  derivatives land (ADR-0004) and is essentially free in comparison.
- The pixel-direct path's bottleneck is the per-bin `C⁻¹ @ dC^b`
  product, scaling as `(n_bins + 1) × n_pix³`. Binning with a small
  `n_bins` makes it considerably cheaper than the unbinned case.

A useful selector therefore needs to compare these two cost
expressions, not just two dimensions.

For reference: for HEALPix maps `n_pix = n_modes` at `fsky ≈ 0.35`,
independent of `nside`. Below this sky fraction the V operator
strictly expands the problem; above it the dimension argument alone
no longer suffices and the cost-model comparison takes over.

## Decision

Add a direct pixel-space code path inside `PixelBasis` that bypasses
V entirely and works on the pixel-space covariance directly, plus an
`"auto"` method that selects between bases using a leading-order
cost model.

- **Direct vs compressed mode is inferred from construction kwargs,
  not a separate flag.** When `PixelBasis` is constructed with
  `epsilon=None` and `mode_fraction=None`, `setup()` builds the
  direct pixel-space kernels and the V operator is never materialised.
  When either kwarg is non-None, `setup()` builds V, computes
  `P_h = V^T V`, and runs the internal `_apply_compression()` helper
  to eigendecompose the chosen `compression_target`. There is no
  `use_direct` parameter — the configuration is fully determined by
  whether compression is requested.

  Direct mode reuses `do_derivative_step` and `compute_signal_matrix`
  from the no-basis path (the same Numba JIT functions Fisher already
  uses), so there is no second implementation to maintain.

  The previous "V-based but uncompressed" intermediate (no compression
  intent, but V operator built) is no longer expressible — neither as
  a configuration nor as a transient runtime state. It was strictly
  worse than both direct and compressed for any production use.

- Selector `_auto_pick_method(n_pix, n_modes, lmax, n_bins)` in
  `cosmocore/basis/__init__.py` compares
  `cost_harmonic = n_modes³` against
  `cost_pixel = (n_bins + 1) × n_pix³`
  and returns `"harmonic"` or `"pixel"` for the smaller cost. The
  pixel pick passes no compression kwargs, so direct mode is the
  natural default. Dimensions are computed at the **effective lmax**
  (post-lswitch, not the basis lmax). `n_bins` defaults to
  `max(lmax − 1, 1)` (unbinned, the worst case for pixel-direct) when
  the caller does not pass a value.

- Factory: `setup_computation_basis(method="auto", ...)` resolves the
  method via the selector. Forcing `method="harmonic"` or
  `method="pixel"` explicitly emits a warning when the cost model
  says it is suboptimal.

- `method="auto"` is the default for `Core`.

- `lswitch` semantics extended to pixel basis: pixel receives
  `N_eff = N + S_fixed` directly with reduced lmax = params.lmax;
  harmonic still receives N and S_fixed split for SMW.

## Consequences

- **Performance at small fsky / small n_bins**: pixel-direct gives
  optimal scaling whenever `(n_bins + 1) × n_pix³ < n_modes³`.
  Empirically at QU nside=16 lmax=48 fsky=0.10 unbinned, Fisher drops
  from 62s (pixel V-based) and 19s (harmonic) to 3.3s on the same
  laptop.
- **Binning interacts with the choice**: a binned analysis can keep
  pixel-direct optimal at considerably larger fsky than the unbinned
  case suggests. Callers who know `n_bins` should pass it through.
- **Cost model is interlocked with ADR-0004**: the harmonic-side cost
  is `n_modes³` because sparse-COO derivatives reduced trace
  evaluation to O(lmax⁴). Reverting that optimisation would change
  the harmonic cost expression and the auto-selection threshold with
  it.
- **Platform-agnostic**: the V-based paths scale strongly with
  threaded LAPACK (cluster Fisher 6.5s vs laptop 62s — 10× from
  threading). Pixel-direct is bounded by Numba `prange` over pixel
  pairs and a single `n_pix × n_pix` Cholesky, which are CPU-saturated
  on a laptop. Same wall-time on laptop and cluster (~3.5s).
- **Paper messaging**: quote cluster numbers as the realistic case
  (a serious user has threaded BLAS). Pixel-direct's story is
  *robustness across platforms*, not just "fastest in absolute
  terms" — though it does win in both regimes when the cost model
  selects it. See `project_basis_lapack_threading.md` for the full
  numbers.
- **No code duplication**: direct mode reuses Fisher's existing
  no-basis JIT kernels; the basis abstraction owns only the routing
  decision and the FieldCollection wiring.
- **Default behaviour change**: `method="auto"` ships as the default,
  so existing user scripts that did not specify a method now get the
  optimal path automatically. No silent change in numerical results
  (auto picks an exact path either way).

## Validation

- Direct-vs-V-based equivalence test:
  `tests/test_pixel_direct_mode.py`.
- fsky-sweep benchmark:
  `benchmark_pixel_vs_harmonic.py` (laptop + g100 cluster).

## Update (2026-05-14)

The original `_use_direct` flag described above was removed in the
PixelBasis class-model cleanup (architecture backlog #8). Direct mode
is now inferred from the absence of `epsilon` / `mode_fraction` at
construction; the factory's auto-pick returns no extra kwargs for the
pixel pick. The runtime `apply_compression()` entry point became the
internal `_apply_compression(self)` helper called by `setup()`; all
compression intent must be set at construction.

The constructor kwarg formerly named `basis` (the selector for which
matrix is eigendecomposed) became `compression_target` in the same
change, removing the 3-way "basis" vocabulary collision documented in
ADR-0002.

See `project_pixelbasis_class_model_design.md` (memory) for the
design discussion.

## Update (2026-07-07)

The "`method="auto"` is the default for `Core`" decision above was realised
only at `Core.setup_computation_basis()`; the orchestration classes (`Fisher`,
`Spectra`, `PICSLike`) skipped basis setup entirely when their `compression`
kwarg was `None`, so the default run never reached this selector. ADR-0018
closes that gap: `basis=None` (the renamed kwarg) now resolves to
`method="auto"`, the traditional no-basis path survives only behind an explicit
`basis=False`, and `do_cross=True` keeps the default on the traditional path
(the basis layer has no N₂). The "no second implementation to maintain" claim
holds value-level for S and C⁻¹ and for derivatives in exact arithmetic; the
batched binned-derivative kernel reassociates the beam weight, so spin-2
traditional↔pixel-direct agree to 1 ulp rather than bit-for-bit.
