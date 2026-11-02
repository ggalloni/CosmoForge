# Handoff: fiducial anchor for the harmonic basis

> Temporary file. Move it to your local `.claude/plans/` (gitignored) and delete
> it from the branch before opening a PR.

Branch `ccr-86f8d0e7-qerj9w`, based on `master` at `8aa6741` (#102). Not merged,
no PR. Target release: **1.4.0** (new opt-in kwarg), not 1.3.1.

## Goal

The single-process harmonic QUBE run hits a node's RAM. At the QU nside-64,
fsky-60% cell (`n_pix = 59136`, `n_modes = 33274`, lmax 128 inside
lmax_signal 256) the setup peaks at 113 GiB (`memory_budget.py`, measured
113.2). MPI does not help today: setup runs on rank 0 only, and every rank
holds a pickled copy of the basis.

The idea: anchor the SMW algebra at the fiducial covariance
`C0 = N_eff + V^T Λ0 V` instead of at `N_eff`. Everything QUBE needs is
`A = V C^{-1} V^T` (plus `V C^{-1} d` and `T = V C^{-1} N C^{-1} V^T`) at C0, so
at the anchor `M = V C0^{-1} V^T` *is* the answer and the `n_modes³` solve
disappears. Away from C0 the same update works with `ΔΛ = Λ − Λ0`, so
iterative QML keeps an `n_modes³` step per iteration, as today.

## State of the branch

Commit `1515dc0` "feat: experimental fiducial anchor for the harmonic basis":

- `basis={"method": "harmonic", "anchor": "fiducial"}` on Fisher/Spectra;
  `anchor=` on `Core.setup_computation_basis`; `anchor_C_ell=` on
  `HarmonicBasis`. Default stays `"noise"`.
- `cosmocore/basis/harmonic.py`: `_anchor_at_fiducial` (adds `V^T Λ0 V` into
  the base buffer before the Cholesky), `_lambda_for` (Λ − Λ0), `_at_anchor`
  (exact-spectra check, returns M read-only with no solve), anchored `T`
  (drops `M Λ0 M`), guards on the likelihood methods, m-block compression
  refused.
- `cosmocore/basics/linalg.py`: `add_transposed_product` (in-place
  `C += A^T B` via dgemm, ADR-0008 wrapper).
- `cosmocore/core.py`: builds the anchor spectra with
  `spectra_manager.build_inputs(symmetry_mode=...)` after the S_fixed block.
- Tests: `cosmocore/tests/test_harmonic_anchor.py` (35, basis level, T/QU/TQU,
  correlated N, fiducial/near/far, switch path, brute-force pixel reference,
  guards) and `qube/tests/test_harmonic_anchor_pipeline.py` (end-to-end
  Fisher/Spectra/noise bias on nside-4 T/QU/TQU, shortcut taken, accuracy).
  Full suites passed: cosmocore 639, qube 252, picslike 116.
- Changelog entry under Unreleased / Added; CONTEXT.md "SMW anchor" entry.

Follow-up commit: this file and
`src/cosmoforge.qube/benchmarks/validate_anchor_memory.py`.

## What is validated, what is not

Measured with `validate_anchor_memory.py 3000 38` (spin-0 T, correlated N,
switch on, warm JIT; m = 1517):

| | measured | predicted |
|---|---|---|
| today's setup peak | 278 MiB | 276.7 (`memory_budget.py`) |
| committed anchor, setup peak | **295 MiB (worse)** | n/a |
| restructured direct setup peak | 121 MiB intrinsic | 120.9 (n² + mn + m²) |
| Fisher-stage working memory, noise anchor | 70 MiB = 4 m² | (I had said 3 m²) |
| linear algebra, Fisher only | 1.39 → 0.41 s | ~3× |
| linear algebra, with noise bias T | 1.39 → 1.03 s | ~1.3–1.5× |

Accuracy: anchored T Fisher matches the direct pixel-space Fisher to ~1e-13
(eigenvalues); the noise-anchored path reaches ~1e-8 (pinned in the pipeline
tests).

**The committed version is a net loss at setup**: +1 m² peak (dense Λ0 before
sparsifying, plus the `M Λ0 M` correction) and an extra `n² m` dgemm for
`V^T Λ0 V`. Its only gains are the Fisher stage (no solve, no 4 m² transient)
and accuracy. The setup gain needs step 1 below.

Not measured: QU/TQU memory, the pixel filter, MPI, production scale,
conditioning on wide grids.

Corrections to claims made during the session (so they are not repeated):

- "3× fewer ops" omitted the noise-bias T on the direct side. Fisher-only ~3×;
  Fisher + Spectra ~1.3–1.5×.
- Fisher-stage transient is 4 m² (dense Λ + `I + ΛM` + LU copy + result), not 3.
- "Kept 57 → 16.5 GiB" assumes the weighted maps are formed before Z
  (= `V_N_inv^T`, m·n) is dropped. Spectra builds maps after Fisher, so until
  then the kept set is A0 + T0 + Z ≈ 31 GiB at the production cell.

Projected for the production cell after step 1: setup peak ~49 GiB (vs 113),
noise-bias phase ~31 GiB, kept 31 GiB (16.5 once maps are weighted),
Fisher stage −33 GiB of working memory.

## Next steps

1. **Restructure the anchored setup** (replaces `_anchor_at_fiducial` and the
   S_fixed/corr path when `anchor="fiducial"`). Validated design (part B of the
   script):
   - Build the *full* fiducial signal (every ℓ up to lmax_signal) straight into
     one n² buffer. In Core this is the S_fixed block in
     `setup_computation_basis`; `signal_matrix(...)` returns a new array today,
     so it needs an `out=` or the kernel must fill the noise buffer. The kernels
     assign (not accumulate) and fill the lower triangle, which is all
     `dpotrf(lower=1)` reads.
   - Add N into that buffer by column blocks from the covariance file
     (memmap). With an injected noise array (ADR-0017 `noise_cov=`) there is no
     file: keep a copy (+n²) or spill to a temp memmap.
   - `dpotrf` in place → `X = dtrsm(L, V^T)` in place (needs V C-ordered so
     V^T is F-ordered; harmonic_basis builds it F-ordered today, the filtered V
     is already C-contiguous) → `A0 = dsyrk(X)` (symmetrise in place) →
     `Z = dtrsm(L, X, trans)` in place → free L → `T0 = Z^T N Z` with N
     streamed by row blocks. `V_N_inv` is `Z^T`; keep Z until Spectra has
     weighted the maps.
   - Wrap the BLAS calls in `basics/linalg.py` (ADR-0008).
   - numba specialises the kernel on array layout: an F-ordered buffer
     triggers one extra compile (~3 s), harmless but it skews benchmarks.
2. **Memory regression test**: a tracemalloc test at n ≈ 1000 pinning the
   anchored setup peak to `n² + mn + m²` plus slack, so a claim is checked
   before it is made. Reuse the `Steps` helper from the script.
3. **`memory_budget.py`**: add the Fisher-stage harmonic transient (4 m²,
   currently unmodelled) and an anchored-path budget; recalibrate on a cluster
   run.
4. **ADR** (next free number after 0020): anchor choice, conditioning caveat
   (ΔΛ indefinite loses the eigenvalues ≥ 1 guarantee of `I + ΛM`), likelihood
   guard, N streaming requirement. Cite it in the changelog entry.
5. **Cluster A/B** on the production QU cell: peak RSS per stage
   (`fisher.basis_setup`, `fisher.compute.*`) and wall time, anchor noise vs
   fiducial.
6. **Optional: likelihood (PICSLike)** on the anchor. Determinant lemma
   `ln|C| = ln|C0| + ln|I + ΔΛ A0|` and Woodbury around C0. Use the symmetric
   form `I + A0^½ ΔΛ A0^½` (PD whenever C is) to keep Cholesky cost per point;
   precompute `d^T C0^{-1} d` and `y0 = Z^T d` per sim. Per-point cost is the
   same as today; gains are the setup and plausibly accuracy near the peak
   (today's `quadratic_form_from_prepared` is the subtractive `term1 − term2`).
   Needs a wide-grid conditioning test (e.g. r from 0 to 10× fiducial).

## Release notes

- 1.3.0 shipped 2026-10-18. `master`'s Unreleased holds only the #102 fixes, so
  1.3.1 can be cut from `master` without this branch.
- Keep the anchor changelog entry under the next minor (1.4.0) Unreleased
  section when merging `master` back in.

## Context gathered in the session

- **ECLIPSE** (Bilbao-Ahedo et al. 2021, JCAP 07 034;
  github.com/CosmoTool/ECLIPSE): Fortran + ScaLAPACK, everything
  block-cyclic, inverts the full pixel C and forms `Y† C^{-1} Y`. Reference
  case: TQU nside 64, 29 009 pixels per field (n = 87 027), ℓmax 192
  (L = 37 245 per field), 63 min on 96 Haswell cores, peak 252.8 GiB across all
  processes. Its diagonal-noise assumption only enters the noise bias
  (paper §3.3), not the Fisher reduction: ADR-0004's wording ("symbolic trace
  manipulation … assumes diagonal noise") overstates it.
- On that case the direct anchored projection costs ~1.2e15 MACs vs ~4.5e15
  real MACs for ECLIPSE and ~4.0e15 for today's harmonic path; single-process
  peak n² + mn + m² ≈ 222 GiB (≈129 GiB if A is built blockwise during the
  trace).
- **When each path wins**: fiducial anchor for Fisher/Spectra and iterations
  converging to the fiducial; noise anchor for wide PICSLike grids; pixel
  basis when modes ≫ pixels (fsky ≲ 5% at lmax = 2·nside, n_params = 18), and
  per PICSLike point at small fsky (~40× at fsky 10%).
- `_auto_pick_method` sizes pixel-direct by `bins.nbins`, not
  `nbins × nspectra` (`core.py`), undercounting it ~3× for QU.
- **ScaLAPACK** remains the route past one node (all four direct steps map to
  `pdpotrf/pdtrsm/pdsyrk/pdgemm`); ~4–6 weeks for the harmonic Fisher/Spectra
  path. Not needed if step 1 lands the ~49 GiB peak.

## Commands

```bash
uv run pytest src/cosmoforge.cosmocore/tests/test_harmonic_anchor.py \
              src/cosmoforge.qube/tests/test_harmonic_anchor_pipeline.py -q
uv run python src/cosmoforge.qube/benchmarks/validate_anchor_memory.py 3000 38
(cd src/cosmoforge.qube && uv run pytest tests/ -q -m "not slow")
```
