# ADR 0010 — Per-spectrum multipole windows

## Status

Proposed (deferred). Captures the design agreed during the PR3 brainstorming
session on 2026-05-09; implementation deferred until after the cosmoforge
v0.1 / Paper I release.

## Context

ADR 0009 introduced a two-layer multipole API:

- **Layer A (signal-cov band)**: per-component `lmin_signal[i]` and a scalar
  `lmax_signal`.
- **Layer B (inference window)**: scalar `lmin` and `lmax`, with the
  constraint `max(lmin_signal) ≤ lmin ≤ lmax ≤ lmax_signal`.

Real-world CMB analyses routinely cut spectra at *different* multipole
ranges. The Planck baseline likelihoods use ℓ=2–2508 for TT and ℓ=2–1996
for TE/EE. HiLLiPoP cuts certain frequency-pair cross-spectra at
different ℓ from autos to limit dust contamination. The Planck 70 GHz
QML analysis estimates BB/TB/EB on 46 % sky at ℓ=2–29. ADR 0009 explicitly
listed per-spectrum windows as deferred future work.

Three forces motivate the extension:

1. Beam-driven SNR varies per spectrum (TT informative to ℓ~1500, EE to
   ℓ~700, BB to ℓ~300 for a typical satellite). Fixing all spectra at the
   same `[lmin, lmax]` either wastes information (truncating TT) or
   inflates noise (extending BB into the noise-dominated regime).
2. Cross-spectra carry different information than auto-spectra. With
   full-sky T and partial-sky polarisation, TE can be informative at ℓ
   where EE alone is rank-deficient — TE pairs each (mask-scrambled) E
   mode with a clean full-sky T mode, breaking the EE degeneracies.
3. Foreground/template handling sometimes wants `lmin_signal=0` for one
   component (a monopole template) while others stay at `lmin_signal=2`.
   PR2 already supports this; ADR 0010 lets the *inference window* be
   per-spectrum to match.

## Decision

Layer A stays as defined in ADR 0009. Layer B becomes per-spectrum,
keyed by the canonical 3-tuple `(comp_i, comp_j, mode)`.

### Constraint chain

For each spectrum `(i, j, mode)`:

```
max(lmin_signal[i], lmin_signal[j])  ≤  lmin_{ij,mode}
                                     ≤  lmax_{ij,mode}
                                     ≤  lmax_signal
```

Layer A gives the *structural* envelope (V must contain the modes).
Layer B per-spectrum is otherwise free. **No** auto-cross relationship
is enforced — `lmin_TE` may be below `lmin_EE`, `lmax_TE` may be above
`lmax_EE`, and so on.

### API surface

`lmin` and `lmax` accept polymorphic input:

```yaml
# Scalar — broadcast to all spectra (back-compat, common case):
lmin: 2

# Dict by physical label — for canonical CMB analyses:
lmin:
  TT: 2
  TE: 5
  EE: 20
  BB: 2

# Dict by explicit 3-tuple (string-keyed in YAML, tuple-keyed in Python)
# — escape hatch for multi-frequency or non-canonical labels:
lmin:
  "(0,0,0)": 2     # TT
  "(0,1,0)": 5     # TE
  "(1,1,0)": 20    # EE
```

Internally everything resolves to `dict[tuple[int, int, int], int]`.
Validation rejects unknown labels with an enumeration of the legal set
for the current `physical_labels`/`spins` configuration.

### Output layout

Bandpowers, Fisher matrix, error bars, and the bandpower window matrix
stay as flat numpy arrays in canonical spectrum order. A new accessor
`bandpower_slices: dict[tuple[int,int,int], slice]` maps each spectrum
key to its slot in the flat array. The uniform case is byte-identical
to today; the slice map collapses to equal-width slices.

### Out of scope (defer further)

- **Per-component `lmax_signal`.** Stays scalar. Multi-resolution
  analysis (different `nside` per field) is its own ADR (provisional
  ADR 0011).
- **Per-spectrum `delta_ell` / per-spectrum `Bins`.** Bandpower binning
  remains uniform across spectra.
- **Cauchy–Schwarz enforcement** on the fiducial. The framework does
  not check `(C_ℓ^TE)² ≤ C_ℓ^TT · C_ℓ^EE` at frozen ℓ; if the user
  fiducial violates it, `S_fixed` becomes non-PSD and downstream
  Cholesky fails. Documented as a user responsibility.

## Consequences

- Every loop that today reads `params.lmin..params.lmax` becomes a
  per-spectrum iteration. The largest hits land in `Fisher.run` (sparse
  weight indexing, bandpower window matrix), `Spectra` (bandpower
  layout, theory convolution), `Core` (S_fixed assembly, already
  per-pair-floor as of PR3), and `Bins` (per-spectrum bin edges or a
  shared bins object with per-spectrum masks).
- Migration is opt-in: existing scalar configs broadcast and stay
  byte-identical. Existing test fixtures (`test_pipeline_regression`,
  `test_dipole_mc`) keep passing without change.
- The `picslike` blending logic in `_blend_with_fiducial` learns to
  consult per-spectrum windows when deciding what to overwrite at the
  test point.
- The `Fisher.get_bandpower_window_function` Q matrix grows in row
  count to `Σ_spec n_bins_spec`; columns stay at `lmax_signal + 1`.

## Implementation note

PR3 (commit on branch `pr3-s-fixed-perpair-floor`) landed the per-pair
low-band floor in `core._build_fixed_spectra` — a small fix that closed
ADR 0009's known foot-gun for heterogeneous `lmin_signal`. That helper
is the natural integration point for per-spectrum lmin/lmax: replace
`lmin_b`/`lmax_b` with per-key dicts and the rest of the pipeline
follows.

## References

- ADR 0009 — Multipole-range API.
- Planck 2018 V (arXiv:1907.12875) — different multipole ranges per
  spectrum in the baseline likelihood.
- HiLLiPoP / LoLLiPoP (PR4 cosmological parameters paper) —
  per-frequency-pair cuts.
- Brainstorming session 2026-05-09 — Q1 (granularity), Q2 (Layer A
  scope), Q3 (API surface), Q4 (output layout) — full transcript in
  the conversation that produced PR3.
