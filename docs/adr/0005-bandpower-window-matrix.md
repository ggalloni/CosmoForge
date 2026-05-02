# ADR-0005: Bandpower window matrix for parameter inference

## Status

Accepted — implemented in `qube/fisher.py` and `qube/spectra.py`.

## Context

Once a QML run produces binned bandpowers `d_b`, users want to do
parameter inference: find θ such that the theory C_ℓ(θ) is
consistent with `d_b`. The naive recipe — compare each bandpower to
the simple arithmetic average of theory C_ℓ over the bin — looks
right and is wrong.

When C_ℓ varies steeply within a wide bin (low ℓ, large `delta_ell`),
the QML deconvolved bandpower does **not** estimate the simple bin
average. It estimates a *Fisher-weighted* average of the underlying
per-ℓ spectrum:

```
<C_hat_b>  =  ( F_b⁻¹ · Q · F_perell · C_ℓ_true )_b
```

where `Q` is the bin-summing matrix and `F_perell` the per-ℓ Fisher.
Empirically: with `delta_ell=10`, comparing QML to simple bin
averages gave `0.44×` at ℓ=4 (looks like 56% bias). Comparing
against the correct Fisher-weighted theory gave `1.02×` at ℓ=4 (no
bias). Per-ℓ QML appears unbiased even against simple averages
because per-ℓ Fisher windows are essentially delta functions —
hiding the issue at fine binning.

This is a likelihood-construction decision: a wrong theory mapping
silently shifts inferred parameters, with no surface bug for tests
to catch.

## Decision

Inference on QML binned bandpowers must use the bandpower **window
matrix** to map per-ℓ theory into expected bandpowers, not simple
bin averaging.

The likelihood form is

```
−2 ln L  =  (d − W · C(θ))ᵀ  F_b  (d − W · C(θ))
```

with

- `W = F_b⁻¹ · Q · F_perell` — the bin-to-ℓ bandpower window
  function;
- `F_b` the binned Fisher (used as the inverse covariance of `d`);
- `C(θ)` the **unbeamed** per-ℓ theory spectrum (physical CAMB/CLASS
  values). Beam² is already absorbed into `W` via the derivatives.

API:

- `Fisher.get_bandpower_window_function()` — public method, lazily
  triggers a per-ℓ Fisher run if needed and caches the result.
- `Spectra.convolve_theory_for_inference(cl_theory)` — convenience
  that applies `W` to a per-ℓ theory vector and returns expected
  bandpowers.

When a high-ℓ buffer is used to absorb mode-coupling leakage from
`ℓ > lmax_science`, the buffer bins must be dropped *consistently*
from `d_b`, `W`, and `F_b` before forming the likelihood. Dropping
from one but not the others biases the result. Buffer-width
selection (the operational question of *how far* beyond
`lmax_science` to estimate) is captured separately in
`.claude/plans/2026-04-30-noise-threshold-buffer-selector.md`.

## Consequences

- **Correctness**: parameter inference on binned QML output is
  unbiased under the correct theory mapping. The 56% apparent bias
  at low ℓ disappears.
- **API contract**: users must not feed beamed theory into
  `convolve_theory_for_inference` — beam² lives in W. Users must
  not bypass `W` and bin-average theory directly.
- **Documentation requirement**: the QML inference recipe must
  appear in user-facing docs and the paper. Without it, users will
  reproduce the simple-averaging mistake and report spurious biases.
- **Multi-spectrum scope**: current implementation is scaffolded for
  single-spectrum (T-only or single E/B). Multi-spectrum (TT/EE/BB/TE)
  generalises naturally — `Q` block-diagonal in spectrum index,
  `F_perell` full, `W` of shape `(n_spec·n_bins, n_spec·n_ell)` —
  and is a follow-on once polarisation spectra are wired through
  binning.
- **Cost**: lazy. `W` requires a per-ℓ Fisher run; cached after
  first call.
