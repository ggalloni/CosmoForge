# ADR-0019: The bandpower shape function lives in the binning operator

## Status

Proposed — implementation tracked in issues 49, 50 and 20.

## Context

A binned QML estimator does not measure "the C_ℓ in this bin". It measures
the amplitude of an assumed **shape** within the bin. That assumption is
made by the binning operator, whether or not anyone writes it down.

CosmoForge's binned derivative is

```
dC^b = Σ_{ℓ∈b} w_ℓ · b²_ℓ · dC^ℓ
```

and today `w_ℓ = 1` (`Core.get_binned_derivative_matrix`, "the sum runs
over ℓ in the bin with unit weight"). Unit weight is not a neutral choice:
it models `C_ℓ = C_b` for ℓ in the bin. The estimator therefore returns a
flat-C_ℓ bandpower. That is a **declaration about the spectrum**, and it
was never named as one.

The declaration then went unhonoured downstream. With
`output_convention: Dl`, the pipeline estimated C_b under the flat-C model
and afterwards multiplied by a single scalar `ℓ(ℓ+1)/2π` evaluated at the
bin midpoint. Two things are wrong with that:

1. **A scalar cannot do the job.** Once a bin is collapsed to one number,
   the ℓ-dependence of `ℓ(ℓ+1)` inside the bin is gone and no scalar can
   restore it. For weights `w` normalised over the bin, convexity gives

   ```
   Σ_ℓ w_ℓ · ℓ(ℓ+1)  =  ℓ_eff(ℓ_eff+1) + Var_w(ℓ)
   ```

   so *any* single-ℓ scalar undershoots the true D-space factor by exactly
   the weighted variance of ℓ across the bin — the error grows with bin
   width and bites hardest at low ℓ, where bins are widest relative to the
   curvature of `ℓ(ℓ+1)`.

2. **The scalar used the wrong ℓ anyway.** It used the bin midpoint. The
   estimator applies inverse-variance weighting automatically — that is
   what QML *is* — so the bandpower's true ℓ-centroid is the centroid of
   the estimator's realised window, not the midpoint of the bin edges.

The two errors have different characters, and only the second one is
visible. The first silently rescales every D_ℓ output; the second
misplaces it on the ℓ axis.

## Decision

**The shape function is declared once, in the binning operator, and
everything downstream honours it. There is no post-hoc conversion.**

`Bins.shape_weights()` returns the per-ℓ weight `w_ℓ` that declares the
in-bin spectrum shape, and `output_convention` selects it:

| `output_convention` | `w_ℓ` | declares | estimator returns |
|---|---|---|---|
| `Cl` | `1` | C_ℓ flat within the bin | `C_b` |
| `Dl` | `2π / (ℓ(ℓ+1))` | D_ℓ flat within the bin | `D_b` |

The `Dl` weight follows from the declaration: if `D_ℓ = D_b` is constant
across the bin then `C_ℓ = 2π·D_b / (ℓ(ℓ+1))`, so

```
∂C/∂D_b = Σ_{ℓ∈b} [2π/(ℓ(ℓ+1))] · b²_ℓ · dC^ℓ
```

and the QML amplitude *is* `D_b`. The Fisher `F_b` that comes out is
already the Fisher for `D_b`, so the covariance is correct with no
transformation. `_dl_factor` and the `W_dl = W · outer(d, 1/d)` similarity
transform are **deleted** — the conversion is not fixed, it ceases to
exist.

Consequently `output_convention` is an **estimator** setting, not a
presentation setting. `Cl` and `Dl` mode estimate genuinely different
observables, and `D_output ≠ scalar × C_output`. This is the surprising
part of this ADR and the reason it exists.

### Effective multipole

The shape function is an *input*; the window is an *output*. Declaring
`w_ℓ` says what is assumed, not how much each true multipole actually
contributes — that is set by the noise, the mask coupling and cosmic
variance, all of which live in the per-ℓ Fisher. With
`W = F_b⁻¹ · Q · F_perell` (ADR-0005) and `Q[b,ℓ] = w_ℓ` for ℓ ∈ b, the
identity `F_b = Q · F_perell · Qᵀ` gives `W · Qᵀ = I`, i.e.
`Σ_ℓ W[b,ℓ]·w_ℓ = 1`. So the normalised window in the estimated quantity's
own space is

```
W̃[b,ℓ] = W[b,ℓ] · w_ℓ ,        Σ_ℓ W̃[b,ℓ] = 1
ℓ_eff(b) = Σ_ℓ W̃[b,ℓ] · ℓ
```

`Fisher.get_effective_ells()` returns this. `Spectra.get_effective_ells()`
delegates to it, mirroring `convolve_theory_for_inference`. `Bins` carries
no notion of an effective multipole at all: it knows bin edges, and bin
edges cannot know where a bandpower sits.

The bin midpoint survives as `Bins.lmid` — an honest name for a cheap
label, available without a Fisher run.

## Considered alternatives

**Keep the post-hoc scalar, but evaluate it at the inverse-variance
weighted ℓ.** This was the original proposal. It fixes the visible error
(the abscissa) and leaves the invisible one (the `Var_w(ℓ)` rescale)
untouched — for `delta_ell=5` at low ℓ that residual is ≈5%, larger than
the error being fixed. Rejected.

**Weight the scalar by the fiducial spectrum**, i.e.
`f_b = Σ W·ℓ(ℓ+1)·C_ℓ^fid / Σ W·C_ℓ^fid`. This is exact when
`C_ℓ ∝ C_ℓ^fid`, but it assumes a *different* in-bin shape than the
binning operator (`Q = 1`) declares — two contradictory shape functions in
one pipeline. It also makes a published data product depend on
`fiducialfile`, so two users with identical data and identical bins get
different D_ℓ. Rejected as inconsistent, not merely as inexact.

A fiducial *shape function* — declared in the binning operator, where it
would be consistent — is a coherent third option and a plausible future
value of this setting. It is out of scope here.

**Derive the weights analytically** from `w(ℓ) ∝ (2ℓ+1)/(2(C_ℓ+N_ℓ)²)`.
There is no `N_ℓ` in CosmoForge: noise enters as a pixel-space covariance
(`noise_cov1`), never as a spectrum. Constructing one requires assuming
white noise — precisely the assumption QML exists to avoid. Rejected; the
per-ℓ Fisher already carries the exact weighting, including mask coupling,
and needs no analytic stand-in.

## Consequences

- **Breaking, numerically and silently.** Any existing run with
  `output_convention: Dl` gets different D_ℓ values, a different
  covariance and a different Fisher, with no call site changed. The shift
  is not a uniform factor: it varies bin to bin, largest where bins are
  wide and ℓ is low. This must carry a `**Breaking changes:**` changelog
  entry. Reproducing prior results requires pinning the previous release.
- **Cost is confined.** No D_ℓ path touches the bandpower window, so
  `get_power_spectra`, `get_covariance`, `get_noise_bias` and
  `convolve_theory_for_inference` are unchanged in cost.
  `get_effective_ells()` is the only consumer of `W`; it triggers a per-ℓ
  Fisher (≈`delta_ell²` more trace evaluations than the binned one),
  cached after first use, and free for callers already doing inference.
  Callers who only need an axis use `Bins.lmid`, which costs nothing.
- **`get_effective_ells()` now requires a completed Fisher run** and must
  raise if called before one, reusing the guards
  `convolve_theory_for_inference` already carries.
- **MPI.** `_compute_per_ell_fisher()` is collective. Any accessor that
  can reach it must have every rank enter together; an early
  `if rank != 0: return None` in front of a collective call is a silent
  hang, not an exception.
- **ℓ = 0 is excluded from `Dl` mode.** `2π/(ℓ(ℓ+1))` diverges at the
  monopole and `D_ℓ` is not defined there, so `Dl` with `lmin_floor = 0`
  must raise. ℓ = 1 is well defined (`w = π`).
- **Per-spectrum effective ells.** TT, EE and BB have different noise and
  therefore different in-bin weighting, so a multi-spectrum run yields a
  different `ℓ_eff` per spectrum. A single midpoint could not express
  this. This is why the multi-spectrum bandpower window (lifting the
  `nspectra > 1` guard on `get_bandpower_window_function`) is a
  prerequisite rather than a follow-up.
- **`Bins` loses its P/Q operator machinery**, which has no production
  caller and encodes a *flat* bin average the estimator does not use.
  ADR-0007's rationale cites those methods and must be amended; the xQML
  attribution and the GPLv3 obligation are unaffected.

## References

- Bond, Jaffe & Knox 1998 — bandpower binning operator formalism; the
  shape function is theirs, we are only naming it.
- ADR-0005 — the bandpower window matrix `W = F_b⁻¹ Q F_perell`, on which
  the effective multipole is built.
- ADR-0007 — xQML attribution for `Bins`; amended by the removal above.
