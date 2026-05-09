# ADR 0009 — Multipole-range API

## Status

Accepted (2026-05-07).

## Context

Two distinct multipole windows govern any QML / pixel-likelihood computation:

- The **signal-cov band** is the range of multipoles whose contributions
  are represented in the basis (V operator, Lambda, signal matrix).
  Outside this band the signal contribution is zero.
- The **inference window** is the subset of the signal-cov band whose
  ``C_ell`` actually vary with the parameters being inferred. Outside
  this window — but inside the signal-cov band — the fiducial spectrum
  is used and the contribution is precomputed into ``S_fixed``.

Until this ADR the code used ``lswitch_low``/``lswitch_high`` for the
inference window and an implicit ``[2, 4*nside]`` for the signal-cov
band. The naming confused new contributors, hard-coded the spin-2 floor
of 2 onto the temperature monopole/dipole, and prevented per-component
low-ell floors needed for foreground templates and direct dipole
estimation.

## Decision

Adopt four explicit names with a constraint chain:

| Name           | Scope                                | Constraint                          |
| -------------- | ------------------------------------ | ----------------------------------- |
| ``lmin_signal``| per-component, ``int \| list[int]``  | ``lmin_signal[i] >= \|spins[i]\|``  |
| ``lmax_signal``| scalar                               | defaults to ``4*nside``             |
| ``lmin``       | scalar                               | ``lmin >= max(lmin_signal)``        |
| ``lmax``       | scalar                               | ``lmax <= lmax_signal``             |

``max(lmin_signal) <= lmin <= lmax <= lmax_signal``.

### Cl-array convention

All ``cl`` arrays are ℓ-indexed (length ``lmax_signal + 1``, ``cl[ell] = C_ℓ``).
This was the groundwork delivered in PR1 (``cl-ell-indexed``).

### Per-component spin floor

``lmin_signal`` is per-component. A scalar ``lmin_signal=N`` is broadcast
to ``[N]*n_components`` after the field count is known. Each component's
value must be at least ``|spin|`` (representation-theory bound). Validation
lives in ``Fisher.__init__`` / ``PICSLike.__init__``, with a defensive
assert in ``setup_computation_basis()``.

### ``S_fixed`` accumulates both sides of the window

The single ``S_fixed`` matrix sums contributions from
``[lmin_signal_for_pair, lmin)`` and ``(lmax, lmax_signal]`` in one pass —
both sides are linear in ``C_ell``, so a single ``compute_signal_matrix``
call over a ``cl_fixed`` dict suffices.

### Hard rename of ``lswitch_low`` / ``lswitch_high``

Replaced everywhere — no deprecation shim.

## Consequences

- Direct dipole estimation becomes possible (``lmin_signal=[1, 2]`` for
  spin-0 + spin-2; ``lmin=lmax=1`` for monopole/dipole-only fits).
- Per-component low-ell floors enable foreground template marginalization
  via ``lmin_signal=0`` for designated components.
- The per-spectrum inference window (per-component ``lmin``/``lmax``) and
  per-component binning are deliberately deferred. They compose cleanly
  with this design but are not in scope here.
- Existing analyses are unchanged: defaults match the old behaviour
  (``lmin_signal=2``, ``lmin=2``, ``lmax_signal=4*nside``,
  ``lmax=lmax_signal``).

## Noise-bias convention with ``S_fixed``

The QML model in QUBE writes the data covariance as
``C(Λ) = N_eff + V Λ V^T`` where ``N_eff = N + S_fixed``. The
Sherman-Morrison-Woodbury inversion uses ``N_eff`` so that ``V`` only
spans the inference window — purely a performance trick on the
inversion, *not* a redefinition of what counts as "noise".

The QML noise bias subtracted from ``q_b`` is therefore the **Tegmark
form**, anchored to the actual instrumental noise N:

```
bias_b = ½ Tr[E_b · V C⁻¹ N C⁻¹ V^T]
```

`<q_b − bias_b>` then evaluates to:

```
F · Λ_truth   +   ½ Tr[E_b · V C⁻¹ S_fixed C⁻¹ V^T]
                       ↑
                  S_fixed residual
```

The second term is the contribution of the frozen ``S_fixed`` band to
``Ĉ_b`` via mode coupling induced by the mask: out-of-window signal
mask-couples into the inference window, exactly as it does for any
pseudo-Cl analysis. It is **not** a bug — the data still contain
``S_fixed``, the model still describes it, and the QML output reports
``Ĉ_b`` *convolved* with the window matrix that encodes both
in-window mode coupling and out-of-window leakage.

For unbiased comparison against a theory model, convolve the model
through the bandpower window function returned by
``Fisher.get_bandpower_window_function()``:

```
<Ĉ_b>  =  Σ_ℓ  W_{b,ℓ}  C_ℓ_truth
```

The W matrix has support both *inside* the inference window
(in-window mode coupling) and *outside* it (the ``S_fixed`` leakage),
and reduces to the identity in the full-sky limit.

### Why not subtract the residual at the estimator level?

A consistent variant ("B1") replaces N by ``N_eff`` in the bias
formula, giving ``<Ĉ_b> = Λ_truth`` directly. We do **not** adopt
this:

- It departs from the Tegmark literature convention.
- It would have to be applied across all three QML paths together —
  ``harmonic.py:_compute_smw_components``,
  ``pixel.py:get_compressed_noise``, and
  ``spectra.py:_compute_qml_spectra_traditional`` — otherwise the
  cross-implementation tests
  (``test_compressed_spectra_T``,
  ``test_pixel_no_compression_matches_harmonic``) fail because the
  paths disagree.
- The bandpower window function machinery already exists for theory
  comparison, so the practical benefit is small.

The convention is locked by
``test_harmonic_basis.py::test_nondiagonal_n_with_switch_optimization``,
which compares the production ``A·_noise_cov_T·A^T`` against a direct
evaluation of ``V C⁻¹ N_orig C⁻¹ V^T`` (with ``N_orig`` passed
explicitly) and expects them to agree to machine precision. Removing
the algebraic ``S_fixed`` correction in ``_compute_smw_components``
breaks that test deliberately.
