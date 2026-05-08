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
