# ADR-0018: Auto basis is the true default; `compression=` → `basis=`; post-1.0 deprecation policy

## Status

Accepted — Phase D of the in-memory pipeline (branch `feat/default-auto-basis`).
Realises the already-decided default of [ADR-0003](0003-pixel-basis-direct-mode.md)
at the orchestration layer and completes the vocabulary migration begun in
[ADR-0002](0002-computation-basis-abstraction.md). Establishes the post-1.0
deprecation policy that supersedes the "no-shim" stance of ADR-0002 and
[ADR-0013](0013-spectrum-key-public-identity.md).

## Context

Three long-standing inconsistencies met here.

1. **The default was not `auto`.** ADR-0003's Decision states *"`method="auto"`
   is the default for `Core`"* and its Consequences promise that scripts which
   do not specify a method "now get the optimal path automatically". That was
   true only of `Core.setup_computation_basis()` in isolation. The orchestration
   classes (`Fisher`, `Spectra`, `PICSLike`) all stored `compression: dict | None
   = None` and **skipped basis setup entirely when it was `None`** — so the
   actual default run never reached the ADR-0003 selector. It fell into Core's
   no-basis pixel-space branches (the traditional path). The documented default
   and the real default disagreed.

2. **The kwarg still carried the retired vocabulary.** ADR-0002 replaced the
   "compression" framing with "computation basis" and renamed the directory,
   manager, and method — but the public constructor kwarg `compression=` was
   left behind. Basis and compression are orthogonal axes (ADR-0002's 2×2
   model); naming the basis selector `compression` conflated them.

3. **Docstring drift.** `core.py` documented `method` as *default "harmonic"*
   while the signature said `"auto"`.

Additionally, the project is now public (v1.0.1 on PyPI). ADR-0002 and ADR-0013
each performed a hard cut with no compatibility shim, explicitly justified by
"no published external API yet". That justification has expired.

## Decision

**1. `basis=None` (the default) selects `method="auto"`.** `Fisher`, `Spectra`,
and `PICSLike` route the default run through `setup_computation_basis(method="auto")`,
so the ADR-0003 cost selector picks harmonic vs pixel-direct. The internal
`_basis_config` sentinel is preserved (`None` → traditional path), so every
`run()` and MPI branch is byte-identical; only the constructor's mapping from
the public kwarg to `_basis_config` changes.

**2. The public kwarg is `basis=`, with this value surface:**

| `basis=` | resolves to |
|---|---|
| `None` *(default)* | `method="auto"` |
| `False` | traditional no-basis pixel-space path (explicit opt-out) |
| `"auto"` / `"harmonic"` / `"pixel"` | `{"method": …}` — the basis with **no compression** |
| `dict` | explicit config; the only way to request compression (`epsilon`, `mode_fraction`, `compress`/`delta_m`) |

String sugar and the default never inject `epsilon`/`mode_fraction`, so they
always yield the uncompressed form of the basis (pixel-*direct* for `"pixel"`).
This keeps the basis axis and the compression axis orthogonal (ADR-0002).
Unknown strings raise `ValueError` naming the valid set and pointing at
`basis=False`; `None` (→auto) and `"none"` (invalid) are deliberately not
synonyms.

**3a. `do_cross=True` keeps the default on the traditional path.** The basis
layer has no second noise covariance — `create_computation_basis` takes only
`N=noise_cov1`, so every basis path silently uses C₁⁻¹ on both sides of the
cross trace (`fisher.py:534,592`). Rather than newly route cross runs onto a
known-wrong path, `basis=None` resolves to the traditional path (correct N₁ and
N₂) whenever `do_cross=True`; `auto` only selects a basis for auto-spectra. This
is behaviour-preserving (cross ran traditional pre-Phase-D) and touches none of
the N₂ machinery. Wiring N₂ through the basis layer — or hard-erroring on
`do_cross`+explicit-basis — is a Phase R decision (see
`.claude/plans/2026-07-07-spin2-traditional-vs-pixeldirect-divergence.md`,
`feedback_cross_qml_n12_not_implemented`).

**3b. The traditional no-basis path survives only behind `basis=False`.** It is
retained as the equivalence reference and because `test_fortran_validation`
depends on its pixel-space attributes (`noise_cov1`, `inv_cov1`,
`fisher_instance.signal_matrix`), which the basis paths do not expose. Its fate
— fold into a `PixelDirectBasis`-shaped adapter or delete — is a Phase R
decision, gated on the spin-2 divergence investigation
(`.claude/plans/2026-07-07-spin2-traditional-vs-pixeldirect-divergence.md`) and
a full-coverage equivalence test.

**4. Post-1.0 deprecation policy.** Breaking public-API changes now ship a
one-release deprecation shim, not a hard cut. `compression=<value>` is still
accepted on all three constructors, emits `DeprecationWarning`, and forwards to
`basis`. Passing both `basis=` and `compression=` is a `TypeError`. This
supersedes the no-shim precedent of ADR-0002 and ADR-0013.

## Consequences

- **Default path changes for every default-constructed run.** For the existing
  (full-sky) test configs the selector picks harmonic (SMW); results agree with
  the traditional path as an operator (~1e-8 eigenvalue agreement; ~6e-11
  absolute for spin-0). No test blows up in time/memory on a laptop
  (n_modes ≤ 2170).
- **Equivalence references must be pinned.** Tests that used `Fisher(config)` /
  `Spectra(config)` as the traditional reference (`_cached_traditional`,
  `test_fortran_validation`) now pin `basis=False`, else they silently compare
  auto-vs-auto. Diagnosed per-test, not bulk-edited.
- **`basis=False` (traditional) and pixel-direct agree to 1 ulp for spin-2.**
  Bit-identical for spin-0; for spin-2 they differ by a single ulp from a
  beam-weight reassociation in the batched binned-derivative kernel (S and C⁻¹
  are bit-identical). So either is a valid spin-2 reference — the opt-out is
  kept for the pixel-space *attribute* surface (below), not because the numbers
  differ meaningfully.
- **B2's disk adapter (ADR-0016) is reachable only via `basis=False`.** The
  two-job file handoff lives on the traditional path; retiring that path (Phase
  R) would also retire the adapter.
- **Full doc-drift sweep** across docstrings, per-package READMEs, examples, and
  `main_*.py` (all carried `compression={"method": …}` snippets).

## References

- [ADR-0002](0002-computation-basis-abstraction.md) — basis/compression 2×2 model; the kwarg rename completes it.
- [ADR-0003](0003-pixel-basis-direct-mode.md) — the cost-based selector and the `auto` default this ADR realises.
- [ADR-0013](0013-spectrum-key-public-identity.md), [ADR-0016](0016-fisher-spectra-handoff-transport.md) — no-shim precedent (retired) and the disk adapter interaction.
- `project_post_1_0_deprecation_policy` (memory) — the standing policy.
