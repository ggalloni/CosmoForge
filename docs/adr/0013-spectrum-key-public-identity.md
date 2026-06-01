# ADR-0013: `SpectrumKey` is the single public identity for spectra

## Status

Accepted — landed in PR #25 (2026-05-12). Companion ADR-0011 covers the `SymmetryMode` sub-decision for spin-2 × spin-2 cross-component pairs. Companion ADR-0009 covers the multipole-range type that is keyed by `SpectrumKey`.

## Context

Before this decision, every public method that referred to a single spectrum carried three positional integers: `comp_i: int`, `comp_j: int`, `mode: int`. The mode encoding was implicit — `0` for SS, then a `(comp_i, comp_j)`-dependent decoding for the spin-2 GG/CC/GC/CG slots, with an additional spin-0×spin-2 SG/SC convention layered on top. The same int triple appeared on:

- `ComputationBasis.get_derivative_matrix`, `get_binned_derivative_matrix`, `get_binned_derivative_direct`,
- `HarmonicBasis._get_derivative_diagonal`,
- `Fisher.compute` outputs (`Fisher.fisher[comp_i, comp_j, mode, ell]`),
- `Spectra` multi-spectrum bookkeeping (`spectra_list = [(i, j, mode), ...]`),
- Test fixtures across all three packages.

Three problems forced a cleanup:

1. **Encoding ambiguity.** `mode=1` means something different for a spin-0×spin-0 pair (invalid) vs a spin-2×spin-2 pair (CC) vs a spin-0×spin-2 pair (SC). Every consumer re-implemented the decoder, every test had to know the rule, and the rule was undocumented outside source comments. Bugs (the spin-0×spin-2 sign convention, the spin-2 diagonal-path `is_diagonal` check, the `dim` for spin-2 derivatives — see `feedback_spin2_fisher_gotchas.md`) traced back to consumers each carrying their own decode.
2. **No spin validation at the boundary.** Constructing `(comp_i=0, comp_j=1, mode=3)` where component 1 was spin-0 was accepted silently and produced garbage downstream. There was nothing to validate against because `mode` was an opaque int.
3. **CMB conventions had no canonical seat.** External users want `TT/EE/BB/EB/TE/TB`. The library spoke in `(comp_i, comp_j, mode)`. Translation lived in user scripts, which made the multi-field generalisation of CMB conventions (`TQU × TQU` joint analyses with arbitrary component ordering) impossible to express without an ordering convention shared between the library and its callers.

Alternatives considered:

- **Keep the int triple, add a separate validator.** Doesn't address the encoding-ambiguity problem at the type level; a separate validator is something every consumer has to remember to call.
- **`NamedTuple` instead of frozen dataclass.** Rejected because `SpectrumKey` carries derived metadata (`spins`) that is validated against `kind`; a `NamedTuple` would expose `spins` as a positional field rather than a kwarg-only, validated attribute.
- **Push `SpectrumKey` all the way into the Numba kernels.** Rejected — Numba cannot hash a frozen dataclass cheaply, and `spectrum_idx` is the natural array index for the algebra inside the inner loop.

## Decision

`SpectrumKey` is the single public identity for spectra across `cosmocore.basis`, `qube.Fisher`, `qube.Spectra`, and `picslike`. There is no parallel int-mode API.

Concretely:

1. **Type.** `SpectrumKey(comp_i, comp_j, kind, spins=...)` is a frozen dataclass in `cosmocore.spectrum_key`. The constructor validates that `kind ∈ SpectrumKind` is consistent with the `(spins[comp_i], spins[comp_j])` pair (spin-0×spin-0 → only `SS`; spin-2×spin-2 → only `GG/CC/GC/CG`; spin-0×spin-2 → only `SG/SC/GS/CS`). The dataclass is hashable and usable as a dict key.

2. **Kind enumeration is directional.** `SpectrumKind` carries nine ordered slot pairs: `SS / GG / CC / GC / CG / SG / SC / GS / CS`. The directionality of GC vs CG (and SG vs GS, SC vs CS) is preserved in the type; collapsing GC and CG to a symmetrised pair is the job of `SymmetryMode` (ADR-0011), not of `SpectrumKey`.

3. **Single keyed surface, hard rename.** The pre-PR int-mode methods (`get_derivative_matrix(ell, comp_i, comp_j, mode, ...)`, `get_binned_derivative_matrix(...)`, `get_binned_derivative_direct(...)`) are **deleted**, not deprecated. The transitional `_keyed`-suffixed methods introduced during the Slice 5 migration drop the suffix and become the canonical names. `HarmonicBasis._get_derivative_diagonal_keyed` is promoted to public `get_derivative_diagonal`. No shim. Pre-1.0 hard rename, consistent with ADR-0002 vocabulary debt and the top-level `harmonic.py`/`pixel.py` split (PR #30).

4. **Numerical kernel boundary keeps int-mode signatures.** `_build_derivative_matrix_with_spins`, `_build_derivative_matrix(ell, spectrum_idx)`, and the Numba `compute_00/02/22_contribution` kernels stay int-keyed. `spectrum_idx` is the natural array index inside the algebra; `SpectrumKey` lives at the orchestration boundary one level up. Translation between the two happens in `Core._resolve_spectrum_idx`, which lazily builds a `_spec_idx_by_key` reverse-lookup cache on first use, invalidates on length mismatch, and raises on duplicate-key collisions.

5. **CMB conventions are a view, not a parallel key.** `cosmocore.conventions.cmb` exposes the standard aliases (`TT = SpectrumKind.SS`, `EE = SpectrumKind.GG`, `BB = SpectrumKind.CC`, `EB`, `BE`, `TE`, `ET`, `TB`, `BT`) and a `to_cmb_canonical(result_dict, *, spins)` re-keying helper that rewrites a `SpectrumKey`-keyed result dict into a CMB-aliased one. The CMB aliases are values, not types — there is one canonical `SpectrumKey` per spectrum, and `to_cmb_canonical` is a presentation layer over it.

6. **Defensive boundaries.** Single-field fast paths in both `HarmonicBasis` and `PixelBasis` reject non-`SS` keys explicitly. `Fisher.__init__` coerces `symmetry_mode` strings to the enum so YAML-driven configs don't silently misroute. `_resolve_spectrum_idx` raises (not silent slab-0) when `nspectra > 1` and `spectra_list` is unpopulated.

## Consequences

- **The encoding ambiguity is gone.** A future reader sees `SpectrumKey(0, 1, kind=SpectrumKind.GC, spins=(2, 2))` and reads it directly — no triple-decoder in their head.
- **Spin validation at construction.** Bugs that lived in consumer decoders (the spin-0×spin-2 sign, the `is_diagonal` spin-2 case, the `dim` mismatch) cannot recur at the type boundary; they are caught when `SpectrumKey(...)` is built.
- **CMB callers can speak CMB.** A downstream likelihood receiving the output of `Spectra.compute_qml_spectra` calls `to_cmb_canonical(result, spins=fields.spins)` and gets `{"TT": ..., "EE": ..., "BB": ..., "TE": ...}`. The library itself never branches on whether the caller wants CMB aliases.
- **Numba kernels are unchanged.** The hot loop still sees int indices; the performance characterisation in ADR-0004 (sparse-COO traces) is preserved. The keyed surface adds one cache lookup per derivative-matrix request, amortised against the matrix construction cost.
- **No tuple-input shim, no `_keyed` suffix.** A reader skimming the API does not encounter a "legacy" surface they have to ignore. The price is that any external code written against the pre-PR-25 API breaks on upgrade — accepted because the library is pre-1.0 and the legacy surface had no documented external consumers.
- **`_spec_idx_by_key` is an implementation detail of `Core`.** Future refactors that change how `spectra_list` is populated can rebuild the cache however they want, as long as they preserve the public `_resolve_spectrum_idx(key) → int` contract.
- **CONTEXT.md is the source of truth for the vocabulary.** `SpectrumKey`, `SpectrumKind`, `Slot`, and `SymmetryMode` are defined there; this ADR records the architectural commitment, not the term definitions.

## Validation

- White-box: 13 int-mode test sites migrated to `SpectrumKey` in PR #25. Two bridge-equivalence tests deleted as scaffolding for the dual-API era.
- Cluster sanity: representative QU run at fsky=0.10, nside ∈ {16, 32, 64} confirmed no perf regression (peak RSS −0.43%, Fisher +0.78%, Spectra +2.69% on the load-bearing QU_nside64 cell — all inside run-to-run noise).

## References

- PR #25 — implementation.
- ADR-0011 — `SymmetryMode` (the GC/CG directionality sub-decision that rides on top of `SpectrumKind`).
- ADR-0009 — multipole-range API (keyed by `SpectrumKey`).
- ADR-0002 — basis abstraction (the upstream rename that established the no-shim pre-1.0 precedent).
- `cosmocore/spectrum_key.py`, `cosmocore/conventions/cmb.py`, `cosmocore/core.py` (`_resolve_spectrum_idx`, `_spec_idx_by_key`).
- `feedback_spin2_fisher_gotchas.md` — three bugs that motivated spin validation at construction.
- `.claude/plans/2026-05-11-keyed-api-single-cut.md` — design memo.
