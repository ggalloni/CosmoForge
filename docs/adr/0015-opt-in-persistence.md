# ADR-0015: Opt-in persistence

## Status

Accepted (2026-07-06; merge target `feat/opt-in-writes`).

## Context

Every quantity CosmoForge computes — reduced noise covariance, inverted total
covariance, pixel geometry, the Fisher matrix, the estimator covariance, error
bars — used to be written to disk unconditionally during `run()`. The write
sites were either ungated or guarded by `hasattr(self.params, "out…")`, which
is always true because `set_defaults` populates every `out*` attribute with a
relative default (`outputs/…`). A user running `Fisher(config).run()` from any
directory therefore silently produced an `outputs/` tree, and `run()` even
crashed when `outcovmatfile` was left at its `""` default (writing to an empty
path).

This blocks the in-memory pipeline goal (CosmoForge runnable top-to-bottom
without touching disk) and surprises every caller who did not ask for files.
Two of the artifacts (`outnoisecovmat*`, `outinvcovmatfile*`) additionally
doubled as the Fisher→Spectra transport on the pixel basis; ADR-0015's sibling
work (Slice B2) moves that handoff to the live `Spectra(fisher=…)` object seam,
leaving these files as pure opt-in artifacts.

## Decision

**No quantity CosmoForge computes is persisted unless the caller provides an
output path.** Providing a path is the save trigger; a `None` (or empty) path
means "do not write".

Concretely:

1. All `out*` / `output_geometry_file` defaults in `InputParams.set_defaults`
   become `None` (previously relative `outputs/…` paths, and `""` for
   `outcovmatfile` / `outerrfile`).
2. The four file-writing primitives — `write_covmat_reduced`, `write_out_matrix`,
   `writecl`, `output_geometry` — no-op when their path is falsy (`None` or
   `""`). The gate lives in the helpers, not at each call site, so any present
   or future write site is gated uniformly and cannot silently reintroduce
   implicit I/O. The now-redundant `hasattr` wrappers at the call sites are
   removed; calls become unconditional and rely on the helper gate.
3. Persistence remains available two ways: set the corresponding `out*` path in
   the config (written during `run()`), or call the explicit imperative method
   `Spectra.write_power_spectra(mode=…, filename=…)` for the C_ℓ estimates. No
   new `save()` API is introduced.

A hard cut was chosen over a deprecation cycle: the package is v1.0.1 with a
small user base, the shipped example YAMLs (`T/QU/TQU/TEB_defaults.yaml`) all
carry explicit `out*` paths so their workflows are unchanged, and a deprecation
release would keep the implicit-write behaviour alive for months while adding
machinery to distinguish an omitted key from an explicit `None`.

## Consequences

- A bare config that omits `out*` keys now produces **no files** where it
  previously wrote an `outputs/` tree. This is the intended behaviour change.
- The two-job HPC workflow (Fisher writes covariance/inverse files in job 1;
  Spectra reads them in job 2 on the pixel basis) **requires `out*` paths to be
  set**. The shipped `*_defaults.yaml` already set them, so documented workflows
  are unbroken; users with hand-rolled configs that relied on default output
  paths must add explicit paths to restore the old behaviour. This is the
  migration note for the release.
- The `outcovmatfile=""` crash disappears (falsy path → no-op).
- Byte-for-byte outputs are unchanged when paths are set.
- `outnoisecovmat1` is still written twice (once in `Core.setup_covariance_matrices`,
  once in `Fisher.prepare_covariance_matrices`); both are now gated identically.
  Deduplicating that write is out of scope here.

## See also

- Slice B2 (in-memory Fisher→Spectra handoff) removes the file transport role of
  the reduced/inverted covariance files.
- `.claude/plans/2026-07-06-in-memory-pipeline-master-plan.md` — Phase B.
- Code: `cosmocore/settings.py` (defaults), `cosmocore/in_out.py` (helper gates),
  `cosmocore/core.py`, `qube/fisher.py`, `qube/spectra.py` (call sites).
