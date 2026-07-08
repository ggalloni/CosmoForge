# ADR-0017: File-or-array loading seams

## Status

Accepted (2026-07-08; branch `feat/array-inputs`, Phase A slice A1).
Sibling of ADR-0015/0016 — completes the in-memory pipeline on the
input side. Slices A2–A6 apply this convention without re-deciding it.

## Context

ADR-0015 made persistence opt-in and ADR-0016 moved the Fisher→Spectra
handoff in-memory, but every pipeline *input* (mask, noise covariance,
maps, C_ℓ, fiducial C_ℓ, beam) is still file-bound: `InputParams` names a
path, a reader in `in_out.py`/`beam.py` loads it. Notebook and
pipeline-integration users (broom) hold these objects in memory already;
forcing a round-trip through disk is the last obstacle to a fully
in-memory run.

The interface philosophy follows the litebird_sim two-layer split
(low-level computational functions callable without the framework;
a high-level config-driven class wrapping them):

- **High-level interface** — the orchestration classes (`Core`, `Fisher`,
  `Spectra`, `PICSLike`): config-driven via `InputParams`, own MPI,
  caching, and input resolution.
- **Low-level interface** — the library modules (`in_out`, `beam`,
  `fields`, `basis`, `basics`): explicit-argument functions, importable
  and runnable without the framework. File readers are **parsers**
  (path → in-memory object), not validators.

Full decoupling of the QML computational core (stage functions with
explicit array signatures) was considered and rejected here: the shared
state in `Fisher`/`Spectra` (covariances, basis manager, MPI shared-memory
windows, caches) is load-bearing at production sizes; per-seam extractions
(batched SMW quadratic form, binned-derivative ownership, public
`evaluate_point`) remain Phase R candidates with their own benchmarks.

`InputParams` must stay a printable/serialisable paths-and-scalars dict:
arrays are never smuggled into the config (option "params fields accept
`str | ndarray`" rejected upfront).

## Decision

1. **Two adapters per input seam.** Every pipeline input is loadable
   through exactly two adapters converging on one in-memory contract:
   the *file adapter* (path in `InputParams`, parsed by the existing
   reader) and the *injection adapter* (in-memory object handed to the
   high-level class). The injected object is defined as "exactly what the
   reader would have returned" — no new flexibility in accepted forms.

2. **Dispatch lives at the high level.** Only the orchestration class
   knows both adapters exist. Resolution order: injected object wins;
   else the params path is read; readers never dispatch and never accept
   arrays.

3. **Validation split.** Readers only parse (format errors surface
   naturally from the parser). Semantic validation of the in-memory
   object (shape/dtype/nside/ordering consistency against the config)
   happens once at the orchestration layer, on the converged contract —
   both adapters pass through it. Concretely: one private
   `Core._resolve_<input>()` method per seam owns dispatch *and*
   validation (mirroring `_resolve_basis_config`); the `setup_*` methods
   call it. Note the file path had no semantic validation before this
   ADR — the file adapter gains it via the shared check.

4. **Seam mechanics: constructor kwargs, named explicitly everywhere.**
   Injected objects enter as constructor kwargs, named in the signatures
   of `Fisher`/`Spectra`/`PICSLike` *and* `Core` (subclasses forward via
   `super().__init__`), matching how `basis=` landed (ADR-0018) and
   keeping the kwargs IDE-discoverable for the notebook users the feature
   targets. `Core.__init__` stores each injected object on a private
   attribute; the corresponding `setup_*` method checks it before falling
   back to the params path. Subclass docstrings stay one line per kwarg
   with the full description owned by `Core.__init__` (drift containment).
   The kwarg set is closed by this ADR's vocabulary — it does not grow
   beyond the named input seams.

5. **Kwarg vocabulary.** Injection kwargs are named after the in-memory
   object they become (the existing attribute or kwarg downstream), not
   after the path param they shadow:

   | Kwarg | Shadows (params) | Matches (internal) | Slice |
   |---|---|---|---|
   | `mask` | `maskfile` | `mask` in `setup_fields`/`create_field` | A1 |
   | `noise_cov1`, `noise_cov2` | `covmatfile1/2` | `Core.noise_cov1/2` (pre-inversion) | A2 |
   | `maps1`, `maps2` | `inputmapfile1/2` | `Spectra.maps1/2` (Spectra-only kwargs) | A3 |
   | `cls_data` | `inputclfile` | `FieldCollection.set_cls(cls_data=...)` | A4 |
   | `fiducial_cls` | `fiducialfile` | (symmetric with `cls_data`) | A4 |
   | `beam` | `beam_file` (`smoothing_type="file"`) | the b_ℓ window array `hp.read_cl` returns | A5 |

   This set is closed; later slices use it verbatim.

6. **Mask array contract (A1).** Shape `(npix,)` or `(npix, nfields)`
   (1D promoted to a column, as `setup_fields` always did); any
   float64-coercible dtype; float values interpreted binarily downstream
   (`active = mask > 0.5` — apodized values are thresholded, documented
   not enforced); `npix` must equal `12·nside²` for `params.nside` and
   the column count must equal `params.nfields` (`ValueError` otherwise);
   pixel-indexed per `params.ordering`. The file adapter is fixed in A1
   to honour the same ordering contract: `read_mask` gains an explicit
   `nest=` argument driven by `params.ordering` (previously
   `hp.read_map`'s default force-converted to RING, silently delivering
   wrong sky positions for `ordering="NESTED"` — a latent bug this
   contract surfaces and closes).

## Consequences

- A fully in-memory run (arrays in, nothing on disk, ADR-0015 gating
  writes) becomes possible; the broom smoke test (A6) is the acceptance.
- The file-based HPC batch workflow is unchanged: paths in YAML keep
  working identically.
- Each later A-slice is mechanical: add the injection kwarg, route the
  reader result and the injected object through the same semantic
  validation, test equivalence file-vs-array.
