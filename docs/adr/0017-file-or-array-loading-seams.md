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

2. **Pure parsers stay pure (hard); dispatch stays out of them
   (hard); dispatch-location is a default (soft).** The two rules
   §Decision.2 originally bundled are distinct and only one is load-bearing:
   - **Hard.** The pure file-parsers (`readcl`, `read_covmat`,
     `read_covmat_reduced`, `read_mask`, `read_maps`, the beam readers)
     never dispatch and never accept arrays. They stay ignorant of the
     injection concept — that is what keeps `in_out`/`beam` importable and
     testable without the framework.
   - **Soft.** The dispatch decision (injected wins; else read the path)
     lives at the orchestration layer *by default*, but may reuse an
     existing collaborator's dispatch when one already converges both
     adapters and validates — e.g. `FieldCollection.set_cls` already owns
     the `None`→`readcl` branch, so `cls_data` reuses it rather than
     minting a parallel dispatch in Core. A stateful collaborator is not a
     parser; reusing its dispatch does not violate the hard rule.

   Resolution order is always: injected object wins; else the params path
   is read.

3. **Single convergence + validation point per seam is the invariant;
   the `_resolve_<input>()` method is a recommended shape, not a
   requirement.** What every seam must guarantee:
   - **(a) Contract identity** — the injected object is exactly what the
     reader returns (§Decision.1).
   - **(b) One convergence + validation point** — both adapters meet at a
     single place and pass through the same semantic check; readers only
     parse (format errors surface naturally from the parser).
   - **(c)** the hard rule of §Decision.2.

   These can be satisfied by **either** shape:
   - a dedicated private `_resolve_<input>()` method on the class that
     *owns the seam* (not necessarily `Core`: `mask`/`noise_cov` on
     `Core`, `maps` on `Spectra`), mirroring `_resolve_basis_config`; or
   - **reuse of an existing low-level collaborator** that already
     converges both adapters and validates — `cls_data` and
     `fiducial_cls` converge in `SpectraManager.set_cls`, which validates
     labels/column-count and (transitively, for `fiducial_cls`, via
     `_build_fixed_spectra` → `set_cls`) length. No dedicated resolver is
     added where one already exists.

   Trade-off of the reuse shape: the validation error surfaces from the
   collaborator (`"Missing power spectrum for TT"`) rather than a crisp
   injection-site message — accepted, because duplicating a check
   `set_cls` already performs would be speculative (the A4 grill:
   "accept what `set_cls` accepts today, do NOT expand"). Note the file
   path had no semantic validation before this ADR — the file adapter
   gains it via the shared check.

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
   | `maps1`, `maps2` | `inputmapfile1/2` | `maps1/2` on the maps-reading classes (`Spectra`, `PICSLike`) | A3, A5 |
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

### Amendment (2026-07-08, slices A3–A4): mechanism vs invariant

Slices A3 (`maps`) and A4 (`cls_data`/`fiducial_cls`) did not follow
§Decision.3's literal "one `Core._resolve_<input>()` per seam" wording.
Reviewed in a grill-with-docs session; the conclusion was that §Decision.3
had conflated an *invariant* with a *mechanism*, not that A3/A4 were debt.
§Decision.2 and §Decision.3 above are rewritten to state the invariant
((a) contract identity, (b) single convergence + validation point, (c)
pure parsers stay pure) and demote the named method to one of two valid
shapes. All four shipped seams are instances of the same rule:

| Seam | Shape | Owner / convergence point | Validation |
|---|---|---|---|
| `mask` (A1) | dedicated resolver | `Core._resolve_mask` | shape vs nside/nfields, at resolver |
| `noise_cov1/2` (A2) | dedicated resolver | `Core._resolve_noise_cov` | shape vs n_active, at resolver |
| `maps1/2` (A3, A5) | dedicated resolver | `Core._resolve_maps`, called by `Spectra`/`PICSLike` (both read observed maps) | shape vs ntot/nsims, at resolver |
| `cls_data` (A4) | reuse existing | `FieldCollection.set_cls` → `SpectraManager.set_cls` | labels/column-count, at collaborator |
| `fiducial_cls` (A4) | reuse existing (inlined dispatch) | `set_cls` after `_build_fixed_spectra` | labels/length, transitive via `set_cls` |
| `beam` (A5) | reuse existing (dispatch added) | `BeamManager.compute_beams` (injected wins over `smoothtype`) | ≥3 rows, at `compute_beams` |

Owner is "the class(es) that read the input," not always `Core`: `maps`
is read by both `Spectra` and `PICSLike`, so A5 hoisted `_resolve_maps`
onto `Core` as a shared helper (the `maps1/2` kwargs stay on those two
subclasses, off `Core`/`Fisher`). The `beam` seam already had a
convergence point — `compute_beams`, where every `smoothtype` lands as a
`(3, lmax+1)` beam dict — so A5 added an injected-wins branch there rather
than a parallel resolver, with `hp.read_cl` staying a pure parser.
