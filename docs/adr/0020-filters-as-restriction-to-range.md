# ADR-0020: Filters are restriction to the operator's range, unified with compression

## Status

Accepted (2026-09-09). Implemented on `feat/native-filters`, held for the 1.3.0
tag. Extends [ADR-0002](0002-computation-basis-abstraction.md) (a filter is one
more subspace the analysis runs in) and [ADR-0017](0017-file-or-array-loading-seams.md)
(three kwargs join the injection vocabulary; see the amendment there).

## Context

Real data reaches a map-space estimator already filtered. A scan-synchronous
polynomial has been fitted and removed, a band of multipoles has been
deprojected, a template stack has been regressed out, hits weighting has been
applied. The estimator has to know, because a filtered map is not a
realisation of the model covariance the unfiltered estimator assumes: its
power along the removed directions is gone, and treating that absence as
noise-free signal produces a singular covariance, or worse, a finite but wrong
one.

The standard workaround, and the one an external prototype of this feature
used, is to conjugate every covariance-shaped object by the filter,
`C -> F C Fᵀ`, and then regularise the result back to invertibility with a
ridge on the null space. That works, and it is what the acceptance tests still
use as an oracle, but it is expensive (every object stays `n x n` forever, and
the ridge is a free parameter nobody wants to publish) and it reaches the
estimator through private-method overrides rather than a supported input.

The question this ADR settles is what a filter *is* to CosmoForge, not how to
bolt one on.

## Decision

**A filter is a subspace. The analysis is carried out inside it.**

### 1. Restriction to `range(F)`, recorded as the thin SVD

Optimal statistics depend on a linear `F` only through its row space above the
rank threshold: conjugated QML and the Gaussian likelihood are invariant under
invertible maps of the filtered data (measured at 1e-13 and 2e-16). So the
library keeps `F = U Σ Wᵀ` truncated at rank `r` and works in `r` dimensions.
The ridge is deleted from the library and survives only as a test oracle.

Internal coordinates are the **range side**. Raw inputs map by `Uᵀ F = Σ Wᵀ`,
pre-filtered inputs by `Uᵀ`, and matrices by `Σ (Wᵀ X W) Σ` or `Uᵀ X' U`. `Σ`
is an invertible diagonal, so the estimator is unchanged by its presence and
nothing is ever divided by a small singular value. For a projector `U` *is*
`W`, `Σ = I`, and all of this collapses to plain subspace restriction.

The rank threshold is applied to the singular values of `F`, or of the template
stack for a constructed projector, **never to `C`**. It is a user-exposed
bias/variance knob, not an implementation detail.

### 2. Reduce-to-range is compression

A filter and S/N compression are the same primitive, `Qᵀ C Q`, differing only
in where `Q` comes from: geometric for a filter, statistical for compression.
They compose in a fixed order, deproject first and compress inside the
survivors, and the compression eigenproblem runs in the already-restricted
problem so that downstream code sees one subspace, not two.

### 3. One seam per object, always the end object

The filter is applied where the wrappers around the two kernel entry points
end, once per object, never per multipole:

| Object | Seam |
|---|---|
| `N` | `Core.setup_covariance_matrices`, after `_resolve_noise_cov` |
| `d` | `Spectra.setup_maps` and `PICSLike.setup_maps`, after `_resolve_maps` |
| `S` | the free function `cosmocore.signal_matrix(fields, lmax, pixel_filter=)` |
| `E_b` | the return of `Core.get_binned_derivative_matrix` |
| `V` | `ComputationBasis._restrict_V`, `V' = V W Σ` |

On V-based paths (harmonic, compressed pixel) `S` and the per-multipole
derivatives never exist in pixel space, so `V` is the only seam and the
restricted problem is strictly smaller than the unfiltered one. On kernel paths
(traditional, pixel-direct) the end object for the derivative is the **binned**
`E_b`, which costs `n² r` per bin rather than per multipole.

### 4. Pre-filtered inputs are declared, in two independent flags

`maps_prefiltered` and `noise_prefiltered` say that an input already carries
the filter, so it is restricted with `Uᵀ` rather than `Σ Wᵀ`. They are
independent because a pipeline whose noise simulations went through the filter
alongside raw signal maps is a legitimate combination that one merged flag
would forbid.

`noise_prefiltered` means "the covariance of the noise in the maps as handed",
never "the filter whitened my noise". For a projector the two conventions agree
exactly; for a graded filter they differ by `Σ²` and are different noise
models.

### 5. Field-block structure is off wholesale under a filter

Component-mixing filters are allowed, and the per-field bookkeeping is switched
off whenever a filter is present. Block-diagonality is never sniffed from a
global `W`: for a projector every singular value is 1, so LAPACK may rotate
degenerate SVD columns across blocks and a field-diagonal `F` can produce a `W`
with support in both. Block structure has to be *declared* at construction, and
1.3.0 has no way to declare it.

**The basis layer's `n_components` is a field count**, not a component count in
the `CONTEXT.md` sense: `n_components = len(self._theta_tuple)` counts pointing
sets, and `_n_pix_per_component` doubles for spin-2. A filter's guards are
therefore field-granular, and Q-U mixing never leaves its own block. Only T-P
mixing crosses one. The misnomer is documented here rather than renamed: the
rename is breaking on a public basis attribute, costs an ADR-0018 shim, and
buys nothing inside a filter release.

Guards raise `NotImplementedError` uniformly. It subclasses `RuntimeError`, so
it sails through the `except ValueError` in `_apply_compression` that would
otherwise swallow a guard and draw an empty diagnostic panel.

### 6. Scope

All four paths (traditional `basis=False`, pixel-direct, compressed pixel,
harmonic) and all three orchestration classes. The operator constructors ship
in `cosmocore.filters`, not in `qube`, because the scope is CosmoForge.
Filtered cross-spectra come for free on the traditional path, since
`noise_cov2` passes the same seam.

## Considered alternatives

**Ridge-regularised conjugation.** Rejected as the library mechanism, kept as
the test oracle. It leaves every object at `n x n`, and the ridge is a free
parameter that has to be chosen, disclosed and defended. Restriction reproduces
it to 1.4e-14 (projector) and 2.6e-12 (graded), so nothing is lost by dropping
it. One trap is recorded with it: **the oracle must conjugate the truncated
operator** `F_trunc = U_k Σ_k W_kᵀ`. An oracle built on the full `F` is a
different estimator and disagrees at `range_epsilon²`.

**Thresholding `C` instead of `F`.** Rejected. The rank of the analysis is a
property of the operator the data went through, not of the fiducial covariance
the analyst happens to be testing, and a threshold on `C` moves with the
parameter grid.

**Detecting field-block structure from `W`.** Rejected on the degenerate
singular values above. Silently wrong rather than merely imprecise.

**Subclassing the orchestration classes** and overriding the private methods
that build `S` and the derivatives, which is what the external prototype did.
Rejected: it works on the traditional path only and is silently bypassed when
`basis=None` resolves to pixel-direct, which is the default at the fsky where
filtered analyses actually live. A supported input cannot have a default that
turns it off.

**Overloading one threshold argument** so that a value in `(0, 1)` means a
fraction and an integer means a count. Rejected because it collides with
`PixelBasis.mode_fraction`, where a value in `(0, 1)` means the opposite thing.

## Consequences

- **Conditioning tracks the threshold.** Internal coordinates are `Σ Wᵀ d`, so
  the conjugated covariance carries `Σ²` and its condition number goes as
  `range_epsilon⁻²` (measured 4.35e3 / 1.24e6 / 2.90e8 / 1.51e11 at 1e-2 /
  1e-3 / 1e-4 / 1e-5). Hence `MIN_RANGE_EPSILON = sqrt(eps)`, about 1.5e-8, as
  a clipping floor with a loud warning rather than a machine-precision default.
  Agreement with any dense reference floors at `cond · eps_machine`, so a flat
  tolerance is the wrong acceptance criterion.
- **There is no default threshold.** `Filter` raises without one. The right cut
  depends on the operator's spectrum, and a wrong default is invisible.
- **Refused under a filter:** m-block compression, per-field `epsilon` /
  `mode_fraction` lists, per-field eigenspectrum diagnostics, and the dormant
  `out*` file handoff (it writes `r x r` and the read side reshapes to `n x n`;
  ADR-0016 makes the in-memory alias primary, so this is documentation, not
  code).
- **`Spectra(fisher=…)` adopts the Fisher's filter** and raises on a
  conflicting one, matching the 1.2.0 rule for `lmax_signal`. A filtered
  `Spectra` that normalised by an unfiltered Fisher would be wrong in a way no
  shape check catches.
- **Transport is `_shared_array`, never `bcast`** (ADR-0014). Pickling `W`
  costs roughly 22 GB on a 32-rank node. The record is rebuilt around the
  shared buffers by a private non-validating constructor, and a projector's `U`
  must come back as the *same object* as its `W`.
- **The filter is fingerprinted against the mask**, `blake2b` of
  `active_pixel_index(mask)`, checked once in `Core.setup_covariance_matrices`.
  Python's builtin `hash()` is per-process randomised and would break under
  MPI.
- **Kernel paths win nothing in memory.** The kernels fill `n x n` before
  conjugation; the saving is in the `r`-dimensional solves. Traditional-path
  PICSLike pays `n² r` per grid point, so filtered grids belong on the basis
  paths where the filter is paid once.

## Validation

- Restriction against a ridge-regularised conjugation of the truncated
  operator, on the traditional path: 1.4e-14 (projector), 2.6e-12 (graded at
  `range_epsilon = 1e-3`).
- Per-multipole oracle against binned-`E_b` restriction at machine precision,
  which is what validates the seam placement in §3.
- Hits weighting, invertible by construction, is a no-op at 4e-15.
- Acceptance suite: `qube/tests/test_pixel_filter.py`,
  `picslike/tests/test_pixel_filter.py`, and the operator constructors'
  self-checks in `cosmocore/tests/test_filters.py`.
- MPI transport is written but unverified above one rank.

## References

- ADR-0002 (computation basis), ADR-0008 (linalg single source of truth),
  ADR-0014 (broadcast and shared memory), ADR-0016 (Fisher to Spectra
  handoff), ADR-0017 (loading seams, amended by this ADR), ADR-0018
  (deprecation policy).
- Oh, Spergel & Hinshaw 1999 for the azimuthal-symmetry condition that m-block
  compression rests on, and which a filter breaks.
