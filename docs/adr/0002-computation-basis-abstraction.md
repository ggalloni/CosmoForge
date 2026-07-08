# ADR-0002: Computation basis abstraction; rename from "compression"

## Status

Accepted — implemented and merged to master 2026-04-19. The constructor kwarg
rename this ADR left undone (`compression=` → `basis=`) landed in ADR-0018
(2026-07-07); its no-shim precedent (Consequences below) is retired there for
post-1.0 public changes.

## Context

The original code modelled the harmonic transform `V` as a
"compression": user-facing flag `compression="harmonic"`, directory
`cosmocore/compression/`, manager class `compression_manager`. This
naming was wrong on two counts:

1. The harmonic transform from pixels to a/b harmonic modes is a
   **basis change**, not a compression. It is exact when n_modes ≥
   n_pix, and it can in fact *expand* the problem at small fsky
   (n_pix < n_modes) — the opposite of compression.
2. Real compression (lossy approximations like m-block, eigenmode
   truncation) is a property *within* a basis, not the same thing as
   the basis itself. Conflating the two made it impossible to talk
   about "harmonic basis with no compression" or "pixel basis with
   eigenmode truncation".

## Conceptual model

Basis and compression are two orthogonal axes:

|  | **No compression** | **With compression** |
|---|---|---|
| **Pixel basis** | Pixel direct (identity, dim = n_pix) | Eigenmode-truncated (dim < n_pix) |
| **Harmonic basis** | Harmonic (V, no m-block) | Harmonic + m-block |

- **Basis** (rows): the pixel-to-basis projector. `identity` for
  pixel-direct, `V` for harmonic, `U^T` for compressed pixel (the
  transpose of the kept eigenvectors; the `projector` property returns
  this form on both subclasses).
- **Compression** (columns): optional *lossy* reduction within the
  chosen basis. Eigenmode truncation on pixel; m-block on harmonic.
  Pixel-direct and uncompressed harmonic do not compress.

Each basis instance occupies one of the four cells. Methods on the
basis return values in *the basis's current operating space* — a
polymorphic notion that varies per instance but is well-defined for
any given instance.

## Decision

Restructure the abstraction:

- `ComputationBasis` (ABC) with concrete subclasses `HarmonicBasis`
  and `PixelBasis`. Each basis is responsible for the basis change
  itself (V operator construction, Λ block layout).
- Each basis has *optional* compression: m-block (harmonic) or
  eigenmode truncation (pixel). Compression flags live on the basis,
  not on the manager.
- Directory rename `compression/` → `basis/`. Module rename
  `compression_manager` → `basis_manager`. Method rename
  `setup_compression(...)` → `setup_computation_basis(...)`.
- New API:
  `setup_computation_basis(basis="auto"|"harmonic"|"pixel",
   compress=False, delta_m=0, threshold=None)`.

The V operator is built m-ordered (modes grouped by `|m|`, not by ℓ).
This is a prerequisite for m-block compression and is harmless for
the uncompressed path.

### Method naming convention

`ComputationBasis` methods follow the polymorphism convention
*bare = basis-native; `_full_` = pixel-space*:

- **Bare method names** are polymorphic across `ComputationBasis`
  subclasses. They return / refer to the basis-native form — the value
  in *whatever space the instance currently operates in* (n_pix for
  pixel-direct, n_modes for harmonic, the truncated dim for compressed
  instances).
- **`_full_` qualifier** explicitly demands a full n_pix-dim
  pixel-space object, regardless of basis. It marks an explicit break
  from the basis abstraction. Callers use it when they need the full
  pixel-space object for downstream non-basis code (e.g. the full
  Gaussian likelihood covariance).

Examples (post-2026-05 vocabulary cleanup):
- `bm.dim` — dimension of the basis's operating space (polymorphic).
- `bm.to_basis(data)` — inject pixel-space data into the basis.
- `bm.get_inverse(C_ell)` — basis-space inverse, dim × dim.
- `bm.get_full_inverse(C_ell)` — full pixel-space inverse, n_pix × n_pix.
- `bm.get_logdet(C_ell)` — basis-space logdet.
- `bm.get_full_logdet(C_ell)` — full pixel-space logdet.

### ABC contract: symmetric `bare` / `_full_` pairs

When a `_full_` variant exists for a quantity, the bare variant exists
too, and **both are declared on the `ComputationBasis` ABC**.
Subclasses must implement both. This prevents partial APIs (one
implementation provides only the bare variant, another provides only
the full variant) and keeps callers from branching on `bm.method` to
pick the right method name. Any method called polymorphically through
`basis_manager` is similarly declared on the ABC — including the
`prepare_for_basis` / `quadratic_form_from_prepared` fast-path pair
and the `quadratic_form` slow-path.

The `_full_` variant on pixel basis is exact only in pixel-direct
mode. On a *truncated compressed* pixel basis it returns the
quantity's restriction to the kept subspace, lifted back to ``n_pix``
dimensions — a different operator than the full pixel-space one, not
an approximation. The ABC docstrings on `get_full_inverse` and
`get_full_logdet` spell this out per-method.

### Basis-specific form, same QML slot

Some methods return values whose *form* is basis-specific but whose
*role* in downstream math is shared. The canonical case is
`get_noise_for_bias()`:

- `HarmonicBasis.get_noise_for_bias()` returns
  `V N_eff^{-1} N N_eff^{-1} V^T` (the SMW intermediate ``T``).
- `PixelBasis.get_noise_for_bias()` returns `U^T N_raw U` (raw noise
  projected once into the eigenmode basis).

Both fill the same slot in the QML noise-bias sandwich
`Cov(w | noise) = X · get_noise_for_bias() · X^T`, where `X` is the
basis's natural inverse-equivalent (`(I + Λ M)^{-T}` for harmonic;
basis-space `C^{-1}` for pixel). The two return values are *not*
interchangeable — they have different shapes of pre-multiplication
baked in. The ABC docstring documents this contract; callers compose
with the basis's own inverse-equivalent and never assume the two
forms are the same matrix.

## Consequences

- **Public API break**: `compression="harmonic"` and the old
  `setup_compression` signature are gone. Users on older snapshots
  must rename. Acceptable because the project is pre-1.0 and there
  was no published external API surface yet.
- **Conceptual clarity**: "basis" and "compression" are now two
  orthogonal axes. New compressions can be added per-basis without
  growing the top-level flag space.
- **Enables follow-on work**: pixel basis direct mode (see ADR-0003),
  m-block compression (single-field spin-0), field block-diagonal K
  detection (auto-detected, no flag).
- **Documentation**: any reference to "harmonic compression" in
  comments, docstrings, or paper drafts must be updated to "harmonic
  basis". Compression now refers only to *lossy* approximations
  within a basis.
- **Naming-convention enforcement is at the ABC, not at call sites.**
  Any new method on `ComputationBasis` chooses bare or `_full_` (or
  both, as a symmetric pair) at declaration time. Callers in `qube/`
  and `picslike/` consume polymorphically — they ask the basis for the
  value in whatever form they need and never branch on `bm.method` to
  construct the call.
- **The 2×2 model is the load-bearing mental model.** Future
  architectural changes (m-block compression on harmonic, pixel-direct
  as a default configuration, new compression algorithms) should be
  expressible as a move within the 2×2, not as a new outermost mode
  flag. Backlog candidate #8 (`PixelBasis` class-name and
  configuration model) explicitly works on making the implementation
  match this model.

## Update (2026-05-14)

Introducing `cosmocore/basis/harmonic.py` and `cosmocore/basis/pixel.py`
created a name collision with the pre-existing top-level
`cosmocore/harmonic.py` and `cosmocore/pixel.py`, neither of which was
about a basis. PR #30 resolved the collision with a hard pre-1.0 rename
(no shim), asymmetric in shape because the two top-level files mixed
concerns differently:

- `cosmocore/harmonic.py` → split into `cosmocore/beam.py` (`BeamManager`,
  `coswinbeam`) and `cosmocore/spectra_io.py` (`SpectraManager`,
  `cl_to_vec`, `vec_to_cl`). The split reflects that the original file
  mixed two unrelated concerns under the "harmonic" label.
- `cosmocore/pixel.py` → flat-renamed to `cosmocore/signal_kernels.py`
  (Legendre signal-matrix kernels; content unchanged). A single concern,
  so no split.

`_spin_pair_mode_to_kind` moved from the old `harmonic.py` to
`spectrum_key.py` (where its return type lives). The umbrella
`from cosmocore import ...` API is unchanged. The rename is consistent
with the no-shim precedent established by the original ADR-0002 break
and continued by the `SpectrumKey` cut (ADR-0013) and the `PixelBasis`
cleanup (ADR-0003 update).
