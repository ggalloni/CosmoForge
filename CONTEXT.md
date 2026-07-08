# CONTEXT.md — CosmoForge

Domain glossary for the CosmoForge QML / likelihood toolkit. Use these terms verbatim
in issue titles, plans, hypotheses, and test names. Do not paraphrase.

## Subpackages

- **cosmocore** — Core library: computation basis, pixel/harmonic transforms, basics, fields, core algebra.
- **qube** — QML estimator. Fisher matrix construction (`fisher.py`) and power-spectrum estimation (`spectra.py`).
- **picslike** — Pixel-space Gaussian likelihood over a parameter grid.

## Orchestration classes

- **Core** — Abstract base shared by `Fisher`, `Spectra`, `PICSLike`. Owns field initialisation, pixel geometry, noise covariance loading, beam/spectra config, and `setup_computation_basis()`.
- **Fisher** — Builds and inverts the QML Fisher matrix; trace evaluation is per-bin. Optionally caches binned derivatives for re-use by `Spectra`.
- **Spectra** — Evaluates `q_b` per simulation, applies the chosen normalisation (`mode=…`), subtracts noise bias for auto-spectra, supports cross-correlation via independent C₁, C₂ filtering.
- **PICSLike** — Evaluates the pixel-space Gaussian log-likelihood `−½(dᵀ C⁻¹ d + ln|C|)` over a parameter grid; uses the SMW fast path on harmonic basis and the direct path on pixel basis.
- **FieldCollection** — Groups one or more `BaseField` instances (`ScalarField`, `PolarizationField`), auto-derives the relevant auto/cross spectra, and carries per-component multipole floors `lmin_signal`.
- **High-level interface** — The orchestration classes above: config-driven (`InputParams`), own MPI, caching, and the resolution of every pipeline input (from a file path in the config or an injected in-memory object; ADR-0017).
- **Low-level interface** — The library modules (`in_out`, `beam`, `fields`, `basis`, `basics`): importable functions taking explicit values, independently runnable without the framework. File readers are parsers (path → array), not validators. The `InputParams`-free boundary is being realised incrementally (ADR-0017).

## QML estimator

- **QML** — Quadratic Maximum Likelihood estimator for CMB power spectra. Optimal under Gaussian assumptions.
- **Fisher matrix (F)** — Information matrix, used both as covariance of the deconvolved estimator and as the bandpower window matrix W in convolved mode.
- **Window matrix (W)** — Equal to F. Used to convolve theory predictions when `mode="convolved"`.
- **Output modes** — Three normalisation conventions of `Spectra.get_power_spectra(mode=...)`:
  - `"deconvolved"`: F⁻¹ q (standard C_ℓ estimates, covariance F⁻¹).
  - `"decorrelated"`: F⁻¹ᐟ² q (uncorrelated bandpowers, covariance I).
  - `"convolved"`: q + W (raw, must convolve theory).
- **q vector** — Raw quadratic estimator output before normalisation.

## Computation basis

- **ComputationBasis** — Abstract base for the basis in which signal/noise are represented. Concrete: `HarmonicBasis`, `PixelBasis`.
- **basis_manager** — Manages basis state and transforms.
- **HarmonicBasis** — Modes ordered by |m|, not ℓ. SMW kernel path; cost scales as O(n_modes³).
- **PixelBasis** — Pixel-space algebra. Two internal modes: **direct** (no V; signal matrix and derivatives built via Legendre kernels, selected when no `epsilon` / `mode_fraction` is passed) and **compressed** (V with eigenmode truncation). Direct-mode cost ≈ `(n_bins+1)·n_pix³` per setup.
- **`method="auto"`** — The default computation-basis selection, both for `setup_computation_basis()` and for the `Fisher`/`Spectra`/`PICSLike` constructors when `basis` is unset (`basis=None`; ADR-0018). Cost-based selector (ADR-0003): compares `n_modes³` (harmonic) against `(n_bins+1)·n_pix³` (pixel-direct) and picks the cheaper. `n_pix = n_modes` at fsky ≈ 0.35. Auto only selects a basis for auto-spectra; with `do_cross=True` the default stays on the traditional no-basis path (the basis layer has no second noise covariance). Explicit opt-out to that path is `basis=False`.
- **V operator** — Maps pixels → modes. Spin-0 uses normalised `legendre_plm`; spin-2 uses `scale_ell = sqrt((2ℓ+1)/(4π))`.
  Spin-2 layout: rows `[E modes | B modes]`, cols `[Q pixels | U pixels]`.
- **Λ (Lambda)** — Block-diagonal signal covariance in the harmonic basis. Spin-0 diagonal; spin-2 has 2×2 blocks (EE, BB, EB) at each (ℓ, m). For cross-component spin-2×spin-2 pairs in DIRECTIONAL mode the two off-diagonal blocks carry separate GC and CG values (`_build_lambda_block_spin2` accepts `C_GC` and `C_CG` independently); in SYMMETRIC mode a single `C_EB` fills both.
- **SMW** — Sherman-Morrison-Woodbury. Used to invert (S+N) without forming the full pixel-space matrix. Stable form: `M(I + ΛM)⁻¹` (not `M − M K⁻¹ M`).
- **m-block compression** — `compress=True, delta_m=0` treats K as block-diagonal in m. **Exact** for azimuthally symmetric masks (Oh/Spergel/Hinshaw 1999); approximation only for generic masks where the mask induces m–m' coupling. ~lmax² speedup. Currently single-field spin-0 only.
- **Field block-diagonal K** — Auto-detected when no cross-spectra and noise is independent per field. Exact, no flag.

## Basis-native vs pixel-space methods

Naming convention introduced by ADR-0002 (vocab debt, PR #32):

- **Bare method name** (`get_inverse`, `get_logdet`, `quadratic_form`, `to_basis`) — operates in the basis's **native** space. On `HarmonicBasis` this is mode-space; on `PixelBasis` direct mode this is pixel-space.
- **`_full_` prefix** (`get_full_logdet`, etc.) — operates in **pixel-space** regardless of basis. On `PixelBasis` direct mode these coincide with the bare form; on `HarmonicBasis` they reconstruct the pixel-space quantity via SMW.

Other renamed vocabulary (use these, not the legacy names):

- **`dim`** — basis dimension (was `n_kept`).
- **`to_basis(d)`** — project a pixel-space vector into the basis (was `compress_data`).
- **`prepare_for_basis(C_ell)`** / **`BasisPrepared`** — pre-computed per-spectrum-point factor used by `quadratic_form` and `get_logdet` (was `prepare_smw` / `SMWPrepared`).
- **`quadratic_form(d1, d2, C_ell)`** — `d1ᵀ C⁻¹ d2` in the basis (was `compute_quadratic_form`).
- **`get_logdet(C_ell)`** — `ln|C|` in the basis (was `get_compressed_logdet`).

## Binning

- **Bins** — Bandpower binning class in cosmocore. `Bins.fromdeltal(lmin, lmax, delta_ell)`. Adapted from xQML (must cite Vanneste+ 2018).
- **delta_ell** — Bin width. `delta_ell=1` recovers per-ℓ estimation (the default).
- **Binned derivative** — `dC^b = Σ_ℓ w_{b,ℓ} b²_ℓ dC^ℓ` (beam-smoothed).

## Multipole ranges (ADR 0009)

- **Signal-cov band** `[min(lmin_signal), lmax_signal]` — what the basis represents (V, Λ, S). `lmax_signal` defaults to `4·nside`.
- **Inference window** `[lmin, lmax]` — where C_ℓ vary; outside this band but inside the signal-cov band, the fiducial spectrum is used and the contribution is precomputed into `S_fixed`.
- **Per-component low-ℓ floor** `lmin_signal[i]` — must satisfy `lmin_signal[i] >= |spins[i]|`. Enables direct dipole estimation (`lmin_signal=[1, 2]` for T+QU) and foreground/template handling (`lmin_signal=0`).
- **Constraint chain** — `max(lmin_signal) <= lmin <= lmax <= lmax_signal` enforced at params load.

## Spin and polarisation

- **Spin-0 / spin-2** — Spin-0 covers temperature T; spin-2 covers polarisation Q/U → E/B.
- **Slot** — within a spin-`s` component, an index `0..n_slots(s)-1`. Spin-0 has one slot named **S** (CMB alias: T). Spin-2 has two slots, **G** (parity-even, CMB alias: E) and **C** (parity-odd, CMB alias: B). Current scope: spin-0 and spin-2 only; spin-1 etc. would extend the slot vocabulary.
- **SpectrumKind** — directional ordered slot pair `(slot_i, slot_j)`. Nine values: `SS, GG, CC, GC, CG, SG, GS, SC, CS`. CMB aliases (`TT, EE, BB, EB, BE, TE, ET, TB, BT`) live in `cosmocore.conventions.cmb`, alongside the `to_cmb_canonical(result_dict, *, spins)` helper that re-keys an output dict to T-first ordering regardless of declaration order.
- **SpectrumKey** — `SpectrumKey(comp_i, comp_j, kind, spins=...)`. Passive identifier whose constructor validates kind-vs-spins consistency. Used as both list element (in `spectra_list`) and dict key (in `C_ell_dict`, output spectra dicts). Replaces the pre-Slice-5 `(comp_i, comp_j, mode)` 3-tuple.
- **Canonical direction (symmetric mode)** — declaration order picks the component direction: canonical pair is `(i, j)` with `i ≤ j`. For cross-component spin-2 × spin-2 where `GC` (EB) and `CG` (BE) are physically distinct, alphabetical slot ordering breaks the tie (canonical = `GC`).
- **SymmetryMode** — `SYMMETRIC` (default; collapses `GC` and `CG` to their symmetrised form for cross-component spin-2 pairs, reproducing pre-Slice-5 behaviour) or `DIRECTIONAL` (emits both `GC` and `CG` as separate spectra and uses independent covariance entries). Lives on `Fisher` / `Spectra` (Spectra inherits from its Fisher — ADR-0011), never on `SpectrumKey`.
- **Sign convention** — Spin-0×spin-2 Λ and derivatives carry a minus sign because E = −(₂Y + ₋₂Y)/2.

## Likelihood (PICSLike)

- **PICSLike** — Full Gaussian pixel-space log-likelihood: `ln L = −½ [d^T C⁻¹ d + ln|C|]` evaluated over a parameter grid.
- **ParameterGrid** — Cartesian product of parameter ranges; loads spectra from files; supports fiducial blending and MPI distribution.
- **Fiducial blending** — Outside the inference window `[lmin, lmax]` the covariance uses the fiducial spectrum (not the test point's). Stabilises the inversion at high ℓ and lets foreground/template components live in `S_fixed`.
- **LikelihoodResult** — Stores χ², log-L, best-fit, marginalised likelihoods, confidence intervals; serialisable.

## Inputs and conventions

- **Physical C_ℓ** — Standard CAMB/CLASS values. The `(2ℓ+1)/(4π)` normalisation is absorbed into the Legendre basis functions; `apply_normalization()` is a no-op.
- **Beam (b²_ℓ)** — Beam smoothing absorbed into derivatives. Fisher and q are beam-smoothed.
- **`smoothing_type`** — Cosine-window convention used by `coswinbeam`. Two named variants: `cosine_legacy` (ℓ₁=N_side, Aghanim+2019 / Benabed+2009 — the Planck Legacy default; current `InputParams` default) and `cosine_npipe` (ℓ₁=1, Akrami+2020 — recommended to suppress 857-GHz ringing). Bare `cosine` is a deprecated alias for `cosine_legacy` (emits `DeprecationWarning`); the deprecation alias exists because user YAML configs live outside version control we own.
- **Pixel window (pixwin)** — HEALPix pixel window function; must be applied consistently between sims and theory in MC tests.
- **Buffer** — High-ℓ buffer beyond the inference window, included so edge bins are unbiased.
- **Opt-in persistence** (ADR-0015) — No computed quantity (covariances, geometry, Fisher matrix, estimator covariance, error bars) is written to disk unless the caller sets the corresponding `out*` path; an unset path means "do not persist". C_ℓ estimates are additionally writable via the explicit `Spectra.write_power_spectra`.
- **Input injection** (ADR-0017) — Every pipeline input is loadable through two adapters: a file path in `InputParams` or an in-memory object passed as a constructor kwarg (the injected object wins). Fixed kwarg vocabulary: `mask`, `noise_cov1`/`noise_cov2`, `maps1`/`maps2` (Spectra), `cls_data`, `fiducial_cls`, `beam` — each named after the in-memory object it becomes, not the path it shadows.
