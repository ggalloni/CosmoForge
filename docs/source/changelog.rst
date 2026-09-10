Changelog
=========

All notable changes to CosmoForge will be documented here.

Every pull request that touches package source updates this file; see
``CLAUDE.md`` and the pull-request template.

Unreleased
----------

**Breaking changes:**

* ``cosmocore.compute_pointings(nside, active, ordering)`` returns the three
  tuples ``(point_vectors, theta, phi)`` instead of filling caller-supplied
  buffers; the ``npixs``, ``point_vectors``, ``theta_vectors`` and
  ``phi_vectors`` parameters are gone. The old seven-argument call raises
  ``TypeError`` at the call site, and a shim keeping the old positional slots
  would misbind the new call, so this is a hard cut under the ADR-0018 bright
  line. The per-pixel ``healpy`` calls are replaced by one vectorised
  ``pix2ang`` per field; the pointing vectors are bit-identical to the old
  loop at ``nside <= 8`` and within one ulp above.
* ``cosmocore.active_pixels`` (and everything built on it, including
  ``Field.active_pixels`` and every mask handed to ``Fisher``) raises
  ``ValueError`` on a mask with a negative entry other than ``healpy.UNSEEN``.
  ``UNSEEN`` still counts as inactive. Previously any negative value was
  silently thresholded away.
* ``FieldCollection.set_pointing_vectors`` raises ``ValueError`` when the
  number of arrays differs from the number of fields, instead of silently
  truncating.

**Deprecated (removed in 1.4.0):**

* ``cosmocore.count_nonzero_mask``: use ``len(active_pixels(mask)[0])``. The
  two differ on masks with negative entries.
* ``cosmocore.basics.wigner_d_matrix``: use a comprehension over
  ``wigner_d_small``. It is no longer JIT-compiled.
* ``cosmocore.basics.matrix_slogdet``: use ``numpy.linalg.slogdet``.
* ``cosmocore.idx2spec``: nothing in the package calls it; ``SpectrumKey`` is
  the public identity of a spectrum.

**Added:**

* Pixel-space filters are a first-class input. ``Fisher``, ``Spectra`` and
  ``PICSLike`` accept ``pixel_filter=``, a ``cosmocore.filters.Filter``, and
  carry the analysis out by restriction to the operator's range on every path
  (traditional ``basis=False``, pixel-direct, compressed pixel, harmonic). The
  spectra, Fisher matrix and likelihood returned are then those of the filtered
  data. Build a filter with ``Filter.from_operator(F, range_epsilon=…)``,
  ``Filter.from_deprojection(T)`` for a template-annihilating projector, or
  ``Filter.from_subspace(W)``; combine two projectors with ``f1 & f2``.
  ADR-0020.
* ``Filter`` requires a rank threshold and has no default: ``from_operator``
  takes exactly one of ``range_epsilon`` (relative cut on the operator's own
  singular values) or ``range_rank`` (an exact count), and raises otherwise.
  ``range_epsilon`` is clipped to ``MIN_RANGE_EPSILON`` (``sqrt(eps)``, about
  1.5e-8) with a warning, because the restricted covariance carries ``Σ²`` and
  a smaller cut puts its condition number beyond double precision. Truncating
  outside a spectral gap warns and reports where the largest gap is; keeping
  every singular value warns that the filter is invertible and quotes the
  resulting condition number.
* ``maps_prefiltered`` and ``noise_prefiltered`` declare that an input already
  carries the filter, so it is not applied twice. They are independent, because
  pipeline noise simulations that went through the filter alongside raw signal
  maps are a legitimate combination. ``noise_prefiltered`` means "the
  covariance of the noise in the maps as handed", never "the filter whitened my
  noise": for a projector the two conventions are identical, for a graded
  filter they differ by ``Σ²`` and are different noise models.
* ``cosmocore.signal_matrix(fields, lmax, pixel_filter=None)`` allocates and
  fills the pixel-space signal covariance, optionally restricted. It replaces
  five copies of the same allocate-and-fill that sized the matrix from the
  noise buffer, and is the supported way to build ``S`` without a ``Core``.
* ``cosmocore.basics.restrict`` and ``restrict_data`` apply a restriction to a
  symmetric matrix and to pixel-space data.
* ``cosmocore.filters`` ships constructors for the common operators, so a
  caller does not assemble a template stack by hand:
  ``harmonic_deprojection(mask=, spins=, ells=, slot=)`` removes whole
  multipoles of one slot, taking its templates from the estimator's own ``V``
  rather than resynthesising them; ``scan_polynomial(mask=, pole=, n_stripes=,
  degree=)`` removes Legendre polynomials in azimuth within bands of constant
  polar angle, the pixel-space image of per-scan baseline removal; and
  ``hits_weighting(hits, mask=)`` builds inverse-noise weighting, which is
  invertible and therefore a documented no-op control rather than a filter
  anyone needs to apply. Every instrument-specific number is a required
  argument. ``harmonic_deprojection`` is not purification and is not sharp in
  multipole: its reach is set by the patch, roughly ``180 / radius`` degrees,
  not by ``ells``.
* ``cosmocore.basis.harmonic_operator`` and ``ell_mode_index`` are public, so
  the cut-sky harmonics the estimator works in can be built without a
  ``Fisher``. ``harmonic_deprojection`` is their first consumer.
* ``BudgetConfig`` and ``PixelDirectBudgetConfig`` take ``working_dim`` (the
  filter's rank), and ``qube-memory-budget`` takes ``--working-dim``. The dense
  stages then scale with the rank while the kernel buffers stay at the pointing
  count. Omitting it reproduces the previous numbers exactly. The filter terms
  are modelled, not calibrated against a measured run.

**Changed:**

* ``Spectra(fisher=…)`` adopts that Fisher's filter and raises ``ValueError``
  if the constructor asks for a different one, matching the existing rule for
  ``lmax_signal``, ``mask=`` and the covariances.
* The dormant ``out*`` covariance handoff is unsupported with a filter: it
  writes the restricted ``(rank, rank)`` matrix while the read side reshapes to
  the pointing count. In-memory handoff has been the primary path since
  ADR-0016, so this is a documented limitation and not a code change. The
  reduced covariances written by ``outnoisecovmat1/2`` stay unrestricted, so
  the on-disk products remain in pixel space.
* ``ComputationBasis`` now raises ``ValueError`` when the noise covariance
  handed in does not match the working dimension, instead of proceeding until
  a downstream matmul fails. The same check catches an unrestricted ``N`` under
  a filter.
* Under a filter the basis layer refuses the operations whose premises the
  filter breaks, all with ``NotImplementedError``: m-block compression
  (``compress=True``, which needs an azimuthally symmetric range), per-field
  ``epsilon``/``mode_fraction`` lists, and per-component eigen-spectra. Field
  blocks are treated as one coupled group, so the block-diagonal ``K``
  shortcut cannot return a wrong Fisher. ``mode_fraction`` is the recommended
  compression knob under a filter.

**Fixed:**

* ``LikelihoodResult.get_confidence_intervals`` returns the highest-density
  interval its documentation always promised: grid points are taken in order
  of decreasing likelihood until the requested mass is reached. The previous
  loop expanded symmetrically in grid index from the maximum, so on an
  asymmetric posterior it returned a wider interval shifted towards the
  steep side. Reported bounds change for asymmetric posteriors; symmetric
  ones are unaffected.
* The fixed-multipole signal matrix ``S_fixed`` and the pixel-space derivative
  are now sized from the field pixel counts rather than from the noise buffer.
  The old sizing was correct only because the two agreed; it fed truncated
  views to a bounds-unchecked kernel as soon as they did not.

Version 1.2.0 (2026-09-08)
--------------------------

The Paper I referee round. One signal-cov ceiling per run instead of up to
two, a memory model that matches what the profiler measures, and figures that
compare the two estimators on the same quantity. Most of what changed here was
reporting a number that was not the one in effect.

**Breaking changes:**

* ``PixelDirectBudgetConfig`` loses ``has_switch`` and ``qube-memory-budget``
  loses ``--no-switch``. The buffer they modelled is no longer allocated on the
  pixel-direct path, so there is no behaviour left to forward a shim to: a
  warn-and-ignore keyword would accept an argument that can no longer mean
  anything. Cut outright for the same reason ``do_derivative_step`` was
  (ADR-0018). A stale call raises ``TypeError`` and a stale ``--no-switch``
  exits with an argparse error, rather than silently modelling a buffer that
  does not exist.
* ``Spectra.lmax_signal`` now raises ``RuntimeError`` when assigned after
  construction, instead of accepting the value and ignoring it. ``Spectra``
  builds *and runs* its internal Fisher inside ``__init__`` (unlike ``Fisher``,
  which runs only when you call ``run()``), so a later assignment cannot change
  the ceiling anything was computed at. It previously moved the reported
  attribute and no result: ``s.lmax_signal = 8`` produced output bit-identical
  to setting nothing at all. The error names the two routes that do work,
  ``basis={"lmax_signal": N}`` and ``lmax_signal:`` in the config. Scripts that
  set it and silently got the ``4*nside`` answer now fail loudly and need one
  of those instead.
* ``Spectra(fisher=…)`` now adopts that Fisher's ``lmax_signal`` and raises
  ``ValueError`` if the constructor also asks for a different one. A supplied
  Fisher owns the ceiling: its Cls, beams and basis are already built and
  ``run()`` reuses them, so a conflicting request cannot be honoured. This
  matches how ``fisher=`` already refuses ``mask=``, ``noise_cov1=``,
  ``cls_data=`` and ``beam=``. Adopting it also fixes the plain case, where
  ``Spectra`` resolved from ``params.lmax_signal``/``4*nside`` independently of
  the Fisher it was handed and could report a ceiling that was not in effect.
* ``PixelDirectBudgetConfig`` is now keyword-only, and ``lmax_signal`` on it is
  optional (default ``None``). With ``has_switch`` gone, ``lmax_signal`` feeds
  no term on that path: it is echoed in the table header as provenance and read
  by nothing else. Keyword-only rather than reordered, because a reordered
  positional signature would let an old four-argument call rebind ``n_bins`` to
  a multipole with no error. ``qube-memory-budget --lmax-signal`` is
  correspondingly no longer required for ``--path pixel_direct``; it stays
  required for ``--path harmonic``, where it sizes the basis.

**Added:**

* A config that sets ``beam_file`` together with any ``smoothing_type`` other
  than ``file`` now emits one ``UserWarning`` naming both values. Only
  ``smoothing_type: file`` ever opens the file; the analytic beams
  (``none``, ``gaussian``, ``cosine_legacy``, ``cosine_npipe``) ignore it
  silently, and an inert path left in a config has twice been read back as
  provenance for the beam actually applied. The warning fires only on a
  user-supplied key, never on the built-in default, and never on an empty
  ``beam_file``. Resolved beams are unchanged. It is a ``UserWarning``, not a
  ``DeprecationWarning``, so a suite run under ``-W error`` will now fail on a
  config carrying an inert ``beam_file``; three of this repository's own test
  configs did.

* ``qube-memory-budget`` gained ``--cache-derivatives`` (and
  ``PixelDirectBudgetConfig.cache_derivatives``, default ``False`` to match
  ``Fisher``'s own default), which adds the retained binned-derivative cache
  — ``n_params × n_pix²`` — to the predicted ``fisher_run`` and ``spectra_run``
  state. Without it the tool predicted only the uncached configuration and
  under-reported a cached run by 1.6× at QU/nside=64. The term is an upper
  bound: measured 11.8 GiB against the modelled 13.2 GiB at that cell.

**Fixed:**

* ``qube-memory-budget``'s footer no longer claims a "~25–30 GiB" Python
  baseline RSS. Measured baselines are 0.26 GiB for interpreter and imports,
  rising with the run's own input maps and covariance.

* Pixel-direct runs no longer build the fixed-multipole signal matrix
  ``S_fixed``. ``Core.setup_computation_basis`` resolved ``method="auto"``
  *after* the ℓ-switching branch, so a run that ended up pixel-direct still
  paid a full signal-matrix pass and left a third ``n_pix²`` buffer resident
  in the allocator pool for the rest of the run. The path never consumed it —
  pixel-direct carries the high-ℓ signal in pixel space — so persistent
  ``basis_setup`` state drops from three ``n_pix²`` buffers to two, and one
  ``compute_signal_matrix`` pass disappears from setup. Results are unchanged
  (ADR-0003, amended to record that pixel-direct is exempt from the ``lswitch``
  ``S_fixed`` split).
* Setting ``fisher.lmax_signal`` / ``picslike.lmax_signal`` before ``run()``
  now applies to the computation basis too. The basis builder read
  ``params.lmax_signal`` directly, so a setter-set ceiling was ignored: Cls
  and beams were built at the requested ℓ while the basis was built at
  ``4*nside``, and the run failed later with a "beam too short" error from the
  spectra loader. Putting ``lmax_signal:`` in the YAML was unaffected.
* ``Spectra.lmax_signal`` now honours ``params.lmax_signal`` before falling
  back to ``4*nside``, matching ``Fisher`` and ``PICSLike``. A standalone
  ``Spectra`` (no ``fisher=``) previously ignored the config key.
* ``basis={"lmax_signal": …}`` is now an alias for the ``lmax_signal`` setter
  on **all three** of ``Fisher``, ``Spectra`` and ``PICSLike``, rather than a
  basis-only override. It reached only ``Fisher`` before. On ``Spectra`` it set
  the internal Fisher's ceiling while leaving ``Spectra``'s own at ``4*nside``,
  so one run carried two ceilings and the deconvolved Cl shifted by ~0.6% at
  the lowest bandpower against the coherent answer. On ``PICSLike`` the key was
  dropped entirely: it was never in that class's basis-key filter and had no
  setter alias, so the requested ceiling reached neither the Cls and beams nor
  the basis.

Version 1.1.0 (2026-07-21)
--------------------------

The in-memory pipeline programme: every input can now be handed in as an
array instead of a file path, and nothing is written to disk unless asked.

**Breaking changes:**

* **The default computation basis changed.** ``basis=None`` — the default —
  now resolves to ``method="auto"`` and the ADR-0003 cost selector picks
  harmonic or pixel-direct. Previously a bare ``Fisher(config)`` silently used
  the traditional no-basis pixel path. Pass ``basis=False`` to restore that
  path; it also remains the only way to reach the pixel-space attribute
  surface (``noise_cov1``, ``inv_cov1``, ``signal_matrix``) and the ADR-0016
  disk handoff. Scripts that used the bare constructor as a *traditional*
  reference are now comparing auto against auto (ADR-0018).
* ``compression=`` renamed to ``basis=`` on ``Fisher``, ``Spectra``, and
  ``PICSLike``. The old name is accepted for one release and emits a
  ``DeprecationWarning``; passing both raises ``TypeError`` (ADR-0018).
* **Persistence is opt-in.** Every ``out*`` default in ``InputParams`` is now
  ``None``, and providing a path *is* the save trigger. A bare config writes
  no files. The gate lives inside the four write primitives, so no call site
  can reintroduce implicit I/O. Configs that relied on the old default paths
  now silently write nothing (ADR-0015).
* ``do_cross`` now defaults to ``False`` (was ``True``).
* ``calibration`` and ``smooth_pol`` removed from ``InputParams``. The
  ``calibration:`` YAML key stays *recognised* so it can refuse: ``1.0`` is
  accepted, anything else raises ``ValueError``. The ``read_maps(...,
  calibration=)`` kwarg is hard-cut — Python's own ``TypeError`` names it.
* ``nspectra`` removed as a config key from all shipped and test YAMLs. It
  remains a derived attribute set by ``compute_derived()``.
* ``threadpoolctl`` dropped as a cosmocore dependency, along with the
  ``_wide_threadpool()`` context manager: raising OpenBLAS's thread limit
  past its init-time size segfaults under the usual ``OMP_NUM_THREADS=1``.
* The ``main_fisher.py``, ``main_qml.py`` and ``main_picslike.py`` scripts are
  removed, superseded by the console scripts below. They were never installed —
  only reachable from a repository checkout — and ``main_picslike.py`` had
  drifted out of sync with ``PICSLike`` to the point of calling three methods
  that no longer exist. Replace ``python main_qml.py cfg.yaml`` with
  ``qube-run cfg.yaml``, ``main_fisher.py`` with ``qube-fisher-run``, and
  ``main_picslike.py`` with ``picslike-run``.
* ``do_derivative_step`` loses its ``npixs`` and ``spins`` parameters: the
  signature is now ``(S, spectrum, current_ell, fields)``. Both values were
  always overwritten from ``fields`` on entry, so no result changes — but a
  stale call raises ``TypeError`` rather than being silently accepted. Removed
  outright rather than deprecated: they sat *ahead* of the required arguments,
  so any shim would have forced defaults onto ``current_ell``/``fields`` and
  silently misbound new-style positional calls (ADR-0018).
* A spin-2 field's two mask columns must now be identical. Masking Q
  differently from U is not representable — the V operator and the 2x2 Lambda
  blocks assume one pixel set per spin-2 field — and the previous code silently
  used the *second* column of the pair for both. ``ValueError`` at mask
  resolution (ADR-0017).
* ``Bins.lbin`` renamed to ``Bins.lmid``. The name ``lbin`` implied an
  *effective* multipole, which ``Bins`` cannot know — it is only the bin
  midpoint. ``lbin`` survives one release as a property that returns
  ``lmid`` and emits a ``DeprecationWarning``; use ``lmid`` for the
  midpoint or ``Fisher.get_effective_ells()`` for the effective multipole
  (ADR-0018, ADR-0019).
* ``Bins.bins()``, ``Bins.bin_spectra()`` and ``Bins.bin_covariance()``
  removed. No production path used them, and ``bin_spectra`` returned a
  *flat* bin average that silently disagrees with the Fisher-weighted
  bandpower the QML estimator produces — use
  ``Spectra.convolve_theory_for_inference()`` to bin theory (ADR-0007
  amendment, ADR-0019).
* **``output_convention="Dl"`` is now an estimator setting, not a
  presentation one** (ADR-0019). The flat-D_ℓ shape weight
  ``w_ℓ = 2π/(ℓ(ℓ+1))`` enters the binned QML derivative, so the estimated
  bandpower, its covariance and the Fisher matrix come out as ``D_b``
  natively; the post-hoc ``ℓ(ℓ+1)/2π`` scalar evaluated at the bin midpoint is
  **deleted**. Existing ``Dl`` runs now return different ``D_ℓ``, covariance
  and Fisher — varying bin to bin, largest where bins are wide and ℓ is low —
  with no call site changed and, by design, **no runtime warning**; pin the
  previous release to reproduce prior numbers. ``Cl`` output is unchanged.
  ``Dl`` and ``Cl`` now estimate genuinely different observables
  (``D_output ≠ scalar · C_output`` for wide bins), so ``output_convention``
  must be fixed *before* ``run()`` — flipping it on an already-run instance no
  longer rescales the output. Consequently the bandpower window
  (``Fisher.get_bandpower_window_function``) and
  ``Spectra.convolve_theory_for_inference`` now consume per-ℓ theory in the
  active convention: **D_ℓ, not C_ℓ, in Dl mode**. A likelihood that fed
  physical C_ℓ into a Dl window must convert to D_ℓ (``× ℓ(ℓ+1)/2π``) first.
* **``Spectra.get_effective_ells()`` no longer returns the bin midpoint by
  default.** It now returns the inverse-variance-weighted window centroid
  (delegated to ``Fisher.get_effective_ells()``), which requires a completed
  Fisher run and differs per spectrum. Pass ``use_midpoint=True`` for the old
  behaviour (the bin midpoint, no Fisher run). For a multi-spectrum run the
  default result is now shape ``(nspectra·nbins,)`` (spectrum-major), not
  ``(nbins,)`` (ADR-0019).

**Added:**

* **In-memory input injection** (ADR-0017). Every pipeline input resolves
  through two adapters — a path in ``InputParams`` or an in-memory object on
  the constructor — and the injected object wins. Fixed kwarg vocabulary:
  ``mask``, ``noise_cov1``/``noise_cov2``, ``cls_data``, ``fiducial_cls``,
  ``beam`` on ``Core`` and all three orchestration classes; ``maps1``/``maps2``
  on ``Spectra`` and ``PICSLike`` only (``Fisher`` never reads maps). An
  injected object is defined as exactly what the reader would have returned:
  ``mask`` is ``(npix,)`` or ``(npix, nfields)``, ``noise_cov`` and ``maps``
  are *reduced* to active pixels, and ``cls_data``/``fiducial_cls`` must be
  physical C_ℓ — the ``input_convention`` D_ℓ → C_ℓ conversion is applied only
  on the file path. File-based workflows are unchanged.
* **In-memory Fisher → Spectra handoff** (ADR-0016). ``Spectra(fisher=…)``
  aliases the live Fisher's covariances instead of round-tripping N and C⁻¹
  through ``np.fromfile``. The ``out*`` files remain a dormant read adapter.
* ``cosmocore.active_pixel_index(mask)`` and ``cosmocore.active_pixels(mask)``:
  the ordering of every *reduced* array (``noise_cov1/2``, ``maps1/2``), as a
  pure function of the mask. Callers injecting those arrays no longer need to
  build a ``Fisher`` to learn the active-pixel ordering (ADR-0017)::

      index = active_pixel_index(mask)
      maps = maps_full.reshape(nfields * npix, nsims)[index]
      cov = cov_full[np.ix_(index, index)]
* **Console scripts.** The pipelines are now installed entry points, on
  ``PATH`` after a plain ``pip install`` with no repository checkout:
  ``qube-fisher-run`` (Fisher matrix only), ``qube-spectra-run`` (QML spectra
  end to end) with the alias ``qube-run``, and ``picslike-run`` (pixel
  likelihood over the grid). They run unmodified under MPI, e.g.
  ``mpirun -n 8 qube-run config.yaml``. The QUBE commands take an optional
  config path and fall back to the packaged ``TEB_defaults.yaml``;
  ``picslike-run`` requires one. Each warns when it finishes a run whose result
  is not being persisted, rather than exiting quietly (ADR-0015).
* ``in_memory_inputs.ipynb`` notebook and tutorial page, driving the file and
  array adapters over the same data and asserting they agree.
* The demo notebooks now execute in CI (``test-notebooks`` job, matrix over
  qube and picslike), asserting no files land on disk — the ADR-0015/0017
  acceptance test.
* **Multi-spectrum bandpower window.**
  ``Fisher.get_bandpower_window_function()`` no longer raises for
  ``nspectra > 1``; it returns the block-structured window of shape
  ``(nspectra·n_bins, nspectra·n_ell)`` so TEB/TQU runs can convolve per-ℓ
  theory into expected bandpowers. ``Spectra.convolve_theory_for_inference()``
  now accepts either a flat ``(nspectra·n_ell,)`` per-ℓ array (spectrum-major,
  matching ``get_bandpower_slices``) or a label-keyed dict
  ``{"TT": …, "EE": …}``. Single-spectrum output is numerically unchanged.
* ``Bins.shape_weights(convention)`` returns the per-ℓ bandpower shape
  weight ``w_ℓ`` that declares the in-bin spectrum shape: ``1`` for the
  flat-C_ℓ shape (``"Cl"``), ``2π/(ℓ(ℓ+1))`` for the flat-D_ℓ shape
  (``"Dl"``). Requesting ``"Dl"`` when a bin includes ℓ = 0 raises, since
  D_ℓ is undefined at the monopole. This weight now feeds the binned QML
  derivative (see the ``output_convention`` breaking change above).
* ``Fisher.get_effective_ells()``: the effective multipole per bin as the
  inverse-variance-weighted bandpower-window centroid, reported in the active
  ``output_convention`` and per spectrum. Returns shape ``(nspectra·nbins,)``
  on rank 0, ``None`` on workers; it enters the collective per-ℓ Fisher on
  every rank (no worker early-return) and raises before a completed Fisher run
  (ADR-0019). ``Spectra.get_effective_ells(use_midpoint=False)`` delegates to
  it, with ``use_midpoint=True`` for the bin midpoint.

**Fixed:**

* ``read_mask`` honours ``params.ordering`` via an explicit ``nest=``
  argument. ``healpy.read_map``'s default previously force-converted to RING,
  silently producing wrong sky positions for ``ordering="NESTED"``.
* Different temperature and polarisation masks no longer crash. Unequal active
  pixel counts across components raised ``IndexError`` during geometry setup —
  the standard configuration for a CMB analysis.
* With ``spins=[2, 0]`` (polarisation declared before temperature), the
  temperature field was given the *polarisation* mask's sky positions, silently
  and with no error. Pointing vectors are now built per field.
* A transposed ``(ncomponents, npix)`` mask is refused instead of silently
  reducing every array to the wrong pixels.
* On a harmonic basis, the consumers that need the pixel projector
  (``get_covariance``, ``get_inverse``, ``to_basis``, ``projector``, m-block
  compression and PICSLike ``do_cross``) now raise a ``RuntimeError`` naming
  the contract when called after ``release_pixel_projector()``, instead of an
  opaque ``ValueError`` from numpy.
* Evaluating the same parameter point twice on one ``PICSLike`` instance no
  longer drifts. Beam smoothing multiplied the theory spectra in place, so each
  evaluation re-smoothed the parameter grid's own arrays and chi-squared grew
  with every pass. Spectra handed to ``set_cls`` are now copied, which also
  means an injected ``cls_data=`` array is no longer modified by the run that
  consumes it.
* A C_ℓ longer than the analysis ``lmax`` is truncated on the way in rather
  than failing to broadcast inside beam smoothing. Handing a full CAMB run to a
  small-``lmax`` analysis is the normal case; only the dict form of ``cls_data``
  was affected, the array form already truncated.
* Rebuilding the computation basis on a live ``PICSLike`` instance no longer
  produces wrong chi-squared. The compressed-SMW path cached data projected
  through the *previous* basis and kept using it against the new one — silently,
  with no error. ``setup_computation_basis()`` now drops that cache, so the
  documented ``setup_computation_basis → setup_maps → compute`` order is no
  longer the only safe one. The same staleness hit the MPI worker ranks on a
  second pipeline pass, where the cache outlived the broadcast that replaced the
  basis it came from; rank 0 stayed correct, so the disagreement was invisible.

Version 1.0.1 (2026-05-21)
--------------------------

Documentation-only patch release. Refreshes the long descriptions on PyPI
for ``cosmocore``, ``qube-qml``, and ``picslike`` so they reflect the
published companion paper and the current install path.

**Documentation:**

* Paper I (Galloni & Pagano 2026, ``arXiv:2605.21149``) cited in all
  READMEs and in ``CITATION.cff``. BibTeX entries gained
  ``eprint``/``archivePrefix``/``primaryClass`` fields; the legacy
  ``"in preparation"`` notes are removed.
* ``CITATION.cff`` ``url:`` field migrated from the retired GitHub Pages
  location to the canonical Read the Docs URL.
* ``cosmocore``, ``qube-qml``, and ``picslike`` READMEs gained proper
  PyPI installation instructions (previously instructed users to clone
  the repository and run ``pip install -e``); the MPI extra is now
  documented as ``pip install <package>[mpi]`` instead of a bare
  ``pip install mpi4py``.

Version 1.0.0 (2026-05-20)
--------------------------

First stable release. CosmoForge is now a public, paper-validated framework
covering QML power spectrum estimation and pixel-space Gaussian likelihood
evaluation for spin-0 and spin-2 fields on the sphere.

**API stabilisation:**

* ``SpectrumKey`` type system replaces the legacy ``(comp_i, comp_j, mode)``
  tuple-based keying across cosmocore, qube, and picslike. ``SpectrumKind``
  enumerates ordered slot pairs (``SS``, ``GG/CC/GC``, ``SG/SC``); CMB aliases
  (``TT``, ``EE``, ``BB``, ``EB``, ``BE``, ``TE``, ...) and the
  ``to_cmb_canonical`` re-keying helper live in ``cosmocore.conventions.cmb``.
* Label-keyed user-facing accessors on ``Fisher`` and ``Spectra``; legacy
  tuple-keyed inputs and tuple-rewrap shims removed.
* ``SymmetryMode`` flag on ``Fisher`` and ``Spectra`` (``SYMMETRIC`` default,
  ``DIRECTIONAL`` for calibration diagnostics) controls how cross-component
  spin-2 ``GC`` / ``CG`` blocks are handled; the directional path uses
  separate ``Lambda`` and derivative E matrices (ADR-0011).

**Architecture:**

* ``ComputationBasis`` polymorphism convention closed (ADR-0002):
  ``prepare_for_basis``/``BasisPrepared`` contract, ``quadratic_form``
  abstract method, basis-seam leak in ``qube/spectra.py`` closed, single
  SMW-cross quadratic-form path.
* ``PixelBasis`` class-model cleanup: ``use_direct`` flag dropped (pixel-direct
  is the natural default when no ``epsilon``/``mode_fraction`` is passed);
  ``basis`` kwarg renamed to ``compression_target``; ``apply_compression``
  internalised.
* Top-level ``harmonic.py``/``pixel.py`` renamed to remove name collision with
  the ``basis/`` subpackage.

**Numerical stability:**

* Stable SMW rewrite in ``qube/spectra.py`` (replaced the
  catastrophically-cancellation-prone ``M - M K^{-1} M`` form with
  ``M (I + Lambda M)^{-1}``); fixes high-SNR negative ``VCVT`` and yields
  a ~5x speed-up in the multi-spectrum harmonic path.

**Packaging:**

* ``mpi4py`` is now an optional dependency (``cosmocore[mpi]``,
  ``qube-qml[mpi]``, ``picslike[mpi]``). CI runs the test matrix on both
  branches of ``cosmocore._mpi``.
* All remaining ``np.linalg`` / ``scipy.linalg`` call sites routed through
  ``cosmocore.basics`` for consistency.

**Validation:**

* End-to-end reproduction of the Planck low-:math:`\ell` Fortran reference
  for both the QML and pixel-space likelihood pipelines, consistent with
  double-precision arithmetic.

**Documentation:**

* Hosting migrated from GitHub Pages to Read the Docs.
* Doc-drift sweep policy: at the end of every feature touching public surface,
  grep for renamed symbols, rebuild the Sphinx tree, and verify CONTEXT.md
  and per-package READMEs.
* Tutorials refreshed for the SpectrumKey/SymmetryMode API and re-executed
  end-to-end at tier-A.
* Installation guide reworked (developer install opt-in,
  ``pymaster``/``namaster`` trap documented).
* ``CONTEXT.md`` updated with Slot, SpectrumKind, SpectrumKey, SymmetryMode.

**Distribution:**

* Three independently pip-installable packages — ``cosmocore`` (foundation),
  ``qube-qml`` (Fisher and QML estimation), ``picslike`` (pixel-space
  Gaussian likelihood) — plus the ``cosmoforge`` umbrella metapackage that
  installs all three.
