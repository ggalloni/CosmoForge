Changelog
=========

All notable changes to CosmoForge will be documented here.

Every pull request that touches package source updates this file; see
``CLAUDE.md`` and the pull-request template.

Unreleased
----------

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
* ``main_fisher.py`` and ``main_qml.py`` take an optional positional config
  path, defaulting to the packaged ``TEB_defaults.yaml``. Previously the path
  was hardcoded relative to the repo root and only worked from there.
* ``in_memory_inputs.ipynb`` notebook and tutorial page, driving the file and
  array adapters over the same data and asserting they agree.
* The demo notebooks now execute in CI (``test-notebooks`` job, matrix over
  qube and picslike), asserting no files land on disk — the ADR-0015/0017
  acceptance test.

**Fixed:**

* ``read_mask`` honours ``params.ordering`` via an explicit ``nest=``
  argument. ``healpy.read_map``'s default previously force-converted to RING,
  silently producing wrong sky positions for ``ordering="NESTED"``.

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
