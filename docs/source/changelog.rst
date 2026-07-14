Changelog
=========

All notable changes to CosmoForge will be documented here.

Unreleased
----------

**Breaking changes:**

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

**Added:**

* ``cosmocore.active_pixel_index(mask)`` and ``cosmocore.active_pixels(mask)``:
  the ordering of every *reduced* array (``noise_cov1/2``, ``maps1/2``), as a
  pure function of the mask. Callers injecting those arrays no longer need to
  build a ``Fisher`` to learn the active-pixel ordering (ADR-0017)::

      index = active_pixel_index(mask)
      maps = maps_full.reshape(nfields * npix, nsims)[index]
      cov = cov_full[np.ix_(index, index)]

**Fixed:**

* Different temperature and polarisation masks no longer crash. Unequal active
  pixel counts across components raised ``IndexError`` during geometry setup —
  the standard configuration for a CMB analysis.
* With ``spins=[2, 0]`` (polarisation declared before temperature), the
  temperature field was given the *polarisation* mask's sky positions, silently
  and with no error. Pointing vectors are now built per field.
* A transposed ``(ncomponents, npix)`` mask is refused instead of silently
  reducing every array to the wrong pixels.

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
