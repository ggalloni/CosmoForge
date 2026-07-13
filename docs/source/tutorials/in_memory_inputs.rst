In-Memory Inputs
================

Every pipeline input can be handed to CosmoForge as an **in-memory array** instead of a
file path. This is for callers that already hold their data — an upstream component
pipeline, or an interactive session — and should not have to round-trip it through disk
just to run an analysis.

The design is recorded in ADR-0017 (*file-or-array loading seams*), under ``docs/adr/``.

Two adapters, one contract
--------------------------

Every input is loadable through exactly two adapters, converging on a single in-memory
contract:

* the **file adapter** — a path in ``InputParams``, parsed by a low-level reader;
* the **injection adapter** — the object handed straight to the high-level class.

An injected object always wins over the corresponding path. The configuration still
supplies the scalars (``nside``, ``lmax``, ``nsims``, …); only the *data* moves.

The injected object is defined as **exactly what the reader would have returned** — the
injection adapter adds no new accepted forms.

The kwarg vocabulary
--------------------

.. list-table::
   :header-rows: 1
   :widths: 22 26 52

   * - Kwarg
     - Shadows
     - In-memory contract
   * - ``mask``
     - ``maskfile``
     - ``(npix,)`` or ``(npix, nfields)``, pixel-indexed per ``params.ordering``
   * - ``noise_cov1``, ``noise_cov2``
     - ``covmatfile1/2``
     - **reduced** ``(n_active, n_active)``
   * - ``maps1``, ``maps2``
     - ``inputmapfile1/2``
     - **reduced** ``(n_active, n_sims)``
   * - ``cls_data``
     - ``inputclfile``
     - ``{label: C_ell}`` dict of **physical** C_ell
   * - ``fiducial_cls``
     - ``fiducialfile``
     - as ``cls_data``
   * - ``beam``
     - ``beam_file``
     - 2-D, ≥3 rows (T/E/B window functions)

``mask``, ``cls_data``, ``fiducial_cls`` and ``beam`` live on ``Core``, so they are
accepted by ``Fisher``, ``Spectra`` and ``PICSLike`` alike. ``maps1``/``maps2`` are a
``Spectra``/``PICSLike`` seam — ``Fisher`` never reads maps.

A disk-free run
---------------

.. code-block:: python

   from qube import Fisher, Spectra

   fisher = Fisher(
       params,
       mask=mask,
       noise_cov1=noise_cov,
       cls_data=cls,
       fiducial_cls=cls,
       beam=beam,
   )
   fisher.run()

   spectra = Spectra(params, fisher=fisher, maps1=maps)
   spectra.run()
   power_spectra = spectra.get_power_spectra(mode="deconvolved")

Combine this with opt-in persistence — leave every ``out*`` path unset — and the run reads
nothing and writes nothing. The working directory is untouched from end to end.

One contract worth reading twice
--------------------------------

Every seam takes *exactly what its own reader would have returned* — so an injected array is
used verbatim, and any conversion the file path performs on the way in has already happened
by the time the reader returns.

That matters in one place: ``cls_data``/``fiducial_cls`` must be **physical** C_ell. The
``input_convention`` normalisation (e.g. D_ell → C_ell) is applied only on the file path,
never to an injected array. Convert before you inject.

The active-pixel index
----------------------

``noise_cov1``/``noise_cov2`` and ``maps1``/``maps2`` are **reduced** — restricted to the
active (unmasked) pixels and ordered by the *active-pixel index*. QML works on active
pixels, so this is the domain, not an implementation detail: a full-sky noise covariance
is ``(npix * nfields)²``, which is 72 GiB for QU at nside 64 against 0.72 GiB reduced at
fsky 0.1.

You do not need a ``Fisher`` (or any framework object) to learn the ordering. It is a pure
function of the mask:

.. code-block:: python

   from cosmocore import active_pixel_index

   index = active_pixel_index(mask)          # mask: (npix, nfields)

   maps = maps_full.reshape(nfields * npix, nsims)[index]   # (n_active, nsims)
   cov = cov_full[np.ix_(index, index)]                     # (n_active, n_active)

Each entry of ``index`` gives that reduced row's position in the flattened full-pixel
``(nfields * npix,)`` vector. This is the same function the framework itself uses, so the
published ordering *is* the ordering (ADR-0017).

You are free to build the reduced covariance any way you like — the one-liner above is a
convenience, not a requirement. A caller that can produce it directly (from a noise model,
block by block, or streamed) never has to materialise the full matrix at all.

``cosmocore.active_pixels(mask)`` returns the per-component index arrays if you need them
individually.

Worked example
--------------

``src/cosmoforge.qube/notebooks/in_memory_inputs.ipynb`` drives **both** adapters over the
same data — reading each input with its low-level reader, then handing the arrays in with
every input path pointed at a nonexistent location — and asserts the two agree bit-for-bit
(identical Fisher matrix and identical power spectra).

Run it with the ``jupyter`` dependency group:

.. code-block:: bash

   uv sync --group jupyter
   uv run --group jupyter jupyter lab src/cosmoforge.qube/notebooks/in_memory_inputs.ipynb
