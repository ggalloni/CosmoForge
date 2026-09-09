Pixel-Space Filters
===================

Real data reaches a map-space estimator already filtered: a scan-synchronous polynomial
has been fitted and removed, a band of multipoles has been deprojected, a template stack
has been regressed out. CosmoForge takes that operator as an input, ``pixel_filter=``,
and carries the whole analysis out inside the subspace the filter leaves behind.

CosmoForge does **not** filter your maps. It analyses maps that were filtered, which is a
different job: what it needs from you is the operator, so that the model covariance it
compares against has been through the same thing your data has.

The design is recorded in ADR-0020 (*filters are restriction to the operator's range*),
under ``docs/adr/``.

Restriction, not conjugation
----------------------------

The obvious way to handle a filter ``F`` is to conjugate every covariance-shaped object,
``C → F C Fᵀ``, and regularise the singular result with a ridge. CosmoForge does not do
that. An optimal estimator depends on a linear ``F`` only through its row space above the
rank threshold, so the library keeps the thin SVD ``F = U Σ Wᵀ`` truncated at rank ``r``
and works in ``r`` dimensions throughout.

Consequences you can see from the outside:

* the returned spectra, Fisher matrix and likelihood are those of the **filtered** data;
* there is no ridge parameter to choose or disclose;
* the restricted problem is smaller than the unfiltered one, which is a saving on the
  basis paths and a wash on the kernel paths.

Building a filter
-----------------

Three constructors cover the common operators. Every instrument-specific number is a
required argument, so nothing about a particular scan is assumed for you.

.. code-block:: python

   import numpy as np
   from cosmocore import harmonic_deprojection, scan_polynomial, hits_weighting

   # Remove whole multipoles of one slot. Templates come from the estimator's
   # own harmonic operator V, so the removed subspace is exactly the modes the
   # Fisher works in, not a resynthesised approximation of them.
   band = harmonic_deprojection(mask=mask, spins=[2], ells=[2, 3, 4], slot="E")

   # Remove Legendre polynomials in azimuth within bands of constant polar
   # angle about the scan axis: the pixel-space image of per-scan baseline
   # removal in the time domain.
   scan = scan_polynomial(mask=mask, pole=(theta, phi), n_stripes=8, degree=2)

   # Inverse-noise weighting. Invertible, therefore a no-op control.
   weights = hits_weighting(hits, mask=mask, range_epsilon=1e-6)

   # Combine two projectors: the intersection of what each keeps.
   both = band & scan

For anything else, hand the operator or the templates in directly:

.. code-block:: python

   from cosmocore import Filter

   Filter.from_operator(F, range_epsilon=1e-3)   # any linear operator
   Filter.from_deprojection(T)                   # annihilate a template stack
   Filter.from_subspace(W)                       # keep an orthonormal subspace

Deproject one family per filter and combine with ``&``. A single stack mixing families
would put one relative cut across template column norms that are not commensurable, and
the weaker family can be deleted wholesale.

The rank threshold
------------------

**There is no default, and asking for one raises.** The right cut depends on the
operator's spectrum, and a wrong default is invisible in the output.

.. list-table::
   :header-rows: 1
   :widths: 26 20 54

   * - Knob
     - Lives on
     - Cuts
   * - ``template_epsilon``
     - the constructors
     - the template stack, deciding how many templates are independent
   * - ``range_epsilon``
     - ``Filter``
     - ``F``'s own singular values, relative to the largest
   * - ``range_rank``
     - ``Filter``
     - ``F``'s singular values, at an exact count

``range_epsilon`` and ``range_rank`` are mutually exclusive.

``range_epsilon`` clips at ``MIN_RANGE_EPSILON`` (``sqrt(eps)``, about 1.5e-8) with a
warning. Internal coordinates are ``Σ Wᵀ d``, so the restricted covariance carries ``Σ²``
and its condition number goes as ``range_epsilon⁻²``: a machine-precision cut would put a
graded operator at condition 1e26. Two more warnings fire where they should: truncating
outside a spectral gap reports where the largest gap actually is, and keeping every
singular value tells you the filter is invertible and quotes the resulting condition
number.

Handing it to the pipeline
--------------------------

.. code-block:: python

   from qube import Fisher, Spectra

   fisher = Fisher("config.yaml", mask=mask, pixel_filter=band)
   fisher.run()

   spectra = Spectra("config.yaml", fisher=fisher)
   spectra.run()

``Fisher``, ``Spectra`` and ``PICSLike`` all accept ``pixel_filter=``, on every path
(traditional ``basis=False``, pixel-direct, compressed pixel, harmonic).
``Spectra(fisher=…)`` adopts that Fisher's filter and raises if the constructor asks for
a different one: a filtered ``Spectra`` normalised by an unfiltered Fisher would be wrong
in a way no shape check catches.

The filter is fingerprinted against the mask it was built with, and the fingerprint is
checked once, so a filter whose rows would be permuted relative to this run's
active-pixel ordering is refused rather than silently misapplied.

Projector or graded, and why it matters for noise
-------------------------------------------------

A **projector** keeps every surviving direction at unit weight: ``U`` is ``W`` and
``Σ = I``. Deprojection and purification are projectors. A **graded** filter has a
non-trivial ``Σ``: a transfer function, a soft high-pass, hits weighting.

The distinction is not cosmetic, and it decides what the pre-filtered flags mean.

.. code-block:: python

   Spectra("config.yaml", fisher=fisher, maps1=d, maps_prefiltered=True)
   Fisher("config.yaml", noise_cov1=N, pixel_filter=f, noise_prefiltered=True)

Both flags say the array **as handed** already carries the filter, so it is restricted
with ``Uᵀ`` instead of ``Σ Wᵀ``. They are independent, because a pipeline whose noise
simulations went through the filter alongside raw signal maps is a legitimate
combination that one merged flag would forbid.

``noise_prefiltered`` means *the covariance of the noise in the maps as handed*. It never
means *the filter whitened my noise*. That reading conflates time-domain whitening with
map-domain projection: white pre-projection noise pushed through a projector becomes
``σ² F Fᵀ``, which is correlated and has zero power along the removed directions.
Declaring a white post-filter ``N'`` claims noise power along modes the filter deleted.

* **Projector**: the two conventions are *identical*, since
  ``Uᵀ (σ² I) U = Uᵀ (σ² F Fᵀ) U``. The flag is free, and getting it wrong costs nothing.
* **Graded**: they differ by ``Σ²`` and are *different noise models*. The flag is a
  physical claim about your pipeline, and getting it wrong biases the result.

Mixing conventions, pre-filtered maps with raw noise or the reverse, is the classic
wrong-covariance mistake and stays the caller's responsibility. CosmoForge warns when
pre-filtered maps carry non-negligible power outside ``span(U)``, and reports the
fraction of noise power sitting in the filter's null space. The latter cannot be a hard
check: a noise model declared white after filtering has power there by construction,
which is exactly the modelling error the warning is about.

What a filter refuses
---------------------

These raise ``NotImplementedError`` rather than degrading quietly:

* m-block compression (``delta_m=0``), whose exactness rests on an azimuthal symmetry a
  filter breaks;
* per-field ``epsilon`` / ``mode_fraction`` lists (``mode_fraction`` as a scalar is the
  recommended compression knob under a filter);
* per-field eigenspectrum diagnostics, which are undefined once ``W`` mixes fields.

Field-block structure is switched off wholesale under any filter, including one that
happens to mix nothing. Block-diagonality cannot be recovered from a global ``W``: for a
projector every singular value is 1, so degenerate SVD columns may be rotated across
blocks. It has to be declared at construction, and 1.3.0 has no way to declare it.

Separately, the dormant ``out*`` covariance file handoff is unsupported with a filter: it
writes the restricted matrix while the read side reshapes to the pointing count. Use the
in-memory handoff, which has been the primary path since ADR-0016.

Cost
----

On the V-based paths the filter is folded into ``V`` once, ``V' = V W Σ``, and nothing
else in the basis has to know about it. On the kernel paths the derivative is restricted
at the **binned** ``E_b``, so conjugation is paid once per bin instead of once per
multipole; the saving approaches the multipole-per-bin count as the problem grows.

The kernels still fill the full pixel-space matrix before conjugating, so there is no
memory win on those paths, only a smaller solve. Traditional-path ``PICSLike`` pays the
conjugation at every grid point, so filtered parameter grids belong on the basis paths
where the filter is paid once.
