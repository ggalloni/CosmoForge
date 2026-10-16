Citing CosmoForge
=================

Any work that uses CosmoForge should cite Paper I. Work that applies a
pixel-space filter should cite Paper II as well.

Paper I: always
---------------

Paper I describes the framework: the Fisher matrix, the QML estimator, the
pixel-based likelihood and the computation bases.

   Galloni, G. & Pagano, L., *CosmoForge. I. A unified framework for quadratic
   maximum likelihood power spectrum estimation and pixel-based likelihood
   analysis*, A&A (2026),
   `doi:10.1051/0004-6361/202661478 <https://doi.org/10.1051/0004-6361/202661478>`_,
   `arXiv:2605.21149 <https://arxiv.org/abs/2605.21149>`_.

.. code-block:: bibtex

   @article{GalloniPagano_CosmoForgeI,
       author        = {Galloni, G. and Pagano, L.},
       title         = {{CosmoForge}. {I}. A unified framework for quadratic maximum likelihood power spectrum estimation and pixel-based likelihood analysis},
       journal       = {Astronomy \& Astrophysics},
       year          = {2026},
       doi           = {10.1051/0004-6361/202661478},
       eprint        = {2605.21149},
       archivePrefix = {arXiv},
       primaryClass  = {astro-ph.CO},
   }

Paper II: pixel-space filters
-----------------------------

Paper II describes how a linear filter enters the QML estimator and the
pixel-based likelihood exactly. Cite it when an analysis passes
``pixel_filter=`` to ``Fisher``, ``Spectra`` or ``PICSLike``, or uses any
filter built from ``cosmocore.filters``.

   Galloni, G., *CosmoForge. II. An exact treatment of arbitrary linear filters
   in quadratic maximum-likelihood power spectrum estimation*,
   `arXiv:2610.18012 <https://arxiv.org/abs/2610.18012>`_ (2026).

.. code-block:: bibtex

   @article{Galloni_CosmoForgeII,
       author        = {Galloni, G.},
       title         = {{CosmoForge}. {II}. An exact treatment of arbitrary linear filters in quadratic maximum-likelihood power spectrum estimation},
       year          = {2026},
       eprint        = {2610.18012},
       archivePrefix = {arXiv},
       primaryClass  = {astro-ph.CO},
   }

The same references are in ``CITATION.cff`` at the repository root, which
GitHub renders as "Cite this repository".
