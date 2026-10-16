cosmocore.filters module
========================

Pixel-space filters as restriction to the operator's range (ADR-0020). Defines
:class:`Filter`, the frozen record holding the thin SVD ``F = U Σ Wᵀ`` truncated
at a user-supplied rank threshold, and constructors for the common operators:
:func:`harmonic_deprojection`, :func:`scan_polynomial` and
:func:`hits_weighting`. A filter is handed to ``Fisher``, ``Spectra`` and
``PICSLike`` as ``pixel_filter=``. See the *Pixel-Space Filters* tutorial for
the projector-versus-graded distinction and what it means for noise.

.. Note: no ``:undoc-members:``. ``Filter`` is a dataclass whose fields are
   documented in its class docstring's ``Attributes`` section, and autodoc
   would register each one a second time from the annotation.

.. automodule:: cosmocore.filters
   :members:
   :show-inheritance:
