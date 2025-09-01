"""
PICSLike: Pixel-based likelihood for cosmological parameter inference.

This package implements pixel-based likelihood analysis for cosmological parameter
estimation, providing an alternative to harmonic-space methods. The approach is
particularly useful for handling incomplete sky coverage, non-Gaussian features,
and certain computational scenarios where pixel-space analysis is advantageous.

Classes
-------
PICSLike
    Main class for pixel-based likelihood computation.
ParameterGrid
    Helper class for managing parameter ranges and theoretical spectra.
LikelihoodResult
    Container for likelihood computation results.

Notes
-----
The pixel-based likelihood approach computes the likelihood function directly
in map pixel space, avoiding the complications that can arise from harmonic
transformations of masked or incomplete data.

References
----------
.. [1] Wandelt, B.D., Larson, D.L. & Lakshminarayanan, A. "Global, exact cosmic
   microwave background data analysis using Gibbs sampling"
   Phys. Rev. D 70, 083511 (2004)
.. [2] Jewell, J., Levin, S. & Anderson, C.H. "Application of MCMC methods to
   multi-frequency CMB data sets" Astrophys. J. 609, 1-6 (2004)
"""

from .likelihood_result import LikelihoodResult
from .parameter_grid import ParameterGrid
from .picslike import PICSLike

__all__ = ["PICSLike", "ParameterGrid", "LikelihoodResult"]
