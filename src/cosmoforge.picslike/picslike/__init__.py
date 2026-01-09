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
.. [2] Jewell, J., Levin, S. & Anderson, C.H. "Application of Monte Carlo algorithms
   to the Bayesian analysis of the cosmic microwave background"
   Astrophys. J. 609, 1-14 (2004)
.. [3] Eriksen, H.K. et al. "Power Spectrum Estimation from High-Resolution Maps by
   Gibbs Sampling" Astrophys. J. Suppl. 155, 227-241 (2004)
.. [4] Planck Collaboration "Planck 2018 results. V. CMB power spectra and likelihoods"
   Astron. Astrophys. 641, A5 (2020)
"""

from .likelihood_result import LikelihoodResult
from .parameter_grid import ParameterGrid
from .picslike import PICSLike

__all__ = ["PICSLike", "ParameterGrid", "LikelihoodResult"]
