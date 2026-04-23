"""
Computation basis methods for CMB Fisher matrix computation.

This module provides two computation basis approaches:

1. **HarmonicBasis** (Tegmark-like): Direct transformation to harmonic space
   (n_pix → n_modes). Fast and efficient when n_modes << n_pix.

2. **PixelBasis** (Gjerløw-like): Pixel-space projector with
   eigenvalue truncation (n_pix → n_kept). More flexible, handles systematics
   through custom projectors.

Use **create_computation_basis** factory function to create basis instances.

Available eigenvalue bases for PixelBasis:

- **harmonic**: P_h = V^T V (pure harmonic projector)
- **noise_weighted**: P_h N^{-1} P_h (inverse noise weighting)
- **total_covariance**: P_h C^{-1} P_h where C = N + S
- **snr**: S^{1/2} N^{-1} S^{1/2} (signal-to-noise ratio)

References
----------
.. [1] Tegmark, M. "How to measure CMB power spectra without losing information"
   Phys. Rev. D 55, 5895 (1997)
.. [2] Gjerløw, E. et al. "Component separation for the CMB with a
   low-resolution analysis" A&A 629, A51 (2019)
"""

from __future__ import annotations

import inspect

import numpy as np

from .base import ComputationBasis, SMWPrepared
from .harmonic import HarmonicBasis
from .pixel import COMPRESSION_BASES, PixelBasis

_BASIS_CLASSES: dict[str, type[ComputationBasis]] = {
    "harmonic": HarmonicBasis,
    "pixel": PixelBasis,
}


def create_computation_basis(
    method: str,
    N: np.ndarray,
    N_inv: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    lmax: int,
    **kwargs,
) -> ComputationBasis:
    """
    Factory function to create the appropriate computation basis implementation.

    Parameters
    ----------
    method : str
        Computation basis method: "harmonic" or "pixel".
    N : numpy.ndarray
        Noise covariance matrix.
    N_inv : numpy.ndarray
        Precomputed noise inverse matrix.
    theta : numpy.ndarray
        Colatitude angles for active pixels in radians.
    phi : numpy.ndarray
        Longitude angles for active pixels in radians.
    lmax : int
        Maximum multipole for harmonic expansion.
    **kwargs
        Additional keyword arguments passed to the basis constructor
        (beam, spins, basis, C_ell, epsilon, mode_fraction, etc.).
        Arguments not accepted by the chosen class are silently ignored.

    Returns
    -------
    ComputationBasis
        Configured basis instance (not yet set up — call .setup()).
    """
    if method not in _BASIS_CLASSES:
        raise ValueError(
            f"Unknown computation basis method '{method}'. "
            f"Available: {list(_BASIS_CLASSES)}"
        )
    cls = _BASIS_CLASSES[method]
    # Filter kwargs to only those accepted by the target class
    sig = inspect.signature(cls.__init__)
    accepted = {
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(N, N_inv, theta, phi, lmax, **filtered)


__all__ = [
    "ComputationBasis",
    "COMPRESSION_BASES",
    "HarmonicBasis",
    "PixelBasis",
    "SMWPrepared",
    "create_computation_basis",
]
