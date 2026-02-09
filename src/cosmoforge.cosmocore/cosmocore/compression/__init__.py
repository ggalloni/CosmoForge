"""
Compression methods for CMB Fisher matrix computation.

This module provides two compression approaches:

1. **HarmonicCompression** (Tegmark-like): Direct transformation to harmonic space
   (n_pix → n_modes). Fast and efficient when n_modes << n_pix.

2. **PixelProjectedCompression** (Gjerløw-like): Pixel-space projector with
   eigenvalue compression (n_pix → n_kept). More flexible, handles systematics
   through custom projectors.

Use **create_compression** factory function to create compression instances.

Available compression bases for PixelProjectedCompression:

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

from .base import BaseCompression, SMWPrepared
from .harmonic import HarmonicCompression
from .pixel_projected import COMPRESSION_BASES, PixelProjectedCompression

_COMPRESSION_CLASSES: dict[str, type[BaseCompression]] = {
    "harmonic": HarmonicCompression,
    "pixel_projected": PixelProjectedCompression,
}


def create_compression(
    method: str,
    N: np.ndarray,
    N_inv: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    lmax: int,
    **kwargs,
) -> BaseCompression:
    """
    Factory function to create the appropriate compression implementation.

    Parameters
    ----------
    method : str
        Compression method: "harmonic" or "pixel_projected".
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
        Additional keyword arguments passed to the compression constructor
        (beam, spins, basis, C_ell, epsilon, mode_fraction, etc.).
        Arguments not accepted by the chosen class are silently ignored.

    Returns
    -------
    BaseCompression
        Configured compression instance (not yet set up — call .setup()).
    """
    if method not in _COMPRESSION_CLASSES:
        raise ValueError(
            f"Unknown compression method '{method}'. "
            f"Available: {list(_COMPRESSION_CLASSES)}"
        )
    cls = _COMPRESSION_CLASSES[method]
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
    "BaseCompression",
    "COMPRESSION_BASES",
    "HarmonicCompression",
    "PixelProjectedCompression",
    "SMWPrepared",
    "create_compression",
]
