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
import warnings

import numpy as np

from .base import ComputationBasis, SMWPrepared
from .harmonic import HarmonicBasis
from .pixel import COMPRESSION_BASES, PixelBasis

_BASIS_CLASSES: dict[str, type[ComputationBasis]] = {
    "harmonic": HarmonicBasis,
    "pixel": PixelBasis,
}


def _problem_dimensions(
    theta: np.ndarray | tuple[np.ndarray, ...],
    spins: list[int] | None,
    lmax: int,
    lswitch_high: int | None = None,
) -> tuple[int, int]:
    """Compute (n_pix, n_modes) at the effective lmax (after lswitch)."""
    if isinstance(theta, np.ndarray):
        thetas = (theta,)
    else:
        thetas = tuple(theta)

    n_components = len(thetas)
    if spins is None:
        spins = [0] * n_components

    n_pix = sum(2 * len(t) if spins[i] == 2 else len(t) for i, t in enumerate(thetas))

    effective_lmax = lswitch_high if lswitch_high is not None else lmax
    n_modes_base = (effective_lmax + 1) ** 2 - 4
    n_modes = sum(
        2 * n_modes_base if spins[i] == 2 else n_modes_base for i in range(n_components)
    )

    return n_pix, n_modes


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
        Computation basis method:
        - "harmonic": Tegmark-like direct harmonic transformation
        - "pixel": Gjerløw-like pixel-space projector with eigenvalue truncation
        - "auto": Pick the cheapest path based on n_pix vs n_modes (at the
          effective lmax after lswitch). Selects "harmonic" when
          n_pix > n_modes, otherwise "pixel" in direct mode (no V).
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
        (beam, spins, basis, C_ell, epsilon, mode_fraction, fields,
        lswitch_low, lswitch_high, etc.). Arguments not accepted by the
        chosen class are silently ignored.

    Returns
    -------
    ComputationBasis
        Configured basis instance (not yet set up — call .setup()).
    """
    spins = kwargs.get("spins")
    lswitch_high = kwargs.get("lswitch_high")
    n_pix, n_modes = _problem_dimensions(theta, spins, lmax, lswitch_high)

    if method == "auto":
        if n_pix > n_modes:
            method = "harmonic"
        else:
            method = "pixel"
            kwargs["use_direct"] = True
    elif method == "harmonic" and n_pix <= n_modes:
        warnings.warn(
            f"n_pix ({n_pix}) <= n_modes ({n_modes}): harmonic basis expands "
            f"the problem dimension. Consider method='auto' or method='pixel'.",
            UserWarning,
            stacklevel=2,
        )

    if method not in _BASIS_CLASSES:
        raise ValueError(
            f"Unknown computation basis method '{method}'. "
            f"Available: {list(_BASIS_CLASSES) + ['auto']}"
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
    "_problem_dimensions",
    "create_computation_basis",
]
