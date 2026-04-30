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


def _auto_pick_method(
    n_pix: int, n_modes: int, lmax: int, n_bins: int
) -> tuple[str, dict, dict]:
    """Pick the cheaper basis based on a leading-order cost model.

    Two paths:

    - **Harmonic + sparse traces**: dominated by the SMW kernel Cholesky,
      cost ~ n_modes^3. Trace stage is O(lmax^4) with the sparse
      derivative representation and is essentially free in comparison.
    - **Pixel-direct**: cost ~ (n_bins + 1) * n_pix^3. The per-bin
      C_inv @ dC^b products are the bottleneck; n_bins=1 is the
      ratio-only case and binned analyses with small n_bins favour pixel.

    Returns (method, extra_kwargs, costs_dict) so callers can log the
    decision without re-deriving it.
    """
    cost_harmonic = float(n_modes) ** 3
    cost_pixel = float(n_bins + 1) * float(n_pix) ** 3
    costs = {
        "n_pix": n_pix,
        "n_modes": n_modes,
        "n_bins": n_bins,
        "cost_harmonic": cost_harmonic,
        "cost_pixel": cost_pixel,
    }
    if cost_harmonic <= cost_pixel:
        return "harmonic", {}, costs
    return "pixel", {"use_direct": True}, costs


def create_computation_basis(
    method: str,
    N: np.ndarray,
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
        - "auto": Pick the cheaper path using a leading-order cost model.
          Compares ~n_modes^3 (harmonic + sparse traces) vs
          ~(n_bins+1)*n_pix^3 (pixel-direct). Pass ``n_bins`` to refine
          the estimate; defaults to ``lmax-1`` (unbinned, worst case
          for pixel-direct).
    N : numpy.ndarray
        Noise covariance matrix. The basis takes ownership; the caller
        should drop its reference after construction.
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
    n_bins = kwargs.pop("n_bins", None)
    if n_bins is None:
        # Worst case for pixel-direct: assume one bandpower per multipole.
        n_bins = max(lmax - 1, 1)

    from ..logger import get_logger

    logger = get_logger("basis", feedback_level=4)

    if method == "auto":
        method, extra, costs = _auto_pick_method(n_pix, n_modes, lmax, n_bins)
        kwargs.update(extra)
        ratio = costs["cost_pixel"] / max(costs["cost_harmonic"], 1.0)
        # Auto-selection diagnostic: debug-level so the default Core flow
        # (now defaults to method="auto") doesn't spam stderr on every run.
        logger.debug(
            f"basis 'auto' picked '{method}' "
            f"(n_pix={n_pix}, n_modes={n_modes}, n_bins={n_bins}; "
            f"cost_pixel/cost_harmonic={ratio:.2g})"
        )
    elif method == "harmonic":
        # Sparse traces make harmonic competitive even when n_pix < n_modes.
        # Only warn when the cost model also disagrees with the user's
        # explicit choice — a real action item, not a default-path notice.
        _, _, costs = _auto_pick_method(n_pix, n_modes, lmax, n_bins)
        if costs["cost_pixel"] < costs["cost_harmonic"]:
            warnings.warn(
                f"harmonic basis chosen but pixel-direct estimated "
                f"~{costs['cost_harmonic'] / costs['cost_pixel']:.1f}x cheaper "
                f"(n_pix={n_pix}, n_modes={n_modes}, n_bins={n_bins}). "
                "Consider method='auto'.",
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
    return cls(N, theta, phi, lmax, **filtered)


__all__ = [
    "ComputationBasis",
    "COMPRESSION_BASES",
    "HarmonicBasis",
    "PixelBasis",
    "SMWPrepared",
    "_problem_dimensions",
    "create_computation_basis",
]
