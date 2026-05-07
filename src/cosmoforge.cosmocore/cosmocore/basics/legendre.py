"""Legendre polynomial computations for cosmological analysis."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def legendre_00(scalar_prod, legendre):
    """
    In-place computation of normalized Legendre polynomials (2ℓ+1)/(4π) P_l(x).

    Parameters
    ----------
    scalar_prod : float
        Argument x for the Legendre polynomials.
    legendre : numpy.ndarray
        Pre-allocated buffer of length ``lmax + 1``. On output, ``legendre[ell]``
        contains the normalized P_ℓ(x) for ℓ = 1..lmax. Index 0 is left at zero
        (P_0 will be added in the lmin extension).

    Notes
    -----
    Computes the recurrence for P_l(x) and absorbs the (2ℓ+1)/(4π)
    normalization factor, so callers receive the fully normalized basis
    functions ready for signal matrix construction.
    """
    lmax = len(legendre) - 1
    legendre[0] = 0.0
    if lmax >= 1:
        legendre[1] = scalar_prod
    if lmax >= 2:
        legendre[2] = 1.5 * scalar_prod * scalar_prod - 0.5

    # Optimized recurrence
    for ell in range(3, lmax + 1):
        legendre[ell] = (
            (2 * ell - 1) * scalar_prod * legendre[ell - 1]
            - (ell - 1) * legendre[ell - 2]
        ) / ell

    # Absorb (2ℓ+1)/(4π) normalization into the polynomials
    for ell in range(1, lmax + 1):
        legendre[ell] *= (2 * ell + 1) / (4 * np.pi)


@njit(cache=True)
def legendre_22(scalar_prod, legendre, f1, f2):
    """
    In-place computation of normalized spin-2 Legendre functions.

    Computes (2ℓ+1)/(4π) × factor2 × P_l^{22}(x) and related f1, f2 arrays,
    where factor2 = 1/((ℓ+2)(ℓ+1)ℓ(ℓ-1)).

    Parameters
    ----------
    scalar_prod : float
        Argument x for the associated Legendre polynomials.
    legendre : numpy.ndarray
        Pre-allocated array to fill with normalized values.
    f1, f2 : numpy.ndarray
        Pre-allocated arrays for the EE/BB decomposition factors.

    Notes
    -----
    Memory-efficient version for spin-2 computations. Fills the provided
    arrays in-place to avoid allocations in performance-critical loops.
    Used for polarization auto-correlation calculations. Buffers are
    ℓ-indexed: ``legendre[ell]``, ``f1[ell]``, ``f2[ell]`` correspond to
    multipole ℓ. Indices 0 and 1 are zero by spin-2 physics.
    """
    lmax = len(legendre) - 1

    # Zero the arrays and set base case at ℓ=2
    legendre[:] = 0.0
    f1[:] = 0.0
    f2[:] = 0.0
    if lmax >= 2:
        legendre[2] = 3.0
        f1[2] = 6.0 * (1.0 + scalar_prod * scalar_prod)
        f2[2] = -12.0 * scalar_prod

    # Optimized recurrence
    for ell in range(3, lmax + 1):
        legendre[ell] = (
            scalar_prod * (2 * ell - 1) * legendre[ell - 1]
            - (ell + 1) * legendre[ell - 2]
        ) / (ell - 2)
        f1[ell] = (
            -(2 * ell - 8 + ell * (ell - 1) * (1.0 - scalar_prod * scalar_prod))
            * legendre[ell]
            + (2 * ell + 4) * scalar_prod * legendre[ell - 1]
        )
        f2[ell] = 4.0 * (
            -(ell - 1) * scalar_prod * legendre[ell] + (ell + 2) * legendre[ell - 1]
        )

    # Absorb (2ℓ+1)/(4π) × factor2 normalization into the polynomials
    for ell in range(2, lmax + 1):
        factor2 = 1.0 / ((ell + 2) * (ell + 1) * ell * (ell - 1))
        norm = (2 * ell + 1) / (4 * np.pi) * factor2
        legendre[ell] *= norm
        f1[ell] *= norm
        f2[ell] *= norm


@njit(cache=True)
def legendre_02(scalar_prod, legendre):
    """
    In-place computation of normalized spin-0×2 Legendre functions.

    Computes (2ℓ+1)/(4π) × factor × P_l^{02}(x), where
    factor = 1/sqrt((ℓ+2)(ℓ+1)ℓ(ℓ-1)).

    Parameters
    ----------
    scalar_prod : float
        Argument x for the associated Legendre polynomials.
    legendre : numpy.ndarray
        Pre-allocated array to fill with normalized values (will be zeroed first).

    Notes
    -----
    Memory-efficient version that avoids allocations in hot loops.
    Used for temperature-polarization cross-correlations. Buffer is
    ℓ-indexed: ``legendre[ell]`` corresponds to multipole ℓ. Indices 0 and
    1 are zero (spin-0×spin-2 requires ℓ ≥ 2).
    """
    lmax = len(legendre) - 1

    # Zero the array and set base case at ℓ=2
    legendre[:] = 0.0
    if lmax >= 2:
        legendre[2] = 3.0 * (1.0 - scalar_prod * scalar_prod)

    # Optimized recurrence
    for ell in range(3, lmax + 1):
        legendre[ell] = (
            scalar_prod * (2 * ell - 1) * legendre[ell - 1]
            - (ell + 1) * legendre[ell - 2]
        ) / (ell - 2)

    # Absorb (2ℓ+1)/(4π) × factor normalization into the polynomials
    for ell in range(2, lmax + 1):
        factor = np.sqrt(1.0 / ((ell + 2) * (ell + 1) * ell * (ell - 1)))
        legendre[ell] *= (2 * ell + 1) / (4 * np.pi) * factor


@njit(cache=True)
def legendre_plm(cos_theta, sin_theta, plm):
    """
    In-place computation of fully normalized associated Legendre polynomials.

    Computes sqrt((2ℓ+1)/(4π)) × N_l^m(x) for all l = 0..lmax and m = 0..l,
    where N_l^m = sqrt((l-m)!/(l+m)!) × P_l^m(x) and x = cos(theta).

    Parameters
    ----------
    cos_theta : float
        Cosine of colatitude angle.
    sin_theta : float
        Sine of colatitude angle (must be non-negative).
    plm : numpy.ndarray
        Pre-allocated array of shape (lmax+1, lmax+1) to fill with values.
        On output, plm[ell, m] contains the fully normalized value for m <= l.
        Values for m > l are set to zero.
    """
    lmax = plm.shape[0] - 1
    x = cos_theta
    s = sin_theta

    # Initialize to zero
    plm[:, :] = 0.0

    # m = 0 column: standard Legendre polynomials P_l(x)
    # (normalization factor is 1 for m = 0)
    plm[0, 0] = 1.0
    if lmax >= 1:
        plm[1, 0] = x
    for ell in range(2, lmax + 1):
        plm[ell, 0] = (
            (2 * ell - 1) * x * plm[ell - 1, 0] - (ell - 1) * plm[ell - 2, 0]
        ) / ell

    # m > 0: use sectoral and l-recurrence
    for m in range(1, lmax + 1):
        # Sectoral: N_m^m = -sin(theta) * sqrt((2m-1)/(2m)) * N_{m-1}^{m-1}
        plm[m, m] = -s * np.sqrt((2 * m - 1) / (2 * m)) * plm[m - 1, m - 1]

        # Semi-sectoral: N_{m+1}^m = cos(theta) * sqrt(2m+1) * N_m^m
        if m < lmax:
            plm[m + 1, m] = x * np.sqrt(2 * m + 1) * plm[m, m]

        # l-recurrence for l > m + 1
        for ell in range(m + 2, lmax + 1):
            ell2_m2 = ell * ell - m * m
            ell1_2_m2 = (ell - 1) ** 2 - m * m
            a = (2 * ell - 1) / np.sqrt(ell2_m2)
            b = np.sqrt(ell1_2_m2 / ell2_m2)
            plm[ell, m] = a * x * plm[ell - 1, m] - b * plm[ell - 2, m]

    # Absorb sqrt((2ℓ+1)/(4π)) normalization into the polynomials
    for ell in range(lmax + 1):
        scale = np.sqrt((2 * ell + 1) / (4 * np.pi))
        for m in range(ell + 1):
            plm[ell, m] *= scale
