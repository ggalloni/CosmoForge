"""Legendre polynomial computations for cosmological analysis."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def legendre_00(scalar_prod, legendre):
    """
    In-place computation of standard Legendre polynomials P_l(x).

    Parameters
    ----------
    scalar_prod : float
        Argument x for the Legendre polynomials.
    legendre : numpy.ndarray
        Pre-allocated array to fill with P_l(x) values.

    Notes
    -----
    Memory-efficient version that avoids allocations in hot loops.
    Uses the standard three-term recurrence relation for Legendre
    polynomials but operates on pre-allocated arrays for maximum performance.
    """
    lmax = len(legendre)
    # Base cases
    legendre[0] = scalar_prod
    legendre[1] = 1.5 * scalar_prod * scalar_prod - 0.5

    # Optimized recurrence
    for ell in range(3, lmax + 1):
        legendre[ell - 1] = (
            (2 * ell - 1) * scalar_prod * legendre[ell - 2]
            - (ell - 1) * legendre[ell - 3]
        ) / ell

    # Absorb (2ℓ+1)/(4π) normalization into the polynomials
    for k in range(lmax):
        legendre[k] *= (2 * (k + 1) + 1) / (4 * np.pi)


@njit(cache=True)
def legendre_22(scalar_prod, legendre, f1, f2):
    """
    In-place computation of associated Legendre polynomials P_l^{22}(x).

    Parameters
    ----------
    scalar_prod : float
        Argument x for the associated Legendre polynomials.
    legendre : numpy.ndarray
        Pre-allocated array to fill with P_l^{22}(x) values.
    f1, f2 : numpy.ndarray
        Pre-allocated temporary arrays for intermediate calculations.

    Notes
    -----
    Memory-efficient version for spin-2 computations. Fills the provided
    arrays in-place to avoid allocations in performance-critical loops.
    Used for polarization correlation calculations.
    """
    lmax = len(legendre)

    # Zero the array and set base case
    legendre[:] = 0.0
    f1[:] = 0.0
    f2[:] = 0.0
    legendre[1] = 3.0
    f1[1] = 6.0 * (1.0 + scalar_prod * scalar_prod)
    f2[1] = -12.0 * scalar_prod

    # Optimized recurrence
    for ell in range(3, lmax + 1):
        legendre[ell - 1] = (
            scalar_prod * (2 * ell - 1) * legendre[ell - 2]
            - (ell + 1) * legendre[ell - 3]
        ) / (ell - 2)
        f1[ell - 1] = (
            -(2 * ell - 8 + ell * (ell - 1) * (1.0 - scalar_prod * scalar_prod))
            * legendre[ell - 1]
            + (2 * ell + 4) * scalar_prod * legendre[ell - 2]
        )
        f2[ell - 1] = 4.0 * (
            -(ell - 1) * scalar_prod * legendre[ell - 1] + (ell + 2) * legendre[ell - 2]
        )

    # Absorb (2ℓ+1)/(4π) × factor2 normalization into the polynomials
    for k in range(1, lmax):
        ell = k + 1
        factor2 = 1.0 / ((ell + 2) * (ell + 1) * ell * (ell - 1))
        norm = (2 * ell + 1) / (4 * np.pi) * factor2
        legendre[k] *= norm
        f1[k] *= norm
        f2[k] *= norm


@njit(cache=True)
def legendre_02(scalar_prod, legendre):
    """
    In-place computation of associated Legendre polynomials P_l^{02}(x).

    Parameters
    ----------
    scalar_prod : float
        Argument x for the associated Legendre polynomials.
    legendre : numpy.ndarray
        Pre-allocated array to fill with P_l^{02}(x) values (will be zeroed first).

    Notes
    -----
    Memory-efficient version that avoids allocations in hot loops.
    Used for temperature-polarization cross-correlations. The array
    is zeroed and then filled using the optimized recurrence relation.
    """
    lmax = len(legendre)

    # Zero the array and set base case
    legendre[:] = 0.0
    legendre[1] = 3.0 * (1.0 - scalar_prod * scalar_prod)

    # Optimized recurrence
    for ell in range(3, lmax + 1):
        legendre[ell - 1] = (
            scalar_prod * (2 * ell - 1) * legendre[ell - 2]
            - (ell + 1) * legendre[ell - 3]
        ) / (ell - 2)

    # Absorb (2ℓ+1)/(4π) × factor normalization into the polynomials
    for k in range(1, lmax):
        ell = k + 1
        factor = np.sqrt(1.0 / ((ell + 2) * (ell + 1) * ell * (ell - 1)))
        legendre[k] *= (2 * ell + 1) / (4 * np.pi) * factor


@njit(cache=True)
def legendre_plm(cos_theta, sin_theta, plm):
    """
    In-place computation of normalized associated Legendre polynomials.

    Computes N_l^m(x) = sqrt((l-m)!/(l+m)!) * P_l^m(x) for all l = 0..lmax
    and m = 0..l, where x = cos(theta).

    Parameters
    ----------
    cos_theta : float
        Cosine of colatitude angle.
    sin_theta : float
        Sine of colatitude angle (must be non-negative).
    plm : numpy.ndarray
        Pre-allocated array of shape (lmax+1, lmax+1) to fill with values.
        On output, plm[ell, m] contains N_l^m for m <= l.
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
