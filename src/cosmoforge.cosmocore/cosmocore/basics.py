"""
Basic mathematical utilities for cosmological computations.

This module provides fundamental mathematical functions optimized for
cosmological analysis, including Legendre polynomials, vector operations,
rotation calculations, and matrix operations. Many functions are optimized
with Numba for performance in numerical calculations.

Functions
---------
legendre_00
    In-place computation of standard Legendre polynomials.
legendre_22
    In-place computation of spin-2 associated Legendre polynomials.
legendre_02
    In-place computation of spin-0 to spin-2 associated Legendre polynomials.
legendre_plm
    In-place computation of normalized associated Legendre polynomials P_ℓ^m.
spec2idx
    Convert field indices to spectrum index for compressed storage.
idx2spec
    Convert spectrum index back to field indices.
get_rotation_angle
    Compute rotation angles between vectors on the sphere.
matrix_mult
    Matrix multiplication wrapper.
matrix_trace
    Compute trace of matrix product A @ B.
matrix_inverse_symm
    Compute inverse of symmetric positive definite matrix.
matrix_slogdet
    Compute sign and logarithm of determinant using LU decomposition.
matrix_slogdet_symm
    Compute sign and logarithm of determinant for symmetric positive definite matrices.

Notes
-----
This module is heavily optimized for performance in cosmological analysis:
- Numba @njit decorators for compiled performance
- In-place operations to avoid memory allocations
- Optimized recurrence relations for Legendre polynomials
- Efficient vector operations for spherical geometry
- Robust matrix operations using LAPACK routines

References
----------
Legendre Polynomials and Spin-Weighted Harmonics:
.. [1] Abramowitz, M. & Stegun, I.A. "Handbook of Mathematical Functions"
   Dover Publications (1972) - Chapter 8: Legendre Functions
.. [2] Newman, E.T. & Penrose, R. "Note on the Bondi-Metzner-Sachs Group"
   J. Math. Phys. 7, 863 (1966) - Spin-weighted spherical harmonics
.. [3] Goldberg, J.N. et al. "Spin-s Spherical Harmonics and edth"
   J. Math. Phys. 8, 2155 (1967)
.. [4] Zaldarriaga, M. & Seljak, U. "All-sky analysis of polarization in the
   microwave background" Phys. Rev. D 55, 1830 (1997)

Numerical Methods:
.. [5] Anderson, E. et al. "LAPACK Users' Guide" SIAM, Philadelphia (1999)
   - Cholesky decomposition (dpotrf/dpotri), LU factorization (dgetrf)
.. [6] Golub, G.H. & Van Loan, C.F. "Matrix Computations" Johns Hopkins
   University Press (2013) - Numerical stability of matrix algorithms
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit
from scipy.linalg import lapack


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
    if lmax == 2:
        return

    # Optimized recurrence
    for ell in range(3, lmax + 1):
        legendre[ell - 1] = (
            (2 * ell - 1) * scalar_prod * legendre[ell - 2]
            - (ell - 1) * legendre[ell - 3]
        ) / ell


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
    if lmax == 2:
        return

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
    if lmax == 2:
        return

    # Optimized recurrence
    for ell in range(3, lmax + 1):
        legendre[ell - 1] = (
            scalar_prod * (2 * ell - 1) * legendre[ell - 2]
            - (ell + 1) * legendre[ell - 3]
        ) / (ell - 2)


@njit(cache=True)
def legendre_plm(cos_theta, sin_theta, plm):
    """
    In-place computation of normalized associated Legendre polynomials.

    Computes N_ℓ^m(x) = sqrt((ℓ-m)!/(ℓ+m)!) × P_ℓ^m(x) for all ℓ = 0..lmax
    and m = 0..ℓ, where x = cos(θ). This normalization is used in real
    spherical harmonics for CMB analysis.

    Parameters
    ----------
    cos_theta : float
        Cosine of colatitude angle.
    sin_theta : float
        Sine of colatitude angle (must be non-negative).
    plm : numpy.ndarray
        Pre-allocated array of shape (lmax+1, lmax+1) to fill with values.
        On output, plm[ell, m] contains N_ℓ^m for m <= ℓ.
        Values for m > ℓ are set to zero.

    Notes
    -----
    Uses stable recurrence relations to avoid factorial overflow:

    - Sectoral (m = ℓ): N_m^m = -sin(θ) × sqrt((2m-1)/(2m)) × N_{m-1}^{m-1}
    - Semi-sectoral (ℓ = m+1): N_{m+1}^m = cos(θ) × sqrt(2m+1) × N_m^m
    - General ℓ-recurrence: N_ℓ^m = a × cos(θ) × N_{ℓ-1}^m - b × N_{ℓ-2}^m
      where a = (2ℓ-1)/sqrt(ℓ²-m²), b = sqrt((ℓ-1)²-m²)/sqrt(ℓ²-m²)

    The m = 0 column uses the standard Legendre recurrence (same as legendre_00).

    References
    ----------
    .. [1] Press, W.H. et al. "Numerical Recipes" Cambridge University Press
       - Section on associated Legendre functions
    .. [2] Holmes, S.A. & Featherstone, W.E. "A unified approach to the Clenshaw
       summation and the recursive computation of very high degree and order
       normalised associated Legendre functions" J. Geodesy 76, 279-299 (2002)
    """
    lmax = plm.shape[0] - 1
    x = cos_theta
    s = sin_theta

    # Initialize to zero
    plm[:, :] = 0.0

    # m = 0 column: standard Legendre polynomials P_ℓ(x)
    # (normalization factor is 1 for m = 0)
    plm[0, 0] = 1.0
    if lmax >= 1:
        plm[1, 0] = x
    for ell in range(2, lmax + 1):
        plm[ell, 0] = (
            (2 * ell - 1) * x * plm[ell - 1, 0] - (ell - 1) * plm[ell - 2, 0]
        ) / ell

    # m > 0: use sectoral and ℓ-recurrence
    for m in range(1, lmax + 1):
        # Sectoral: N_m^m = -sin(θ) × sqrt((2m-1)/(2m)) × N_{m-1}^{m-1}
        plm[m, m] = -s * np.sqrt((2 * m - 1) / (2 * m)) * plm[m - 1, m - 1]

        # Semi-sectoral: N_{m+1}^m = cos(θ) × sqrt(2m+1) × N_m^m
        if m < lmax:
            plm[m + 1, m] = x * np.sqrt(2 * m + 1) * plm[m, m]

        # ℓ-recurrence for ℓ > m + 1
        for ell in range(m + 2, lmax + 1):
            ell2_m2 = ell * ell - m * m
            ell1_2_m2 = (ell - 1) ** 2 - m * m
            a = (2 * ell - 1) / np.sqrt(ell2_m2)
            b = np.sqrt(ell1_2_m2 / ell2_m2)
            plm[ell, m] = a * x * plm[ell - 1, m] - b * plm[ell - 2, m]


@njit(cache=True)
def wigner_d_small(ell: int, m: int, s: int, cos_theta: float, sin_theta: float) -> float:
    """
    Compute Wigner d-matrix element d^ℓ_{m,s}(θ) for spin-weighted spherical harmonics.

    Uses stable recurrence in ℓ derived from the Jacobi polynomial
    three-term recurrence (DLMF 18.9.2):
        d^{ℓ+1}_{m,s} = (α_ℓ cos θ + β_ℓ) d^ℓ_{m,s} + γ_ℓ d^{ℓ-1}_{m,s}
    with coefficients:
        α_ℓ = (2ℓ+1)(ℓ+1) / [√((ℓ+1)²-m²) √((ℓ+1)²-s²)]
        β_ℓ = -(2ℓ+1)ms / [ℓ √((ℓ+1)²-m²) √((ℓ+1)²-s²)]
        γ_ℓ = -(ℓ+1)/ℓ × √(ℓ²-m²)√(ℓ²-s²) / [√((ℓ+1)²-m²)√((ℓ+1)²-s²)]

    Parameters
    ----------
    ell : int
        Multipole degree (ℓ ≥ max(|m|, |s|)).
    m : int
        Azimuthal order (-ℓ ≤ m ≤ ℓ).
    s : int
        Spin weight (typically ±2 for polarization).
    cos_theta : float
        Cosine of colatitude angle.
    sin_theta : float
        Sine of colatitude angle (must be ≥ 0).

    Returns
    -------
    float
        Value of d^ℓ_{m,s}(θ).

    References
    ----------
    .. [1] Goldberg, J.N. et al. "Spin-s Spherical Harmonics and ð"
       J. Math. Phys. 8, 2155 (1967)
    .. [2] Risbo, T. "Fourier transform summation of Legendre series and
       D-functions" J. Geodesy 70, 383-396 (1996)
    """
    # Handle trivial cases
    if ell < abs(m) or ell < abs(s):
        return 0.0

    # Use symmetry: d^l_{m,s} = (-1)^{m-s} d^l_{s,m} = (-1)^{m-s} d^l_{-m,-s}
    # This allows us to always have |m| >= |s| for the recurrence
    swap_sign = 1.0
    if abs(m) < abs(s):
        # Swap m and s
        m, s = s, m
        if (m - s) % 2 != 0:
            swap_sign = -1.0

    # Ensure m >= 0 using d^l_{m,s} = (-1)^{m-s} d^l_{-m,-s}
    if m < 0:
        if (m - s) % 2 != 0:
            swap_sign *= -1.0
        m = -m
        s = -s

    # Now we have |m| >= |s| and m >= 0
    # Compute half-angles
    half_theta = 0.5 * np.arccos(cos_theta)
    cos_half = np.cos(half_theta)
    sin_half = np.sin(half_theta)

    # Starting value: d^m_{m,s}(θ)
    # d^m_{m,s} = sqrt((2m)! / ((m+s)!(m-s)!)) * cos^{m+s}(θ/2) * sin^{m-s}(θ/2)
    #           × (-1)^{m-s} if s < 0 convention
    l_start = m

    if l_start < abs(s):
        l_start = abs(s)

    # Compute d^{l_start}_{m,s}
    # Using direct formula for the starting value
    d_curr = _wigner_d_start(l_start, m, s, cos_half, sin_half)

    if ell == l_start:
        return swap_sign * d_curr

    # Recurrence in ℓ derived from the Jacobi polynomial three-term recurrence:
    #   d^{l+1}_{m,s} = (α_l cos θ + β_l) d^l_{m,s} + γ_l d^{l-1}_{m,s}
    # where:
    #   α_l = (2l+1)(l+1) / [√((l+1)²-m²) √((l+1)²-s²)]
    #   β_l = -(2l+1)ms / [l √((l+1)²-m²) √((l+1)²-s²)]
    #   γ_l = -(l+1)/l × √(l²-m²)√(l²-s²) / [√((l+1)²-m²)√((l+1)²-s²)]

    # One-step recurrence to get d^{l_start+1}
    d_prev = d_curr
    el = l_start
    l_next = el + 1

    fm_next = np.sqrt(l_next * l_next - m * m)
    fs_next = np.sqrt(l_next * l_next - s * s)
    factor_next = fm_next * fs_next

    if factor_next > 0:
        alpha = (2 * el + 1) * (el + 1) / factor_next
        # beta = -(2l+1)*m*s / (l * factor_next), but l could be 0
        # When l=0, m=s=0 (since l_start=max(|m|,|s|)=0), so m*s=0 and beta=0
        if el > 0:
            beta = -(2 * el + 1) * m * s / (el * factor_next)
        else:
            beta = 0.0
        d_curr = (alpha * cos_theta + beta) * d_prev
        # gamma term is 0 since d^{l_start-1} doesn't exist
    else:
        d_curr = 0.0

    if ell == l_start + 1:
        return swap_sign * d_curr

    # Full recurrence for higher l
    for el in range(l_start + 1, ell):
        l_next = el + 1
        fm_l = np.sqrt(el * el - m * m)
        fs_l = np.sqrt(el * el - s * s)
        fm_next = np.sqrt(l_next * l_next - m * m)
        fs_next = np.sqrt(l_next * l_next - s * s)
        factor_next = fm_next * fs_next

        if factor_next > 0:
            alpha = (2 * el + 1) * (el + 1) / factor_next
            beta = -(2 * el + 1) * m * s / (el * factor_next)
            gamma = -(el + 1) / el * fm_l * fs_l / factor_next
            d_new = (alpha * cos_theta + beta) * d_curr + gamma * d_prev
        else:
            d_new = 0.0

        d_prev = d_curr
        d_curr = d_new

    return swap_sign * d_curr


@njit(cache=True)
def _wigner_d_start(ell: int, m: int, s: int, cos_half: float, sin_half: float) -> float:
    """
    Compute starting value d^ℓ_{m,s}(θ) where ℓ = max(|m|, |s|).

    Uses direct formula:
        d^ℓ_{m,s} = ε × √[(ℓ+m)!(ℓ-m)!/((ℓ+s)!(ℓ-s)!)]
                   × cos^{m+s}(θ/2) × sin^{m-s}(θ/2)

    where ε is a phase factor.
    """
    # For ℓ = m = s = 0, return 1
    if ell == 0:
        return 1.0

    # Compute factorial ratio using log to avoid overflow
    # sqrt[(ℓ+m)!(ℓ-m)!/((ℓ+s)!(ℓ-s)!)]
    log_ratio = 0.0
    for k in range(1, ell + m + 1):
        log_ratio += np.log(k)
    for k in range(1, ell - m + 1):
        log_ratio += np.log(k)
    for k in range(1, ell + s + 1):
        log_ratio -= np.log(k)
    for k in range(1, ell - s + 1):
        log_ratio -= np.log(k)

    sqrt_ratio = np.exp(0.5 * log_ratio)

    # Powers of cos and sin of half-angle
    power_cos = m + s
    power_sin = m - s

    # Handle negative powers (shouldn't happen if ℓ = max(|m|, |s|) and m >= 0)
    cos_term = 1.0
    sin_term = 1.0

    if power_cos > 0:
        cos_term = cos_half**power_cos
    elif power_cos < 0:
        if abs(cos_half) < 1e-15:
            return 0.0
        cos_term = cos_half**power_cos

    if power_sin > 0:
        sin_term = sin_half**power_sin
    elif power_sin < 0:
        if abs(sin_half) < 1e-15:
            return 0.0
        sin_term = sin_half**power_sin

    # Phase factor: (-1)^{m-s} from the standard Wigner d convention
    phase = 1.0
    if (m - s) % 2 != 0:
        phase = -1.0

    result = phase * sqrt_ratio * cos_term * sin_term

    return result


@njit(cache=True)
def wigner_d_matrix(
    ell: int, s: int, cos_theta: float, sin_theta: float, d_out: np.ndarray
) -> None:
    """
    Compute all d^ℓ_{m,s}(θ) for m = -ℓ to ℓ for a fixed ℓ and s.

    This is more efficient than calling wigner_d_small repeatedly
    when all m values are needed.

    Parameters
    ----------
    ell : int
        Multipole degree.
    s : int
        Spin weight (typically ±2 for polarization).
    cos_theta : float
        Cosine of colatitude angle.
    sin_theta : float
        Sine of colatitude angle.
    d_out : numpy.ndarray
        Pre-allocated output array of length (2*ℓ+1) to store d^ℓ_{m,s}
        for m = -ℓ, -ℓ+1, ..., ℓ-1, ℓ. Index i corresponds to m = i - ℓ.

    Notes
    -----
    Uses recurrence in m for efficiency.
    """
    # Simple implementation: call wigner_d_small for each m
    # Could be optimized with m-recurrence
    for i in range(2 * ell + 1):
        m = i - ell
        d_out[i] = wigner_d_small(ell, m, s, cos_theta, sin_theta)


@lru_cache
def spec2idx(i, j, nfields):
    """
    Convert field indices to spectrum index for compressed storage.

    Parameters
    ----------
    i, j : int
        Field indices for the spectrum.
    nfields : int
        Total number of fields.

    Returns
    -------
    int
        Linear index for spectrum storage in compressed format.

    Notes
    -----
    Converts 2D field indices to 1D spectrum indices for efficient storage.
    Auto-spectra (i==j) are stored first, followed by cross-spectra in
    upper triangular order. Uses LRU cache for performance.
    """
    if i == j:
        return i  # auto
    elif i < j:
        return nfields + (i * (2 * nfields - i - 1)) // 2 + (j - i - 1)
    else:
        return spec2idx(j, i, nfields)


@lru_cache
def idx2spec(idx, nfields):
    """
    Convert spectrum index back to field indices.

    Parameters
    ----------
    idx : int
        Linear spectrum index in compressed storage.
    nfields : int
        Total number of fields.

    Returns
    -------
    tuple of int
        Field indices (i, j) corresponding to the spectrum index.

    Raises
    ------
    ValueError
        If index is out of bounds for the given number of fields.

    Notes
    -----
    Inverse operation of spec2idx. Converts linear spectrum indices back
    to the original field pair indices. Uses LRU cache for performance.
    """
    if idx < nfields:
        return idx, idx
    idx_cross = idx - nfields
    total_cross = nfields * (nfields - 1) // 2
    if idx_cross < 0 or idx_cross >= total_cross:
        raise ValueError(f"Index {idx} out of bounds for nfields={nfields}")
    i = 0
    while idx_cross >= nfields - i - 1:
        idx_cross -= nfields - i - 1
        i += 1
    j = i + idx_cross + 1
    return i, j


# Module‐level constants (one allocation, reused)
_eps = np.pi / 180.0 / 3600.0 / 100.0
_zzx, _zzy, _zzz = 0.0, 0.0, 1.0
_ex_eps_x, _ex_eps_y, _ex_eps_z = _eps, 0.0, 0.0


@njit(cache=True)
def _project_and_norm(vx, vy, vz):
    """
    Project vector onto tangent plane and normalize.

    Parameters
    ----------
    vx, vy, vz : float
        Components of the 3D vector to project.

    Returns
    -------
    tuple of float
        Normalized projected vector components (px, py, pz).

    Notes
    -----
    Projects the input vector onto the tangent plane at the north pole
    by taking the cross product with the z-axis (0,0,1). If the projection
    is too small (nearly parallel to z), adds a small epsilon perturbation
    in the x-direction to ensure numerical stability.
    """
    # cross with zz = (0,0,1)
    px = _zzy * vz - _zzz * vy  # 0*vz - 1*vy = -vy
    py = _zzz * vx - _zzx * vz  # 1*vx - 0*vz = vx
    pz = 0.0
    nm = np.sqrt(px * px + py * py)
    if nm < 1e-8:
        # bump by epsilon in x
        px += _ex_eps_x
        py += _ex_eps_y
        pz += _ex_eps_z
        nm = np.sqrt(px * px + py * py)
    return px / nm, py / nm, pz


@njit(cache=True)
def get_rotation_angle(r1, r2):
    """
    Compute rotation angles between two 3D vectors on the sphere.

    Parameters
    ----------
    r1, r2 : numpy.ndarray
        3D unit vectors pointing to different positions on the sphere.

    Returns
    -------
    tuple of float
        Two rotation angles (a12, a21) needed for coordinate transformations
        between the local coordinate systems at the two positions.

    Notes
    -----
    This function computes the rotation angles needed for transforming
    spin-2 quantities (like polarization) between different coordinate
    systems on the sphere. Essential for proper handling of polarization
    correlations in curved geometry.
    """
    # cross-product r1 × r2 → r12
    r12x = r1[1] * r2[2] - r1[2] * r2[1]
    r12y = r1[2] * r2[0] - r1[0] * r2[2]
    r12z = r1[0] * r2[1] - r1[1] * r2[0]
    # norm
    mod = np.sqrt(r12x * r12x + r12y * r12y + r12z * r12z)
    if mod < 1e-8:
        return 0.0, 0.0
    # normalize
    r12x /= mod
    r12y /= mod
    r12z /= mod

    r1sx, r1sy, _ = _project_and_norm(r1[0], r1[1], r1[2])
    r2sx, r2sy, _ = _project_and_norm(r2[0], r2[1], r2[2])

    # dot r12·r1star  and decide sign from r12·zz
    dot1 = r12x * r1sx + r12y * r1sy
    # clamp
    if dot1 > 1.0:
        dot1 = 1.0
    elif dot1 < -1.0:
        dot1 = -1.0
    sign1 = 1.0 if r12z > 0.0 else -1.0
    a12 = 2.0 * np.arccos(dot1) * sign1

    # now flip r12 for the second angle
    r12x = -r12x
    r12y = -r12y
    r12z = -r12z

    # dot r12·r2star
    dot2 = r12x * r2sx + r12y * r2sy
    if dot2 > 1.0:
        dot2 = 1.0
    elif dot2 < -1.0:
        dot2 = -1.0
    sign2 = 1.0 if r12z > 0.0 else -1.0
    a21 = 2.0 * np.arccos(dot2) * sign2

    return a12, a21


def matrix_mult(A, B):
    """
    Matrix multiplication wrapper.

    Parameters
    ----------
    A, B : numpy.ndarray
        2D matrices to multiply.

    Returns
    -------
    numpy.ndarray
        Result of matrix multiplication A @ B.

    Notes
    -----
    Simple wrapper around NumPy's matmul for consistency with
    other matrix operations in the module.
    """
    return np.matmul(A, B)


@njit(cache=True)
def add_diagonal(M, d, out=None):
    """
    Add diagonal vector to matrix: result = M + diag(d).

    Parameters
    ----------
    M : numpy.ndarray
        2D square matrix.
    d : numpy.ndarray
        1D vector to add to diagonal.
    out : numpy.ndarray, optional
        Output array. If None, a new array is created.
        If provided, must have same shape as M.

    Returns
    -------
    numpy.ndarray
        M + diag(d), computed without creating full diagonal matrix.

    Notes
    -----
    More efficient than M + np.diag(d) as it avoids creating an
    intermediate n×n diagonal matrix.
    """
    if out is None:
        out = M.copy()
    elif out is not M:
        out[:] = M
    n = M.shape[0]
    # Explicit loop works with both C and F ordered arrays in numba
    for i in range(n):
        out[i, i] += d[i]
    return out


@njit(cache=True)
def matrix_trace(A, B):
    """
    Compute trace of matrix product A @ B.

    Parameters
    ----------
    A, B : numpy.ndarray
        2D square matrices of the same size.

    Returns
    -------
    float
        Trace of the matrix product: tr(A @ B) = Σ_i Σ_j A_ij B_ji.

    Notes
    -----
    Optimized computation that avoids explicitly forming the matrix
    product. Computes tr(AB) = Σ_i Σ_j A_ij B_ji directly for better
    performance and memory efficiency.
    """
    n = A.shape[0]
    s = 0.0
    for i in range(n):
        for j in range(n):
            s += A[i, j] * B[j, i]
    return s


@njit
def _copy_lower_to_upper(M):
    """
    Copy lower triangular part to upper triangular part of matrix.

    Parameters
    ----------
    M : numpy.ndarray
        2D square matrix to symmetrize.

    Returns
    -------
    numpy.ndarray
        Symmetric matrix with upper part copied from lower part.

    Notes
    -----
    In-place operation that makes a matrix symmetric by copying
    the lower triangular elements to the corresponding upper
    triangular positions. Used after Cholesky decomposition
    operations to reconstruct full symmetric matrices.
    """
    n = M.shape[0]
    for i in range(n):
        for j in range(i):
            M[j, i] = M[i, j]
    return M


def matrix_inverse_symm(M, overwrite=False):
    """
    Compute inverse of symmetric positive definite matrix.

    Parameters
    ----------
    M : numpy.ndarray
        2D square symmetric positive definite matrix to invert.
    overwrite : bool, optional
        If True, the input matrix M may be overwritten with intermediate
        results for better performance. If False (default), a copy is made
        to preserve the input.

    Returns
    -------
    numpy.ndarray
        Inverse of the input matrix.

    Raises
    ------
    ValueError
        If matrix is not square or if Cholesky decomposition fails.

    Notes
    -----
    Uses LAPACK's dpotrf (Cholesky decomposition) and dpotri (inverse)
    for efficient and numerically stable inversion of symmetric positive
    definite matrices. Essential for covariance matrix operations.
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix must be square")

    if not overwrite:
        M = np.asfortranarray(M.copy())

    L, info = lapack.dpotrf(M, lower=True, overwrite_a=True, clean=True)
    if info != 0:
        raise ValueError(f"dpotrf failed with info={info}")

    inv_L, info = lapack.dpotri(L, lower=True, overwrite_c=True)
    if info != 0:
        raise ValueError(f"dpotri failed with info={info}")

    return _copy_lower_to_upper(inv_L)


def smw_inverse(N_inv, V_N_inv, V_Ninv_VT, Lambda_diag, threshold=1e-30):
    """
    Sherman-Morrison-Woodbury inverse: (N + V^T Λ V)^{-1}.

    Computes the inverse efficiently when the matrix has the form
    N + V^T Λ V, where N is n×n (with known inverse), V is k×n,
    and Λ is k×k diagonal.

    Parameters
    ----------
    N_inv : numpy.ndarray
        Precomputed inverse of N, shape (n, n).
    V_N_inv : numpy.ndarray
        Precomputed V @ N^{-1}, shape (k, n).
    V_Ninv_VT : numpy.ndarray
        Precomputed V @ N^{-1} @ V^T, shape (k, k).
    Lambda_diag : numpy.ndarray
        Diagonal of Λ, shape (k,). This is the only quantity that
        typically varies between calls.
    threshold : float, optional
        Minimum value for Lambda diagonal elements. Values below this
        are treated as threshold to avoid division by zero. Default is 1e-30.

    Returns
    -------
    numpy.ndarray
        The inverse (N + V^T Λ V)^{-1}, shape (n, n).

    Notes
    -----
    The SMW formula:
        (N + V^T Λ V)^{-1} = N^{-1} - (V N^{-1})^T @ K^{-1} @ (V N^{-1})

    where K = Λ^{-1} + V N^{-1} V^T is a k×k matrix.

    Precompute V_N_inv and V_Ninv_VT once, then call this function
    repeatedly with different Lambda_diag values.

    Computational cost: O(nk² + k³) instead of O(n³).
    """
    Lambda_inv_diag = np.where(
        Lambda_diag > threshold, 1.0 / Lambda_diag, 1.0 / threshold
    )
    K = np.asfortranarray(V_Ninv_VT.copy())
    n = K.shape[0]
    K.flat[:: n + 1] += Lambda_inv_diag
    K_inv = matrix_inverse_symm(K, overwrite=True)
    return N_inv - V_N_inv.T @ K_inv @ V_N_inv


def smw_logdet(log_det_N, V_Ninv_VT, Lambda_diag, threshold=1e-30):
    """
    Sherman-Morrison-Woodbury log determinant: log|N + V^T Λ V|.

    Computes the log determinant efficiently using the identity:
        log|N + V^T Λ V| = log|N| + log|Λ| + log|Λ^{-1} + V N^{-1} V^T|

    Parameters
    ----------
    log_det_N : float
        Precomputed log|N|.
    V_Ninv_VT : numpy.ndarray
        Precomputed V @ N^{-1} @ V^T, shape (k, k).
    Lambda_diag : numpy.ndarray
        Diagonal of Λ, shape (k,).
    threshold : float, optional
        Minimum value for Lambda diagonal elements. Default is 1e-30.

    Returns
    -------
    float
        The log determinant log|N + V^T Λ V|.

    Notes
    -----
    Precompute log_det_N and V_Ninv_VT once, then call this function
    repeatedly with different Lambda_diag values.

    Computational cost: O(k³) instead of O(n³).
    """
    log_det_Lambda = np.sum(np.log(np.maximum(Lambda_diag, threshold)))
    Lambda_inv_diag = np.where(
        Lambda_diag > threshold, 1.0 / Lambda_diag, 1.0 / threshold
    )
    K = np.asfortranarray(V_Ninv_VT.copy())
    n = K.shape[0]
    K.flat[:: n + 1] += Lambda_inv_diag
    _, log_det_K = matrix_slogdet_symm(K)
    return log_det_N + log_det_Lambda + log_det_K


def smw_kernel(V_Ninv_VT, Lambda_diag, threshold=1e-30):
    """
    Build the SMW kernel matrix K = Λ^{-1} + V N^{-1} V^T.

    This is the central matrix that appears in all SMW formulas.
    Building it separately allows reuse across different SMW operations.

    Parameters
    ----------
    V_Ninv_VT : numpy.ndarray
        Precomputed V @ N^{-1} @ V^T, shape (k, k).
    Lambda_diag : numpy.ndarray
        Diagonal of Λ, shape (k,).
    threshold : float, optional
        Minimum value for Lambda diagonal elements. Default is 1e-30.

    Returns
    -------
    numpy.ndarray
        The kernel matrix K = Λ^{-1} + V N^{-1} V^T, shape (k, k).
        Returned in Fortran order for efficient LAPACK operations.
    """
    Lambda_inv_diag = np.where(
        Lambda_diag > threshold, 1.0 / Lambda_diag, 1.0 / threshold
    )
    K = np.asfortranarray(V_Ninv_VT.copy())
    n = K.shape[0]
    K.flat[:: n + 1] += Lambda_inv_diag
    return K


def smw_quadratic_form(data, N_inv, V_N_inv, V_Ninv_VT, Lambda_diag, threshold=1e-30):
    """
    SMW quadratic form: d^T (N + V^T Λ V)^{-1} d.

    Computes the quadratic form efficiently for likelihood evaluation:
        χ² = d^T C^{-1} d = d^T N^{-1} d - y^T K^{-1} y

    where y = V N^{-1} d and K = Λ^{-1} + V N^{-1} V^T.

    Parameters
    ----------
    data : numpy.ndarray
        Data vector d, shape (n,).
    N_inv : numpy.ndarray
        Precomputed inverse of N, shape (n, n).
    V_N_inv : numpy.ndarray
        Precomputed V @ N^{-1}, shape (k, n).
    V_Ninv_VT : numpy.ndarray
        Precomputed V @ N^{-1} @ V^T, shape (k, k).
    Lambda_diag : numpy.ndarray
        Diagonal of Λ, shape (k,).
    threshold : float, optional
        Minimum value for Lambda diagonal elements. Default is 1e-30.

    Returns
    -------
    float
        The quadratic form d^T C^{-1} d.

    Notes
    -----
    Computational cost: O(nk + k³) instead of O(n³).

    This is particularly useful for Gaussian likelihood computation where
    log L = -0.5 * (d^T C^{-1} d + log|C| + n*log(2π))
    """
    # Term 1: d^T @ N^{-1} @ d
    term1 = float(matrix_mult(data.T, matrix_mult(N_inv, data)))

    # y = V @ N^{-1} @ d
    y = matrix_mult(V_N_inv, data)

    # Build and solve with K
    K = smw_kernel(V_Ninv_VT, Lambda_diag, threshold)
    L = cholesky_decomposition(K)
    K_inv_y = lapack.dpotrs(L, y, lower=True)[0]

    # Term 2: y^T @ K^{-1} @ y
    term2 = float(matrix_mult(y.T, K_inv_y))

    return term1 - term2


def matrix_slogdet(M):
    """
    Compute sign and logarithm of the determinant of a matrix.

    Parameters
    ----------
    M : numpy.ndarray
        2D square matrix for which to compute the signed log determinant.

    Returns
    -------
    tuple of float
        (sign, logdet) where sign is ±1 and logdet is log(|det(M)|).
        If det(M) = 0, returns (0.0, -inf).

    Raises
    ------
    ValueError
        If matrix is not square or if LU decomposition fails.

    Notes
    -----
    Uses LAPACK's dgetrf (LU decomposition with partial pivoting) for
    numerically stable computation of the determinant. More robust than
    direct determinant computation, especially for large matrices.

    For symmetric positive definite matrices, considers using the
    Cholesky-based version matrix_slogdet_symm for better performance.

    Examples
    --------
    >>> import numpy as np
    >>> M = np.array([[2.0, 1.0], [1.0, 2.0]])
    >>> sign, logdet = matrix_slogdet(M)
    >>> det_value = sign * np.exp(logdet)  # Recover determinant
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix must be square")

    # Use LU decomposition for general matrices
    lu, piv, info = lapack.dgetrf(M, overwrite_a=False)
    if info != 0:
        raise ValueError(f"dgetrf failed with info={info}")

    # Compute log determinant from diagonal of U
    logdet = 0.0
    sign = 1.0

    n = M.shape[0]
    for i in range(n):
        diag_val = lu[i, i]
        if abs(diag_val) < 1e-15:  # Singular matrix
            return 0.0, -np.inf
        if diag_val < 0:
            sign *= -1.0
        logdet += np.log(abs(diag_val))

    # Account for permutations in pivoting
    for i in range(n):
        if piv[i] != i + 1:  # LAPACK uses 1-based indexing
            sign *= -1.0

    return sign, logdet


def cholesky_decomposition(M):
    """
    Perform Cholesky decomposition of a symmetric positive definite matrix.

    Parameters
    ----------
    M : numpy.ndarray
        2D square symmetric positive definite matrix to decompose.

    Returns
    -------
    numpy.ndarray
        Lower triangular matrix L such that M = L @ L.T.

    Raises
    ------
    ValueError
        If matrix is not square or if Cholesky decomposition fails.

    Notes
    -----
    Uses LAPACK's dpotrf for efficient Cholesky decomposition.
    The input matrix must be symmetric positive definite.
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix must be square")

    L, info = lapack.dpotrf(M, lower=True, overwrite_a=False, clean=True)
    if info != 0:
        raise ValueError(f"dpotrf failed with info={info}")

    return L


def matrix_slogdet_symm(M):
    """
    Compute sign and logarithm of determinant for symmetric positive definite matrix.

    Parameters
    ----------
    M : numpy.ndarray
        2D square symmetric positive definite matrix.

    Returns
    -------
    tuple of float
        (sign, logdet) where sign is +1 and logdet is log(det(M)).
        For positive definite matrices, sign is always +1.

    Raises
    ------
    ValueError
        If matrix is not square or if Cholesky decomposition fails.

    Notes
    -----
    Uses LAPACK's dpotrf (Cholesky decomposition) for efficient computation.
    For symmetric positive definite matrices, this is more efficient than
    the general LU-based approach. Since the matrix is positive definite,
    the determinant is always positive (sign = +1).

    The determinant of a Cholesky factorization M = L @ L.T is:
    det(M) = det(L)^2 = (∏ L_ii)^2

    Examples
    --------
    >>> import numpy as np
    >>> M = np.array([[4.0, 2.0], [2.0, 3.0]])  # Positive definite
    >>> sign, logdet = matrix_slogdet_symm(M)
    >>> # sign will be 1.0, logdet = log(det(M))
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix must be square")

    # Cholesky decomposition
    L = cholesky_decomposition(M)

    # For Cholesky L @ L.T, det(M) = det(L)^2 = (∏ L_ii)^2
    # So log(det(M)) = 2 * log(∏ L_ii) = 2 * Σ log(L_ii)
    logdet = 0.0
    n = M.shape[0]
    for i in range(n):
        diag_val = L[i, i]
        if diag_val <= 0:  # Should not happen for positive definite
            raise ValueError("Non-positive diagonal element in Cholesky factor")
        logdet += np.log(diag_val)

    # Factor of 2 because det(M) = det(L)^2
    logdet *= 2.0

    # Sign is always +1 for positive definite matrices
    return 1.0, logdet
