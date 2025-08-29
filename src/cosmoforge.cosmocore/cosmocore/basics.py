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

Notes
-----
This module is heavily optimized for performance in cosmological analysis:
- Numba @njit decorators for compiled performance
- In-place operations to avoid memory allocations
- Optimized recurrence relations for Legendre polynomials
- Efficient vector operations for spherical geometry
- Robust matrix operations using LAPACK routines
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
        msg = f"Index {idx} out of bounds for nfields={nfields}"
        raise ValueError(msg)
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


def matrix_inverse_symm(M):
    """
    Compute inverse of symmetric positive definite matrix.

    Parameters
    ----------
    M : numpy.ndarray
        2D square symmetric positive definite matrix to invert.

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

    L, info = lapack.dpotrf(M, lower=True, overwrite_a=True, clean=True)
    if info != 0:
        raise ValueError(f"dpotrf failed with info={info}")

    inv_L, info = lapack.dpotri(L, lower=True, overwrite_c=True)
    if info != 0:
        raise ValueError(f"dpotri failed with info={info}")

    return _copy_lower_to_upper(inv_L)
