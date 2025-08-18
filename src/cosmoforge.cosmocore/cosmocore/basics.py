from functools import lru_cache

import numpy as np
from numba import njit
from scipy.linalg import lapack


@lru_cache
def cross_index(i, j, nfields):
    if i > j:
        i, j = j, i
    return i * nfields + j - ((i + 2) * (i + 1)) // 2


@njit(cache=True)
def ext_prod(vec1, vec2):
    vec3 = np.empty(3, dtype=np.float64)
    vec3[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    vec3[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    vec3[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    return vec3


@njit(cache=True)
def scalar_prod(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]


@njit(cache=True)
def legendre_00(scalar_prod, lmax):
    """Optimized P_l(x) Legendre polynomials for l=0 case.

    Uses three-term recurrence: P_l(x) = ((2l-1)xP_{l-1}(x) - (l-1)P_{l-2}(x))/l
    """
    if lmax <= 0:
        return np.empty(0, dtype=np.float64)

    legendre = np.empty(lmax, dtype=np.float64)

    # Base cases
    legendre[0] = scalar_prod
    if lmax == 1:
        return legendre

    legendre[1] = 1.5 * scalar_prod * scalar_prod - 0.5
    if lmax == 2:
        return legendre

    # Optimized recurrence: avoid division in loop
    x = scalar_prod
    p_prev2 = legendre[0]  # P_{l-2}
    p_prev1 = legendre[1]  # P_{l-1}

    for ell in range(3, lmax + 1):
        # P_l = ((2l-1)xP_{l-1} - (l-1)P_{l-2})/l
        p_curr = ((2 * ell - 1) * x * p_prev1 - (ell - 1) * p_prev2) / ell
        legendre[ell - 1] = p_curr
        p_prev2 = p_prev1
        p_prev1 = p_curr

    return legendre


@njit(cache=True)
def legendre_22(scalar_prod, lmax):
    """Optimized P_l^{22}(x) associated Legendre polynomials for spin-2.

    Uses modified recurrence for the 22 case.
    """
    if lmax <= 1:
        return np.zeros(max(0, lmax), dtype=np.float64)

    legendre = np.zeros(lmax, dtype=np.float64)

    # Base case for l=2
    legendre[1] = 3.0
    if lmax == 2:
        return legendre

    # Optimized recurrence: avoid division in loop, cache scalar_prod
    x = scalar_prod
    p_prev2 = 0.0  # P_{l-2}^{22}
    p_prev1 = legendre[1]  # P_{l-1}^{22}

    for ell in range(3, lmax + 1):
        # Modified recurrence for 22 case
        p_curr = (x * (2 * ell - 1) * p_prev1 - (ell + 1) * p_prev2) / (ell - 2)
        legendre[ell - 1] = p_curr
        p_prev2 = p_prev1
        p_prev1 = p_curr

    return legendre


@njit(cache=True)
def legendre_00_inplace(scalar_prod, legendre):
    """In-place version of legendre_00 to avoid allocations.

    Args:
        scalar_prod: The argument x for P_l(x)
        legendre: Pre-allocated array to fill with results
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
def legendre_22_inplace(scalar_prod, legendre, f1, f2):
    """In-place version of legendre_22 to avoid allocations.

    Args:
        scalar_prod: The argument x for P_l^{22}(x)
        legendre: Pre-allocated array to fill with results (will be zeroed first)
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
def legendre_02(scalar_prod, lmax):
    """Optimized P_l^{02}(x) associated Legendre polynomials for spin-0 to spin-2.

    Uses modified recurrence: P_l^{02} = (x(2l-1)P_{l-1}^{02} - (l+1)P_{l-2}^{02})/(l-2)
    Base case: P_2^{02} = 3(1 - x^2)
    """
    if lmax <= 1:
        return np.zeros(max(0, lmax), dtype=np.float64)

    legendre = np.zeros(lmax, dtype=np.float64)

    # Base case for l=2: P_2^{02} = 3(1 - x^2)
    legendre[1] = 3.0 * (1.0 - scalar_prod * scalar_prod)
    if lmax == 2:
        return legendre

    # Optimized recurrence: avoid division in loop, cache scalar_prod
    x = scalar_prod
    p_prev2 = 0.0  # P_{l-2}^{02}
    p_prev1 = legendre[1]  # P_{l-1}^{02}

    for ell in range(3, lmax + 1):
        # Modified recurrence for 02 case: (x(2l-1)P_{l-1} - (l+1)P_{l-2})/(l-2)
        p_curr = (x * (2 * ell - 1) * p_prev1 - (ell + 1) * p_prev2) / (ell - 2)
        legendre[ell - 1] = p_curr
        p_prev2 = p_prev1
        p_prev1 = p_curr

    return legendre


@njit(cache=True)
def legendre_02_inplace(scalar_prod, legendre):
    """In-place version of legendre_02 to avoid allocations.

    Args:
        scalar_prod: The argument x for P_l^{02}(x)
        legendre: Pre-allocated array to fill with results (will be zeroed first)
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
def legendre_unified(scalar_prod, lmax, spin_case):
    """Unified Legendre polynomial computation for all spin cases.

    Args:
        scalar_prod: The argument x
        lmax: Maximum l value
        spin_case: 0 for P_l (00 case), 1 for P_l^{22} (22 case), 2 for P_l^{02} (02 case)

    Returns:
        Array of Legendre polynomial values

    This unified function reduces code duplication and allows for better optimization
    by factorizing common operations.
    """
    if lmax <= 0:
        return np.empty(0, dtype=np.float64)

    legendre = np.zeros(lmax, dtype=np.float64)
    x = scalar_prod
    x2 = x * x

    # Set base cases based on spin case
    if spin_case == "00":  # P_l case (00)
        if lmax >= 1:
            legendre[0] = x
        if lmax >= 2:
            legendre[1] = 1.5 * x2 - 0.5
    elif spin_case == "22":  # P_l^{22} case (22)
        if lmax >= 2:
            legendre[1] = 3.0
    elif spin_case == "02":  # P_l^{02} case (02)
        if lmax >= 2:
            legendre[1] = 3.0 * (1.0 - x2)

    if lmax == 2:
        return legendre

    # Unified recurrence computation
    # All cases follow pattern: P_l = (a*x*P_{l-1} - b*P_{l-2}) / c
    for ell in range(3, lmax + 1):
        if spin_case == "00":  # Standard Legendre: ((2l-1)xP_{l-1} - (l-1)P_{l-2})/l
            a, b, c = 2 * ell - 1, ell - 1, ell
        elif spin_case == "22":  # P_l^{22}: (x(2l-1)P_{l-1} - (l+1)P_{l-2})/(l-2)
            a, b, c = 2 * ell - 1, ell + 1, ell - 2
        elif spin_case == "02":  # P_l^{02}: (x(2l-1)P_{l-1} - (l+1)P_{l-2})/(l-2)
            a, b, c = 2 * ell - 1, ell + 1, ell - 2

        legendre[ell - 1] = (a * x * legendre[ell - 2] - b * legendre[ell - 3]) / c

    return legendre


@njit(cache=True)
def legendre_unified_inplace(scalar_prod, legendre, spin_case):
    """In-place unified Legendre polynomial computation.

    This is the most efficient version for hot loops - avoids all allocations
    and unifies the three recurrence relations.
    """
    lmax = len(legendre)
    if lmax <= 0:
        return

    # Clear array
    legendre[:] = 0.0

    x = scalar_prod
    x2 = x * x

    # Set base cases
    if spin_case == "00":  # P_l case (00)
        if lmax >= 1:
            legendre[0] = x
        if lmax >= 2:
            legendre[1] = 1.5 * x2 - 0.5
    elif spin_case == "22":  # P_l^{22} case (22)
        if lmax >= 2:
            legendre[1] = 3.0
    elif spin_case == "02":  # P_l^{02} case (02)
        if lmax >= 2:
            legendre[1] = 3.0 * (1.0 - x2)

    if lmax == 2:
        return

    # Unified recurrence - factorized for maximum efficiency
    for ell in range(3, lmax + 1):
        coeff_2l_minus_1 = 2 * ell - 1

        if spin_case == "00":  # ((2l-1)xP_{l-1} - (l-1)P_{l-2})/l
            numerator = (
                coeff_2l_minus_1 * x * legendre[ell - 2] - (ell - 1) * legendre[ell - 3]
            )
            legendre[ell - 1] = numerator / ell
        elif spin_case == "22":  # (x(2l-1)P_{l-1} - (l+1)P_{l-2})/(l-2)
            numerator = (
                coeff_2l_minus_1 * x * legendre[ell - 2] - (ell + 1) * legendre[ell - 3]
            )
            legendre[ell - 1] = numerator / (ell - 2)
        elif spin_case == "02":  # (x(2l-1)P_{l-1} - (l+1)P_{l-2})/(l-2)
            numerator = (
                coeff_2l_minus_1 * x * legendre[ell - 2] - (ell + 1) * legendre[ell - 3]
            )
            legendre[ell - 1] = numerator / (ell - 2)


@lru_cache
def spec2idx(i, j, nfields):
    if i == j:
        return i  # auto
    elif i < j:
        return nfields + (i * (2 * nfields - i - 1)) // 2 + (j - i - 1)
    else:
        return spec2idx(j, i, nfields)


@lru_cache
def idx2spec(idx, nfields):
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
def project_and_norm(vx, vy, vz):
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

    r1sx, r1sy, _ = project_and_norm(r1[0], r1[1], r1[2])
    r2sx, r2sy, _ = project_and_norm(r2[0], r2[1], r2[2])

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
    return np.matmul(A, B)


@njit(cache=True)
def matrix_trace(A, B):
    n = A.shape[0]
    s = 0.0
    for i in range(n):
        for j in range(n):
            s += A[i, j] * B[j, i]
    return s


# Basic Cholesky factorization (lower triangular)
@njit(cache=True)
def cholesky_lower(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 0.0:
                    msg = "Matrix is not positive definite"
                    raise ValueError(msg)
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


# Inverse of lower triangular matrix L
@njit(cache=True)
def invert_lower_triangular(L):
    n = L.shape[0]
    Linv = np.zeros_like(L)
    for i in range(n):
        Linv[i, i] = 1.0 / L[i, i]
        for j in range(i):
            s = 0.0
            for k in range(j, i):
                s -= L[i, k] * Linv[k, j]
            Linv[i, j] = s / L[i, i]
    return Linv


@njit
def copy_lower_to_upper(M):
    n = M.shape[0]
    for i in range(n):
        for j in range(i):
            M[j, i] = M[i, j]
    return M


def matrix_inverse_symm(M):
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix must be square")

    L, info = lapack.dpotrf(M, lower=True, overwrite_a=True, clean=True)
    if info != 0:
        raise ValueError(f"dpotrf failed with info={info}")

    inv_L, info = lapack.dpotri(L, lower=True, overwrite_c=True)
    if info != 0:
        raise ValueError(f"dpotri failed with info={info}")

    return copy_lower_to_upper(inv_L)
