"""Wigner d-matrix computations for spin-weighted spherical harmonics."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def wigner_d_small(ell: int, m: int, s: int, cos_theta: float, sin_theta: float) -> float:
    """
    Compute Wigner d-matrix element d^l_{m,s}(theta) for spin-weighted
    spherical harmonics.

    Uses stable recurrence in l derived from the Jacobi polynomial
    three-term recurrence.

    Parameters
    ----------
    ell : int
        Multipole degree (l >= max(|m|, |s|)).
    m : int
        Azimuthal order (-l <= m <= l).
    s : int
        Spin weight (typically +/-2 for polarization).
    cos_theta : float
        Cosine of colatitude angle.
    sin_theta : float
        Sine of colatitude angle (must be >= 0).

    Returns
    -------
    float
        Value of d^l_{m,s}(theta).
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

    # Starting value: d^m_{m,s}(theta)
    l_start = m

    if l_start < abs(s):
        l_start = abs(s)

    # Compute d^{l_start}_{m,s}
    d_curr = _wigner_d_start(l_start, m, s, cos_half, sin_half)

    if ell == l_start:
        return swap_sign * d_curr

    # Recurrence in l
    d_prev = d_curr
    el = l_start
    l_next = el + 1

    fm_next = np.sqrt(l_next * l_next - m * m)
    fs_next = np.sqrt(l_next * l_next - s * s)
    factor_next = fm_next * fs_next

    if factor_next > 0:
        alpha = (2 * el + 1) * (el + 1) / factor_next
        if el > 0:
            beta = -(2 * el + 1) * m * s / (el * factor_next)
        else:
            beta = 0.0
        d_curr = (alpha * cos_theta + beta) * d_prev
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
    Compute starting value d^l_{m,s}(theta) where l = max(|m|, |s|).

    Uses direct formula with log-factorial to avoid overflow.
    """
    # For l = m = s = 0, return 1
    if ell == 0:
        return 1.0

    # Compute factorial ratio using log to avoid overflow
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

    # Handle negative powers
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

    # Phase factor: (-1)^{m-s}
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
    Compute all d^l_{m,s}(theta) for m = -l to l for a fixed l and s.

    Parameters
    ----------
    ell : int
        Multipole degree.
    s : int
        Spin weight (typically +/-2 for polarization).
    cos_theta : float
        Cosine of colatitude angle.
    sin_theta : float
        Sine of colatitude angle.
    d_out : numpy.ndarray
        Pre-allocated output array of length (2*l+1) to store d^l_{m,s}
        for m = -l, -l+1, ..., l-1, l. Index i corresponds to m = i - l.
    """
    for i in range(2 * ell + 1):
        m = i - ell
        d_out[i] = wigner_d_small(ell, m, s, cos_theta, sin_theta)
