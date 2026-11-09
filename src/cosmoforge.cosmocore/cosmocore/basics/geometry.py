"""Spherical geometry utilities for rotation angle computation."""

from __future__ import annotations

import numpy as np
from numba import njit

# Module-level constants (one allocation, reused)
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
    by taking the cross product with the z-axis (0,0,1).
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
    systems on the sphere.

    Each angle is ``-2 A``, with ``A`` the angle from the local ``e_phi`` to the
    normal ``r12 = r1 x r2 / |r1 x r2|`` in the tangent plane, taken by
    ``atan2`` of its ``e_theta`` and ``e_phi`` components. An ``arccos`` of the
    ``e_phi`` component alone loses half the digits where ``A`` is 0 or pi, which
    is every pair of pixels on a common meridian: the sine of the angle came out
    at up to 5e-8 instead of 0.
    """
    # cross-product r1 x r2 -> r12
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

    a12 = -2.0 * _tangent_angle(r1, r12x, r12y, r12z)
    # the normal seen from r2 is r21 = -r12
    a21 = -2.0 * _tangent_angle(r2, -r12x, -r12y, -r12z)

    return a12, a21


@njit(cache=True)
def _tangent_angle(r, nx, ny, nz):
    """Signed angle from ``e_phi`` at ``r`` to the unit tangent vector ``n``."""
    # e_phi, bumped at a pole by _project_and_norm
    ex, ey, _ = _project_and_norm(r[0], r[1], r[2])
    # e_theta = e_phi x r
    tx = ey * r[2]
    ty = -ex * r[2]
    tz = ex * r[1] - ey * r[0]
    return np.arctan2(nx * tx + ny * ty + nz * tz, nx * ex + ny * ey)
