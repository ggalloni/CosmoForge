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

    r1sx, r1sy, _ = _project_and_norm(r1[0], r1[1], r1[2])
    r2sx, r2sy, _ = _project_and_norm(r2[0], r2[1], r2[2])

    # dot r12*r1star and decide sign from r12*zz
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

    # dot r12*r2star
    dot2 = r12x * r2sx + r12y * r2sy
    if dot2 > 1.0:
        dot2 = 1.0
    elif dot2 < -1.0:
        dot2 = -1.0
    sign2 = 1.0 if r12z > 0.0 else -1.0
    a21 = 2.0 * np.arccos(dot2) * sign2

    return a12, a21
