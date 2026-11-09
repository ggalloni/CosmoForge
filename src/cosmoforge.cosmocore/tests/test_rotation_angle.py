"""The spin-2 rotation angle and the pixel-space derivative kernels that use it.

The kernels are checked against healpy: ``dS/dC_ell`` is the covariance of maps made
from unit alms, summed over m with each alm's variance (C for m = 0, C/2 for the real
and imaginary parts of m > 0).
"""

import healpy as hp
import numpy as np
import pytest

from cosmocore import compute_pointings, get_rotation_angle
from cosmocore.signal_kernels import derivative_step_02, derivative_step_22

NSIDE = 4


def _pointings(nside):
    return compute_pointings(nside, [np.arange(hp.nside2npix(nside))], "RING")[0][0]


def test_rotation_angle_is_exact_on_a_common_meridian():
    """Two pixels on one meridian (phi equal, or apart by pi) are joined by it, so
    both angles are 0 or 2 pi and their sines vanish. An arccos of the dot product
    put up to 5e-8 there, with a sign set by rounding. The bound leaves room for the
    pointings themselves, which sit about 1e-15 rad off their meridian: on pairs
    close to antipodal that moves the angle by up to 2e-14 (the median is 2e-16)."""
    nside = 8
    vec = _pointings(nside)
    phi = hp.pix2ang(nside, np.arange(len(vec)))[1]
    worst = 0.0
    for i in range(len(vec)):
        dphi = (phi - phi[i]) % np.pi
        for j in np.flatnonzero(np.minimum(dphi, np.pi - dphi) < 1e-12):
            if j > i:
                a12, a21 = get_rotation_angle(vec[j], vec[i])
                worst = max(worst, abs(np.sin(a12)), abs(np.sin(a21)))
    assert worst < 1e-13


def _unit_alm_maps(nside, ell, m, value):
    """(T, QU) maps of a unit alm at (ell, m) in each of T, E, B."""
    maps = {}
    for k, name in enumerate("TEB"):
        alm = np.zeros((3, hp.Alm.getsize(ell)), dtype=complex)
        alm[k, hp.Alm.getidx(ell, ell, m)] = value
        t, q, u = hp.alm2map(alm, nside, lmax=ell, pol=True)
        maps[name] = t if name == "T" else np.concatenate([q, u])
    return maps


def _healpy_derivative(nside, ell, a, b):
    """Cov(x_a, x_b) for C_ell^{ab} = 1, with x_T the T map and x_E, x_B the QU maps."""
    out = 0.0
    for m in range(ell + 1):
        weights = ((1.0, 1.0), (1j, 0.0)) if m == 0 else ((1.0, 0.5), (1j, 0.5))
        for value, weight in weights:
            if weight:
                maps = _unit_alm_maps(nside, ell, m, value)
                cross = np.outer(maps[a], maps[b])
                out = out + weight * (cross if a == b or a == "T" else cross + cross.T)
    return out


@pytest.mark.parametrize("ell", [2, 5, 8])
@pytest.mark.parametrize("mode, a, b", [(0, "E", "E"), (1, "B", "B"), (2, "E", "B")])
def test_spin2_derivative_matches_healpy(ell, mode, a, b):
    vec = _pointings(NSIDE)
    n = len(vec)
    got = np.zeros((2 * n, 2 * n))
    buffers = [np.empty(ell + 1) for _ in range(3)]
    derivative_step_22(got, vec, vec, ell, mode, *buffers)
    want = _healpy_derivative(NSIDE, ell, a, b)
    assert np.abs(got - want).max() < 1e-12 * np.abs(want).max()


@pytest.mark.parametrize("ell", [2, 5, 8])
def test_spin0_spin2_derivative_matches_healpy(ell):
    vec = _pointings(NSIDE)
    n = len(vec)
    got = np.zeros((2 * n, n))
    derivative_step_02(got, vec, vec, ell, 0, np.empty(ell + 1))
    # the kernel fills the (QU, T) block of dS/dC_ell^TE
    want = _healpy_derivative(NSIDE, ell, "T", "E").T
    assert np.abs(got - want).max() < 1e-12 * np.abs(want).max()
