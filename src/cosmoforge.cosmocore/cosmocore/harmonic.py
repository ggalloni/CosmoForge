import healpy as hp
import numpy as np
from numba import njit


@njit(cache=True)
def cl_to_vec(cl, vec):
    lmax = cl.shape[0] + 1
    n_spec = cl.shape[1]
    counter = 0
    for ispec in range(n_spec):
        for il in range(2, lmax):
            vec[counter] = cl[il - 2, ispec]
            counter += 1


@njit(cache=True)
def vec_to_cl(vec, cl):
    lmax = cl.shape[0] + 1
    n_spec = cl.shape[1]
    counter = 0
    for ispec in range(n_spec):
        for il in range(2, lmax):
            cl[il - 2, ispec] = vec[counter]
            counter += 1


def coswinbeam(nside):
    L = 4 * nside + 1
    beam = np.zeros(L, dtype=np.float64)
    # flat top
    beam[: nside + 1] = 1.0
    # cosine roll-off
    ell = np.arange(nside + 1, 3 * nside + 1)
    beam[nside + 1 : 3 * nside + 1] = 0.5 * (
        1.0 + np.cos((ell - nside) * np.pi / (2.0 * nside))
    )
    return beam


def compute_beam(lmax, nside, smoothtype, fwhmarcmin, beam_file):
    if smoothtype == 0:
        beam = np.ones((lmax - 1, 3), dtype=np.float64)
    elif smoothtype == 1:
        # fwhmarcmin in arcminutes → fwhm_rad
        beam = np.array(
            hp.gauss_beam(np.deg2rad(fwhmarcmin / 60.0), lmax=lmax + 1, pol=True)[
                2 : lmax + 1
            ],
            dtype=np.float64,
        ).T
    elif smoothtype == 2:
        b = coswinbeam(nside)[2 : lmax + 1]
        beam = np.column_stack([b] * 3).T
    elif smoothtype == 3:
        # assume beam_file contains three columns of ell-window
        # healpy.read_cl returns a tuple of arrays when multiple fields
        bls = hp.read_cl(beam_file.strip()).astype(np.float64)
        beam = np.column_stack([bls[i][2 : lmax + 1] for i in range(bls.shape[0])]).T
    else:
        msg = f"Unknown smoothtype={smoothtype}"
        raise ValueError(msg)
    if beam.shape[0] != 3 or beam.shape[1] != lmax - 1:
        msg = f"Beam shape mismatch: expected (3, {lmax - 1}), got {beam.shape}"
        raise ValueError(msg)

    return {
        "T": beam[0, :],
        "E": beam[1, :],
        "B": beam[2, :],
    }
