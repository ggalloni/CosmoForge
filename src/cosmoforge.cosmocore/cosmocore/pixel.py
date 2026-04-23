"""
Pixel-space operations for cosmological analysis.

This module provides optimized functions for pixel-space computations in
cosmological data analysis, including signal matrix calculations, pointing
vector computations, and pixel masking operations. Many functions are
optimized with Numba for performance.

Functions
---------
count_nonzero_mask
    Count non-zero pixels in a mask.
compute_pointings
    Compute 3D pointing vectors for HEALPix pixels.
compute_00_contribution
    Compute spin-0 x spin-0 field contributions to signal matrix.
compute_22_contribution
    Compute spin-2 x spin-2 field contributions to signal matrix.
compute_02_contribution
    Compute spin-0 x spin-2 field contributions to signal matrix.
compute_signal_matrix
    Build the full signal covariance matrix.
derivative_step_00
    Derivative computation for spin-0 x spin-0 correlations.
derivative_step_02
    Derivative computation for spin-0 x spin-2 correlations.
derivative_step_22
    Derivative computation for spin-2 x spin-2 correlations.
do_derivative_step
    Execute derivative step for Fisher matrix calculations.

Notes
-----
This module heavily uses Numba's @njit decorator for performance optimization.
The signal matrix calculations implement the full pixel-space likelihood for
CMB analysis with proper handling of spin-0 (temperature) and spin-2
(polarization) fields.

References
----------
.. [1] Gorski, K.M. et al. "HEALPix: A Framework for High-Resolution Discretization
   and Fast Analysis of Data Distributed on the Sphere"
   Astrophys. J. 622, 759-771 (2005)
.. [2] Kamionkowski, M., Kosowsky, A. & Stebbins, A. "Statistics of cosmic microwave
   background polarization" Phys. Rev. D 55, 7368 (1997)
.. [3] Zaldarriaga, M. & Seljak, U. "All-sky analysis of polarization in the microwave
   background" Phys. Rev. D 55, 1830 (1997)
.. [4] Ng, K.-W. & Liu, G.-C. "Correlation Functions of CMB Anisotropy and Polarization"
   Int. J. Mod. Phys. D 8, 61-83 (1999)
.. [5] Challinor, A. et al. "All-sky convolution for polarimetry experiments"
   Phys. Rev. D 62, 123002 (2000)
"""

import healpy as hp
import numpy as np
from numba import njit, prange

from .basics import (
    get_rotation_angle,
    legendre_00,
    legendre_02,
    legendre_22,
)
from .fields import FieldCollection


@njit(cache=True)
def count_nonzero_mask(mask):
    """
    Count non-zero pixels in a 1D mask array.

    Parameters
    ----------
    mask : numpy.ndarray
        1D array representing pixel mask values.

    Returns
    -------
    int
        Number of pixels with absolute value > 0.5.

    Notes
    -----
    This function is optimized with Numba for performance in tight loops.
    Uses threshold of 0.5 to determine active pixels.
    """
    npix = mask.shape[0]
    npixs = 0
    ntemp = 0
    for j in range(npix):
        if abs(mask[j]) > 0.5:
            ntemp += 1
    npixs = ntemp
    return npixs


def compute_pointings(
    nside, npixs, point_vectors, theta_vectors, phi_vectors, active, ordering
):
    """
    Compute 3D pointing vectors for active HEALPix pixels.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    npixs : list of int
        Number of active pixels for each field.
    point_vectors : tuple of numpy.ndarray
        Tuple of arrays to store pointing vectors for each field.
        Each array has shape (n_active, 3).
    theta_vectors : tuple of numpy.ndarray
        Tuple of arrays to store theta unit vectors for each field.
        Each array has shape (n_active).
    phi_vectors : tuple of numpy.ndarray
        Tuple of arrays to store phi unit vectors for each field.
        Each array has shape (n_active).
    active : numpy.ndarray or tuple
        Active pixel indices for each field.
    ordering : str
        HEALPix pixel ordering: ``"RING"`` or ``"NESTED"``.

    Returns
    -------
    tuple of numpy.ndarray
        Updated pointing vectors with normalized 3D unit vectors.

    Notes
    -----
    Converts HEALPix pixel indices to 3D Cartesian unit vectors pointing
    to pixel centers. Used for spherical harmonic calculations and
    geometric operations.
    """
    nmaps = len(npixs)

    for field_idx in range(nmaps):
        ntemp = npixs[field_idx]

        for i in range(ntemp):
            theta, phi = hp.pix2ang(
                nside, active[field_idx, i], nest=(ordering == "NESTED")
            )
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            norm = np.sqrt(x**2 + y**2 + z**2)
            point_vectors[field_idx][i, 0] = x / norm
            point_vectors[field_idx][i, 1] = y / norm
            point_vectors[field_idx][i, 2] = z / norm
            theta_vectors[field_idx][i] = theta
            phi_vectors[field_idx][i] = phi

    return point_vectors, theta_vectors, phi_vectors


@njit
def compute_00_contribution(cl, S_slice, vec1, vec2, legendre, mode, remove_dipole=False):
    """
    Compute spin-0 x spin-0 field contribution to signal matrix.

    Parameters
    ----------
    cl : numpy.ndarray
        Power spectrum coefficients for the field correlation.
    S_slice : numpy.ndarray
        2D array slice of signal matrix to fill.
    vec1, vec2 : numpy.ndarray
        3D pointing vectors for the two fields, shape (npix, 3).
    legendre : numpy.ndarray
        Temporary array for Legendre polynomial computation.
    mode : int
        Computation mode (0 for symmetric, 1 for general).
    remove_dipole : bool, optional
        Whether to suppress dipole contribution for temperature autocorrelation.

    Notes
    -----
    Implements the pixel-space covariance for scalar fields (e.g., temperature).
    Uses Legendre polynomials to compute angular correlations. The dipole
    removal is applied for temperature-temperature correlations to handle
    coordinate system effects.

    The outer pixel loop uses prange for thread-level parallelism.
    """
    npix = S_slice.shape[0]
    if mode == 0:
        legendre_00(1.0, legendre)
        entry = np.sum(cl * legendre[1:])
        if remove_dipole:
            entry += 1000.0 * cl[0] * 2.0
        for i in range(npix):
            S_slice[i, i] = entry

        _compute_00_symmetric(cl, S_slice, vec1, vec2, npix, remove_dipole)

    elif mode == 1:
        _compute_00_general(cl, S_slice, vec1, vec2, npix)


@njit(parallel=True)
def _compute_00_symmetric(cl, S_slice, vec1, vec2, npix, remove_dipole):
    """Parallel pixel-pair loop for symmetric spin-0 signal matrix."""
    for i in prange(npix):
        leg = np.empty(len(cl) + 1, dtype=np.float64)
        for j in range(i + 1, npix):
            x = (
                vec1[j, 0] * vec2[i, 0]
                + vec1[j, 1] * vec2[i, 1]
                + vec1[j, 2] * vec2[i, 2]
            )
            legendre_00(x, leg)
            val = np.float64(0.0)
            for k in range(len(cl)):
                val += cl[k] * leg[k + 1]
            if remove_dipole:
                val += 1000.0 * cl[0] * (1.0 + x)
            S_slice[j, i] = val


@njit(parallel=True)
def _compute_00_general(cl, S_slice, vec1, vec2, npix):
    """Parallel pixel loop for general spin-0 signal matrix."""
    for i in prange(npix):
        leg = np.empty(len(cl) + 1, dtype=np.float64)
        for j in range(npix):
            x = (
                vec1[j, 0] * vec2[i, 0]
                + vec1[j, 1] * vec2[i, 1]
                + vec1[j, 2] * vec2[i, 2]
            )
            legendre_00(x, leg)
            val = np.float64(0.0)
            for k in range(len(cl)):
                val += cl[k] * leg[k + 1]
            S_slice[j, i] = val


@njit
def compute_22_contribution(cl11, cl22, cl12, S_slice, vec1, vec2, legendre, f1, f2):
    """
    Compute spin-2 x spin-2 field contribution to signal matrix.

    Parameters
    ----------
    cl11, cl22, cl12 : numpy.ndarray
        Power spectra for EE, BB, and EB correlations respectively.
    S_slice : numpy.ndarray
        2D array slice of signal matrix to fill.
    vec1, vec2 : numpy.ndarray
        3D pointing vectors for the two fields, shape (npix, 3).
    legendre : numpy.ndarray
        Temporary array for Legendre polynomial computation.
    f1, f2 : numpy.ndarray
        Temporary arrays for spin-2 harmonic functions.

    Notes
    -----
    Implements the pixel-space covariance for polarization fields with proper
    rotation handling. The outer pixel loop uses prange for parallelism.
    """
    npix = S_slice.shape[0] // 2

    legendre_22(1.0, legendre, f1, f2)

    qq = np.sum(cl11 * f1[1:]) - np.sum(cl22 * f2[1:])
    uu = np.sum(cl22 * f1[1:]) - np.sum(cl11 * f2[1:])
    qu = np.sum((f1[1:] + f2[1:]) * cl12)

    for i in range(npix):
        S_slice[i, i] = qq
        S_slice[i + npix, i] = qu
    for i in range(npix, npix * 2):
        S_slice[i, i] = uu

    _compute_22_symmetric(cl11, cl22, cl12, S_slice, vec1, vec2, npix)


@njit(parallel=True)
def _compute_22_symmetric(cl11, cl22, cl12, S_slice, vec1, vec2, npix):
    """Parallel pixel-pair loop for symmetric spin-2 signal matrix."""
    n_cl = len(cl11)
    for i in prange(npix):
        leg = np.empty(n_cl + 1, dtype=np.float64)
        _f1 = np.empty(n_cl + 1, dtype=np.float64)
        _f2 = np.empty(n_cl + 1, dtype=np.float64)
        for j in range(i + 1, npix):
            x = (
                vec1[j, 0] * vec2[i, 0]
                + vec1[j, 1] * vec2[i, 1]
                + vec1[j, 2] * vec2[i, 2]
            )
            legendre_22(x, leg, _f1, _f2)

            qq = np.float64(0.0)
            uu = np.float64(0.0)
            qu = np.float64(0.0)
            for k in range(n_cl):
                qq += cl11[k] * _f1[k + 1] - cl22[k] * _f2[k + 1]
                uu += cl22[k] * _f1[k + 1] - cl11[k] * _f2[k + 1]
                qu += (_f1[k + 1] + _f2[k + 1]) * cl12[k]

            ang1, ang2 = get_rotation_angle(vec1[j], vec2[i])
            cos1 = np.cos(ang1)
            cos2 = np.cos(ang2)
            sin1 = -np.sin(ang1)
            sin2 = -np.sin(ang2)
            cos1cos2 = cos1 * cos2
            sin1sin2 = sin1 * sin2
            cos1sin2 = cos1 * sin2
            sin1cos2 = sin1 * cos2
            S_slice[j, i] = qq * cos1cos2 + uu * sin1sin2 + qu * (cos1sin2 + sin1cos2)
            S_slice[j + npix, i + npix] = (
                qq * sin1sin2 + uu * cos1cos2 - qu * (cos1sin2 + sin1cos2)
            )
            S_slice[j + npix, i] = (
                -qq * sin1cos2 + uu * cos1sin2 + qu * (cos1cos2 - sin1sin2)
            )
            S_slice[i + npix, j] = (
                -qq * cos1sin2 + uu * sin1cos2 + qu * (cos1cos2 - sin1sin2)
            )


@njit
def compute_02_contribution(cl12, cl13, S_slice, vec0, vec2, legendre):
    """
    Compute spin-0 x spin-2 field contribution to signal matrix.

    Parameters
    ----------
    cl12, cl13 : numpy.ndarray
        Cross-power spectra for T-Q and T-U correlations respectively.
    S_slice : numpy.ndarray
        2D array slice of signal matrix to fill.
    vec0 : numpy.ndarray
        3D pointing vectors for spin-0 field, shape (npix, 3).
    vec2 : numpy.ndarray
        3D pointing vectors for spin-2 field, shape (npix, 3).
    legendre : numpy.ndarray
        Temporary array for Legendre polynomial computation.

    Notes
    -----
    Implements cross-correlations between temperature (spin-0) and polarization
    (spin-2) fields. The outer pixel loop uses prange for parallelism.
    """
    npix_spin0 = vec0.shape[0]
    npix_spin2 = vec2.shape[0]

    _compute_02_parallel(cl12, cl13, S_slice, vec0, vec2, npix_spin0, npix_spin2)


@njit(parallel=True)
def _compute_02_parallel(cl12, cl13, S_slice, vec0, vec2, npix_spin0, npix_spin2):
    """Parallel pixel loop for spin-0 x spin-2 signal matrix."""
    n_cl = len(cl12)
    for i in prange(npix_spin0):
        leg = np.empty(n_cl + 1, dtype=np.float64)
        for j in range(npix_spin2):
            x = (
                vec2[j, 0] * vec0[i, 0]
                + vec2[j, 1] * vec0[i, 1]
                + vec2[j, 2] * vec0[i, 2]
            )
            legendre_02(x, leg)
            tq = np.float64(0.0)
            tu = np.float64(0.0)
            for k in range(n_cl):
                tq -= cl12[k] * leg[k + 1]
                tu -= cl13[k] * leg[k + 1]
            ang1, _ = get_rotation_angle(vec2[j], vec0[i])
            cos1 = np.cos(ang1)
            sin1 = np.sin(ang1)
            S_slice[j, i] = tq * cos1 - tu * sin1
            S_slice[j + npix_spin2, i] = tq * sin1 + tu * cos1


def compute_signal_matrix(
    S,
    lmax,
    fields: FieldCollection,
):
    """
    Build the full signal covariance matrix for cosmological fields.

    Parameters
    ----------
    S : numpy.ndarray
        2D array to store the signal covariance matrix.
    lmax : int
        Maximum multipole for power spectrum computations.
    fields : FieldCollection
        Collection of cosmological fields with power spectra and pointing vectors.

    Notes
    -----
    Constructs the complete pixel-space signal covariance matrix by combining
    contributions from all field pairs. Handles different spin combinations:
    - Spin-0 x Spin-0: Temperature-like correlations
    - Spin-2 x Spin-2: Polarization-like correlations
    - Spin-0 x Spin-2: Temperature-polarization cross-correlations

    The matrix structure accounts for proper field ordering and spin weights.
    Cross-correlations between different spin-2 fields are not yet implemented.

    Raises
    ------
    NotImplementedError
        If cross-correlation between different spin-2 fields is requested.
    """
    spins = fields.spin
    npixs = fields.n_active

    row_offset = 0
    for i, (npix_i, spin_i) in enumerate(zip(npixs, spins)):
        lf_i = fields.fields[i]
        nrow = 2 * npix_i if spin_i == 2 else npix_i
        col_offset = 0
        for j, (npix_j, spin_j) in enumerate(zip(npixs, spins)):
            ncol = 2 * npix_j if spin_j == 2 else npix_j
            if j < i:
                col_offset += ncol
                continue
            lf_j = fields.fields[j]
            legendre = np.empty(lmax, dtype=np.float64)
            if spin_i == 0 and spin_j == 0:
                # FIXME: remove_dipole feature is broken - it uses cl[0] (ell=2)
                # instead of the actual dipole (ell=1). Disabling for now.
                # The original code was:
                # remove_dipole = (
                #     True if lf_i.maps_label + lf_j.maps_label == "TT" else False
                # )
                remove_dipole = False
                if i == j:
                    compute_00_contribution(
                        fields.get_cls(i, j, 0),
                        S[row_offset : row_offset + nrow, col_offset : col_offset + ncol],
                        lf_i.point_vectors[:, :],
                        lf_j.point_vectors[:, :],
                        legendre,
                        mode=0,
                        remove_dipole=remove_dipole,
                    )
                else:
                    compute_00_contribution(
                        fields.get_cls(i, j, 0),
                        S[col_offset : col_offset + ncol, row_offset : row_offset + nrow],
                        lf_j.point_vectors[:, :],
                        lf_i.point_vectors[:, :],
                        legendre,
                        mode=1,
                    )
            elif spin_i == 2 and spin_j == 2:
                if i == j:
                    cl11 = fields.get_cls(i, i, 0)
                    cl22 = fields.get_cls(i, i, 1)
                    cl12 = fields.get_cls(i, i, 2)
                else:
                    raise NotImplementedError(
                        "Cross-correlation for spin-2 fields not implemented yet."
                    )
                f1 = np.empty(lmax, dtype=np.float64)
                f2 = np.empty(lmax, dtype=np.float64)
                compute_22_contribution(
                    cl11,
                    cl22,
                    cl12,
                    S[col_offset : col_offset + ncol, row_offset : row_offset + nrow],
                    lf_i.point_vectors[:, :],
                    lf_j.point_vectors[:, :],
                    legendre,
                    f1,
                    f2,
                )
            elif (spin_i, spin_j) in ((0, 2), (2, 0)):
                cl12 = fields.get_cls(min(i, j), max(i, j), 0)
                cl13 = fields.get_cls(min(i, j), max(i, j), 1)
                if spin_i == 0:
                    compute_02_contribution(
                        cl12,
                        cl13,
                        S[col_offset : col_offset + ncol, row_offset : row_offset + nrow],
                        lf_i.point_vectors[:, :],
                        lf_j.point_vectors[:, :],
                        legendre,
                    )
                else:
                    compute_02_contribution(
                        cl13,
                        cl12,
                        S[row_offset : row_offset + nrow, col_offset : col_offset + ncol],
                        lf_j.point_vectors[:, :],
                        lf_i.point_vectors[:, :],
                        legendre,
                    )
            col_offset += ncol
        row_offset += nrow
    for i in range(S.shape[0]):
        for j in range(i + 1, S.shape[0]):
            S[i, j] = S[j, i]


@njit(cache=True)
def derivative_step_00(S_slice, vec1, vec2, current_ell, legendre, mode):
    """
    Compute derivative step for spin-0 x spin-0 field correlations.

    Parameters
    ----------
    S_slice : numpy.ndarray
        2D array slice to store derivative contributions.
    vec1, vec2 : numpy.ndarray
        3D pointing vectors for the two fields.
    current_ell : int
        Current multipole moment for the derivative calculation.
    legendre : numpy.ndarray
        Temporary array for Legendre polynomial computation.
    mode : int
        Computation mode (0 for symmetric, 1 for general).

    Notes
    -----
    Computes the derivative of the signal matrix with respect to power spectrum
    parameters for scalar field correlations. Used in Fisher matrix calculations.
    """
    npix = S_slice.shape[0]
    if mode == 0:
        legendre_00(1.0, legendre)
        for i in range(npix):
            S_slice[i, i] = legendre[current_ell - 1]

        for i in range(npix):
            for j in range(i + 1, npix):
                legendre_00(vec1[j] @ vec2[i].T, legendre)
                S_slice[j, i] = legendre[current_ell - 1]

        for i in range(S_slice.shape[0]):
            for j in range(i + 1, S_slice.shape[0]):
                S_slice[i, j] = S_slice[j, i]

    elif mode == 1:
        for i in range(npix):
            for j in range(npix):
                legendre_00(vec1[j] @ vec2[i].T, legendre)
                S_slice[j, i] = legendre[current_ell - 1]


@njit(cache=True)
def derivative_step_02(S_slice, vec0, vec2, current_ell, mode, legendre):
    """
    Compute derivative step for spin-0 x spin-2 field correlations.

    Parameters
    ----------
    S_slice : numpy.ndarray
        2D array slice to store derivative contributions.
    vec0 : numpy.ndarray
        3D pointing vectors for the spin-0 field.
    vec2 : numpy.ndarray
        3D pointing vectors for the spin-2 field.
    current_ell : int
        Current multipole moment for the derivative calculation.
    mode : int
        Computation mode (0 for T-Q, 1 for T-U correlations).
    legendre : numpy.ndarray
        Temporary array for Legendre polynomial computation.

    Notes
    -----
    Computes derivatives for temperature-polarization cross-correlations.
    Handles coordinate rotations and mode-dependent sign conventions.
    Used in Fisher matrix calculations for cross-spectrum parameters.
    """
    npix_spin0 = vec0.shape[0]
    npix_spin2 = vec2.shape[0]

    for i in range(npix_spin0):
        for j in range(npix_spin2):
            legendre_02(vec2[j] @ vec0[i].T, legendre)

            ang1, _ = get_rotation_angle(vec2[j], vec0[i])
            cos1 = np.cos(ang1)
            sin1 = np.sin(ang1)

            if mode == 0:
                S_slice[j, i] = -legendre[current_ell - 1] * cos1
                S_slice[j + npix_spin2, i] = -legendre[current_ell - 1] * sin1
            else:
                S_slice[j, i] = legendre[current_ell - 1] * sin1
                S_slice[j + npix_spin2, i] = -legendre[current_ell - 1] * cos1


@njit(cache=True)
def derivative_step_22(S_slice, vec1, vec2, current_ell, mode, legendre, f1, f2):
    """
    Compute derivative step for spin-2 x spin-2 field correlations.

    Parameters
    ----------
    S_slice : numpy.ndarray
        2D array slice to store derivative contributions.
    vec1, vec2 : numpy.ndarray
        3D pointing vectors for the two spin-2 fields.
    current_ell : int
        Current multipole moment for the derivative calculation.
    mode : int
        Polarization mode (0 for EE, 1 for BB, 2 for EB).
    legendre : numpy.ndarray
        Temporary array for Legendre polynomial computation.
    f1, f2 : numpy.ndarray
        Temporary arrays for spin-2 harmonic functions.

    Notes
    -----
    Computes derivatives for polarization-polarization correlations including
    proper E/B mode decomposition and coordinate rotations. Handles the three
    independent polarization power spectra: EE, BB, and EB.
    """
    npix = S_slice.shape[0] // 2

    legendre_22(1.0, legendre, f1, f2)

    if mode == 0:  # such as EE
        qq = f1[current_ell - 1]
        uu = -f2[current_ell - 1]
        qu = 0.0
    elif mode == 1:  # such as BB
        qq = -f2[current_ell - 1]
        uu = f1[current_ell - 1]
        qu = 0.0
    elif mode == 2:  # such as EB
        qq = 0.0
        uu = 0.0
        qu = f1[current_ell - 1] + f2[current_ell - 1]

    for i in range(npix):
        S_slice[i, i] = qq
        S_slice[i + npix, i] = qu
    for i in range(npix, npix * 2):
        S_slice[i, i] = uu

    for i in range(npix):
        for j in range(i + 1, npix):
            legendre_22(vec1[j] @ vec2[i].T, legendre, f1, f2)
            if mode == 0:  # such as EE
                qq = f1[current_ell - 1]
                uu = -f2[current_ell - 1]
                qu = 0.0
            elif mode == 1:  # such as BB
                qq = -f2[current_ell - 1]
                uu = f1[current_ell - 1]
                qu = 0.0
            elif mode == 2:  # such as EB
                qq = 0.0
                uu = 0.0
                qu = f1[current_ell - 1] + f2[current_ell - 1]

            ang1, ang2 = get_rotation_angle(vec1[j], vec2[i])
            c1 = np.cos(ang1)
            c2 = np.cos(ang2)
            s1 = -np.sin(ang1)
            s2 = -np.sin(ang2)
            c1c2 = c1 * c2
            s1s2 = s1 * s2
            c1s2 = c1 * s2
            s1c2 = s1 * c2

            S_slice[j, i] = qq * c1c2 + uu * s1s2 + qu * (c1s2 + s1c2)
            S_slice[j + npix, i + npix] = qq * s1s2 + uu * c1c2 - qu * (c1s2 + s1c2)
            S_slice[j + npix, i] = -qq * s1c2 + uu * c1s2 + qu * (c1c2 - s1s2)
            S_slice[i + npix, j] = -qq * c1s2 + uu * s1c2 + qu * (c1c2 - s1s2)

    for i in range(S_slice.shape[0]):
        for j in range(i + 1, S_slice.shape[0]):
            S_slice[i, j] = S_slice[j, i]


def do_derivative_step(
    S,
    spectrum,
    npixs,
    spins,
    current_ell,
    fields: FieldCollection,
):
    """
    Execute derivative step for Fisher matrix calculations.

    Parameters
    ----------
    S : numpy.ndarray
        2D signal matrix to fill with derivative contributions.
    spectrum : int
        Index of the power spectrum being differentiated.
    npixs : list of int
        Number of active pixels for each field (deprecated, use fields.n_active).
    spins : list of int
        Spin values for each field (deprecated, use fields.spin).
    current_ell : int
        Current multipole moment for the derivative calculation.
    fields : FieldCollection
        Collection of cosmological fields with power spectra and metadata.

    Notes
    -----
    Orchestrates the computation of signal matrix derivatives with respect to
    power spectrum parameters. Determines the appropriate field combinations
    and calls the corresponding derivative functions based on spin values.

    Used in Fisher matrix calculations to compute parameter sensitivities.
    The function handles different field types and cross-correlations automatically.
    """
    spins = fields.spin
    npixs = fields.n_active
    label = fields.spectra_labels[spectrum]

    S.fill(0.0)

    for idx, field in enumerate(fields.fields):
        if label[0] in field.maps_label:
            idx_i = idx
            spin_i = field.spin
            npix_i = field.n_active[0]
            point_vectors_i = field.point_vectors
            break
    for idx, field in enumerate(fields.fields):
        if label[1] in field.maps_label:
            idx_j = idx
            spin_j = field.spin
            npix_j = field.n_active[0]
            point_vectors_j = field.point_vectors
            break

    if spin_i == 0 and spin_j == 0:
        if idx_i == idx_j:
            mode = 0
        else:
            mode = 1
    elif spin_i == 2 and spin_j == 2:
        if label[0] == label[1]:
            mode = np.where(np.array(fields.fields[idx_j].maps_label) == label[0])[0][0]
        else:
            mode = 2
    elif (spin_i, spin_j) in [(0, 2), (2, 0)]:
        combined_labels = [
            a + b
            for a in fields.fields[idx_i].maps_label
            for b in fields.fields[idx_j].maps_label
        ]
        mode = np.where(np.array(combined_labels) == label)[0][0]

    row_offset = sum(2 * n if s == 2 else n for n, s in zip(npixs[:idx_i], spins[:idx_i]))
    col_offset = sum(2 * n if s == 2 else n for n, s in zip(npixs[:idx_j], spins[:idx_j]))
    nrow = 2 * npix_i if spin_i == 2 else npix_i
    ncol = 2 * npix_j if spin_j == 2 else npix_j

    block = S[col_offset : col_offset + ncol, row_offset : row_offset + nrow]
    legendre = np.empty(current_ell)

    if spin_i == 0 and spin_j == 0:
        derivative_step_00(
            block,
            point_vectors_i,
            point_vectors_j,
            current_ell,
            legendre,
            mode,
        )
        if mode == 1:
            S[row_offset : row_offset + nrow, col_offset : col_offset + ncol] = S[
                col_offset : col_offset + ncol, row_offset : row_offset + nrow
            ].T
    elif (spin_i, spin_j) in [(0, 2), (2, 0)]:
        derivative_step_02(
            block,
            point_vectors_i,
            point_vectors_j,
            current_ell,
            mode,
            legendre,
        )
        S[row_offset : row_offset + nrow, col_offset : col_offset + ncol] = S[
            col_offset : col_offset + ncol, row_offset : row_offset + nrow
        ].T
    elif spin_i == 2 and spin_j == 2:
        f1 = np.empty(current_ell)
        f2 = np.empty(current_ell)
        derivative_step_22(
            block,
            point_vectors_i,
            point_vectors_j,
            current_ell,
            mode,
            legendre,
            f1,
            f2,
        )
