import healpy as hp
import numpy as np
from numba import njit

from .basics import (
    get_rotation_angle,
    legendre_00_inplace,
    legendre_02_inplace,
    legendre_22_inplace,
)
from .fields import FieldCollection


@njit(cache=True)
def count_nonzero_mask(mask):
    npix = mask.shape
    npixs = 0
    ntemp = 0
    for j in range(npix):
        if abs(mask[j]) > 0.5:
            ntemp += 1
    npixs = ntemp
    return npixs


@njit(cache=True)
def pixel_active(mask):
    npix, nmaps = mask.shape
    count = 0
    for j in range(nmaps):
        for i in range(npix):
            if abs(mask[i, j]) > 0.5:
                count += 1

    active = np.empty(count, dtype=np.int32)
    idx = 0
    for j in range(nmaps):
        for i in range(npix):
            if abs(mask[i, j]) > 0.5:
                active[idx] = j * npix + i
                idx += 1
    return active


def compute_pointings(nside, npixs, point_vectors, active, ordering):
    nmaps = len(npixs)

    for field_idx in range(nmaps):
        ntemp = npixs[field_idx]

        for i in range(ntemp):
            theta, phi = hp.pix2ang(nside, active[field_idx, i], nest=(ordering == 1))
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            norm = np.sqrt(x**2 + y**2 + z**2)
            point_vectors[field_idx][i, 0] = x / norm
            point_vectors[field_idx][i, 1] = y / norm
            point_vectors[field_idx][i, 2] = z / norm

    return point_vectors


@njit(cache=True)
def compute_00_contribution(cl, S_slice, vec1, vec2, legendre, mode, remove_dipole=False):
    npix = S_slice.shape[0]
    if mode == 0:
        entry = np.sum(cl)
        if remove_dipole:
            entry += 1000.0 * cl[0] * 2.0
        for i in range(npix):
            S_slice[i, i] = entry

        for i in range(npix):
            for j in range(i + 1, npix):
                legendre_00_inplace(vec1[j] @ vec2[i].T, legendre)
                S_slice[j, i] = np.sum(cl * legendre[1:])
                if remove_dipole:
                    S_slice[j, i] += 1000.0 * cl[0] * (1.0 + legendre[0])

    elif mode == 1:
        for i in range(npix):
            for j in range(npix):
                legendre_00_inplace(vec1[j] @ vec2[i].T, legendre)
                S_slice[j, i] = np.sum(cl * legendre[1:])


@njit(cache=True)
def compute_22_contribution(cl11, cl22, cl12, S_slice, vec1, vec2, legendre, f1, f2):
    npix = S_slice.shape[0] // 2

    legendre_22_inplace(1.0, legendre, f1, f2)

    qq = np.sum(cl11 * f1[1:]) - np.sum(cl22 * f2[1:])
    uu = np.sum(cl22 * f1[1:]) - np.sum(cl11 * f2[1:])
    qu = np.sum((f1[1:] + f2[1:]) * cl12)

    for i in range(npix):
        S_slice[i, i] = qq
        S_slice[i + npix, i] = qu
    for i in range(npix, npix * 2):
        S_slice[i, i] = uu

    for i in range(npix):
        for j in range(i + 1, npix):
            legendre_22_inplace(vec1[j] @ vec2[i].T, legendre, f1, f2)

            qq = np.sum(cl11 * f1[1:]) - np.sum(cl22 * f2[1:])
            uu = np.sum(cl22 * f1[1:]) - np.sum(cl11 * f2[1:])
            qu = np.sum((f1[1:] + f2[1:]) * cl12)

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


@njit(cache=True)
def compute_02_contribution(cl12, cl13, S_slice, vec0, vec2, legendre):
    npix_spin0 = vec0.shape[0]
    npix_spin2 = vec2.shape[0]

    for i in range(npix_spin0):
        for j in range(npix_spin2):
            legendre_02_inplace(vec2[j] @ vec0[i].T, legendre)
            tq = -np.sum(cl12 * legendre[1:])
            tu = -np.sum(cl13 * legendre[1:])
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
                remove_dipole = (
                    True if lf_i.maps_label + lf_j.maps_label == "TT" else False
                )
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
                    msg = "Cross-correlation for spin-2 fields not implemented yet."
                    raise NotImplementedError(msg)
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
def derivative_step_00(S_slice, wl, vec1, vec2, current_ell, legendre, mode):
    npix = S_slice.shape[0]
    if mode == 0:
        for i in range(npix):
            S_slice[i, i] = wl[current_ell - 2]

        for i in range(npix):
            for j in range(i + 1, npix):
                legendre_00_inplace(vec1[j] @ vec2[i].T, legendre)
                S_slice[j, i] = legendre[current_ell - 1] * wl[current_ell - 2]

        for i in range(S_slice.shape[0]):
            for j in range(i + 1, S_slice.shape[0]):
                S_slice[i, j] = S_slice[j, i]

    elif mode == 1:
        for i in range(npix):
            for j in range(npix):
                legendre_00_inplace(vec1[j] @ vec2[i].T, legendre)
                S_slice[j, i] = legendre[current_ell - 1] * wl[current_ell - 2]


@njit(cache=True)
def derivative_step_02(S_slice, wl, vec0, vec2, current_ell, mode, legendre):
    npix_spin0 = vec0.shape[0]
    npix_spin2 = vec2.shape[0]

    for i in range(npix_spin0):
        for j in range(npix_spin2):
            legendre_02_inplace(vec2[j] @ vec0[i].T, legendre)

            ang1, _ = get_rotation_angle(vec2[j], vec0[i])
            cos1 = np.cos(ang1)
            sin1 = np.sin(ang1)

            if mode == 0:
                S_slice[j, i] = -wl[current_ell - 2] * legendre[current_ell - 1] * cos1
                S_slice[j + npix_spin2, i] = (
                    -wl[current_ell - 2] * legendre[current_ell - 1] * sin1
                )
            else:
                S_slice[j, i] = wl[current_ell - 2] * legendre[current_ell - 1] * sin1
                S_slice[j + npix_spin2, i] = (
                    -wl[current_ell - 2] * legendre[current_ell - 1] * cos1
                )


@njit(cache=True)
def derivative_step_22(S_slice, wl, vec1, vec2, current_ell, mode, legendre, f1, f2):
    npix = S_slice.shape[0] // 2

    legendre_22_inplace(1.0, legendre, f1, f2)

    if mode == 0:  # such as EE
        qq = f1[current_ell - 1] * wl[current_ell - 2]
        uu = -f2[current_ell - 1] * wl[current_ell - 2]
        qu = 0.0
    elif mode == 1:  # such as BB
        qq = -f2[current_ell - 1] * wl[current_ell - 2]
        uu = f1[current_ell - 1] * wl[current_ell - 2]
        qu = 0.0
    elif mode == 2:  # such as EB
        qq = 0.0
        uu = 0.0
        qu = (f1[current_ell - 1] + f2[current_ell - 1]) * wl[current_ell - 2]

    for i in range(npix):
        S_slice[i, i] = qq
        S_slice[i + npix, i] = qu
    for i in range(npix, npix * 2):
        S_slice[i, i] = uu

    for i in range(npix):
        for j in range(i + 1, npix):
            legendre_22_inplace(vec1[j] @ vec2[i].T, legendre, f1, f2)
            if mode == 0:  # such as EE
                qq = f1[current_ell - 1] * wl[current_ell - 2]
                uu = -f2[current_ell - 1] * wl[current_ell - 2]
                qu = 0.0
            elif mode == 1:  # such as BB
                qq = -f2[current_ell - 1] * wl[current_ell - 2]
                uu = f1[current_ell - 1] * wl[current_ell - 2]
                qu = 0.0
            elif mode == 2:  # such as EB
                qq = 0.0
                uu = 0.0
                qu = (f1[current_ell - 1] + f2[current_ell - 1]) * wl[current_ell - 2]

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
    spins = fields.spin
    npixs = fields.n_active
    label = fields.spectra_labels[spectrum]

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

    ell = np.arange(2, current_ell + 1)
    factor2 = 1 / ((ell + 2) * (ell + 1) * ell * (ell - 1))
    factor = np.sqrt(factor2)
    chngconv = (2 * ell + 1) / (4 * np.pi)

    block = S[col_offset : col_offset + ncol, row_offset : row_offset + nrow]
    legendre = np.empty(current_ell)

    if spin_i == 0 and spin_j == 0:
        derivative_step_00(
            block,
            chngconv,
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
            chngconv * factor,
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
            chngconv * factor2,
            point_vectors_i,
            point_vectors_j,
            current_ell,
            mode,
            legendre,
            f1,
            f2,
        )
