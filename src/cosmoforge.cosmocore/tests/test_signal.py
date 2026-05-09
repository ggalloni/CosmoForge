import os
import tempfile

import healpy as hp
import numpy as np
import pytest

from cosmocore import (
    FieldCollection,
    InputParams,
    compute_pointings,
    compute_signal_matrix,
    create_field,
    read_mask,
)
from cosmocore.core import Core
from cosmocore.pixel import compute_00_contribution


class _StubCore(Core):
    """Concrete Core for exercising ``read_params`` validation paths."""

    def compute(self):
        pass

    def run(self):
        pass


def test_signal_covmat(data_resolver):
    Par = InputParams()
    mock_config_dict = {
        "nside": 4,
        "lmax": 8,
        "spins": [0, 2],
        "labels": ["T", "E", "B"],
        "physical_labels": ["T", "Q", "U"],
        "maskfile": "tmp/mask.fits",
        "ordering": "NESTED",
    }

    # Create temporary mask file
    mask = np.ones((3, hp.nside2npix(4)), dtype=np.float64)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_mask_file:
        hp.write_map(tmp_mask_file.name, mask, overwrite=True)
        mock_config_dict["maskfile"] = tmp_mask_file.name
    Par.update(mock_config_dict)

    npix = hp.nside2npix(Par.nside)
    mask = np.empty((Par.nfields, npix), dtype=np.float64)
    mask = read_mask(Par.maskfile, mask)

    # Create fields using the new design
    fields = []
    counter = 0
    for spin in Par.spins:
        if spin == 0:
            labels = Par.labels[counter]
            counter += 1
        else:
            labels = [Par.labels[counter], Par.labels[counter + 1]]
            counter += 2

        # Use new factory function for type-safe field creation
        if mask.ndim == 1:
            mask = mask[:, np.newaxis]
        field = create_field(
            spin=spin,
            nside=Par.nside,
            lmax=Par.lmax,
            mask=mask[:, counter - 1],
            labels=labels,
        )
        fields.append(field)

    # Create collection using new design
    collection = FieldCollection(Par, fields)

    npixs = []
    for field in fields:
        npixs += field.n_active if field.spin == 0 else field.n_active * 2

    point_vectors = tuple(
        np.empty((npixs[i], 3), dtype=np.float64) for i in range(len(npixs))
    )
    theta_vectors = tuple(np.empty(npixs[i], dtype=np.float64) for i in range(len(npixs)))
    phi_vectors = tuple(np.empty(npixs[i], dtype=np.float64) for i in range(len(npixs)))
    pixact = collection.get_active_pixels()
    point_vectors, theta_vectors, phi_vectors = compute_pointings(
        Par.nside, npixs, point_vectors, theta_vectors, phi_vectors, pixact, Par.ordering
    )

    collection.set_pointing_vectors(point_vectors)

    # ℓ-indexed file format: rows for ell = 2 .. lmax (ell=0,1 omitted from
    # the file; readcl zero-pads them on read).
    n_rows = Par.lmax - 1
    Cls = np.zeros((n_rows, collection.n_spectra + 1), dtype=np.float64)
    Cls[:, 0] = np.arange(2, Par.lmax + 1)  # ell values
    Cls[:, 1] = 1e-4 / np.arange(2, Par.lmax + 1) ** 2  # TT
    Cls[:, 2] = 0.5e-4 / np.arange(2, Par.lmax + 1) ** 2  # EE
    Cls[:, 3] = 0.1e-4 / np.arange(2, Par.lmax + 1) ** 2  # BB
    Cls[:, 4] = 0.3e-4 / np.arange(2, Par.lmax + 1) ** 2  # TE
    Cls[:, 5] = 0.0  # TB
    Cls[:, 6] = 0.0  # EB

    with tempfile.NamedTemporaryFile(delete=False) as tmp_cls_file:
        Par.inputclfile = tmp_cls_file.name
    np.savetxt(Par.inputclfile, Cls, header="ell TT EE BB TE TB EB", fmt="%.16e")

    collection.set_cls()
    collection.set_beams()

    ntot = collection.total_active_pixels
    signal_covmat = np.zeros((ntot, ntot), dtype=np.float64)
    signal_covmat = np.asfortranarray(signal_covmat, dtype=np.float64)

    compute_signal_matrix(
        S=signal_covmat,
        lmax=Par.lmax,
        fields=collection,
    )

    os.remove(mock_config_dict["maskfile"])

    ref_path = data_resolver("tests/data/ref_TQU_signal.dat")
    ref = np.loadtxt(ref_path)
    np.testing.assert_allclose(signal_covmat, ref, rtol=1e-12, atol=1e-20)


def test_signal_covmat_multiple_scalar_fields(data_resolver):
    """Test signal covariance matrix with multiple scalar fields to trigger mode=1."""
    Par = InputParams()
    mock_config_dict = {
        "nside": 4,
        "lmax": 8,
        "spins": [0, 0],  # Two scalar fields to trigger T1-T2 cross-correlation
        "labels": ["T1", "T2"],
        "physical_labels": ["T1", "T2"],
        "ordering": "NESTED",
        "maskfile": "tmp/mask.fits",
    }

    # Create temporary mask file with 2 scalar fields
    mask = np.ones((2, hp.nside2npix(4)), dtype=np.float64)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_mask_file:
        hp.write_map(tmp_mask_file.name, mask, overwrite=True)
        mock_config_dict["maskfile"] = tmp_mask_file.name
    Par.update(mock_config_dict)

    npix = hp.nside2npix(Par.nside)
    mask = np.empty((Par.nfields, npix), dtype=np.float64)
    mask = read_mask(Par.maskfile, mask)

    # Create fields using the new design
    fields = []
    counter = 0
    for spin in Par.spins:
        if spin == 0:
            labels = Par.labels[counter]
            counter += 1
        else:
            labels = [Par.labels[counter], Par.labels[counter + 1]]
            counter += 2

        if mask.ndim == 1:
            mask = mask[:, np.newaxis]
        field = create_field(
            spin=spin,
            nside=Par.nside,
            lmax=Par.lmax,
            mask=mask[:, counter - 1],
            labels=labels,
        )
        fields.append(field)

    # Create collection using new design
    collection = FieldCollection(Par, fields)

    npixs = []
    for field in fields:
        npixs += field.n_active if field.spin == 0 else field.n_active * 2

    point_vectors = tuple(
        np.empty((npixs[i], 3), dtype=np.float64) for i in range(len(npixs))
    )
    theta_vectors = tuple(np.empty(npixs[i], dtype=np.float64) for i in range(len(npixs)))
    phi_vectors = tuple(np.empty(npixs[i], dtype=np.float64) for i in range(len(npixs)))
    pixact = collection.get_active_pixels()
    point_vectors, theta_vectors, phi_vectors = compute_pointings(
        Par.nside, npixs, point_vectors, theta_vectors, phi_vectors, pixact, Par.ordering
    )

    collection.set_pointing_vectors(point_vectors)

    n_rows = Par.lmax - 1
    Cls = np.zeros((n_rows, collection.n_spectra + 1), dtype=np.float64)
    Cls[:, 0] = np.arange(2, Par.lmax + 1)  # ell values
    Cls[:, 1] = 1e-4 / np.arange(2, Par.lmax + 1) ** 2  # T1-T1
    Cls[:, 2] = 0.8e-4 / np.arange(2, Par.lmax + 1) ** 2  # T2-T2
    Cls[:, 3] = 0.5e-4 / np.arange(2, Par.lmax + 1) ** 2  # T1-T2 cross-correlation

    with tempfile.NamedTemporaryFile(delete=False) as tmp_cls_file:
        Par.inputclfile = tmp_cls_file.name
    np.savetxt(Par.inputclfile, Cls, header="ell T1T1 T2T2 T1T2", fmt="%.16e")

    collection.set_cls()
    collection.set_beams()

    ntot = collection.total_active_pixels
    signal_covmat = np.zeros((ntot, ntot), dtype=np.float64)
    signal_covmat = np.asfortranarray(signal_covmat, dtype=np.float64)

    compute_signal_matrix(
        S=signal_covmat,
        lmax=Par.lmax,
        fields=collection,
    )

    ref_path = data_resolver("tests/data/ref_T1T2_signal.dat")
    ref = np.loadtxt(ref_path)

    # Clean up
    os.remove(mock_config_dict["maskfile"])
    os.remove(Par.inputclfile)

    np.testing.assert_allclose(signal_covmat, ref, rtol=1e-12, atol=1e-20)


# =============================================================================
# ADR 0009 — multipole-range API: per-component lmin_signal validation
# =============================================================================


def _labels_for_spins(spins):
    labels = []
    for s in spins:
        if s == 0:
            labels.append("T")
        else:
            labels.extend(["Q", "U"])
    return labels


def _minimal_core(lmin_signal, spins=(0,), nside=4, lmax=8):
    """Build a Core wrapper exercising the requested ``lmin_signal``."""
    params = InputParams()
    cfg = {
        "nside": nside,
        "lmax": lmax,
        "spins": list(spins),
        "labels": _labels_for_spins(spins),
        "lmin_signal": lmin_signal,
    }
    params.update(cfg)
    return _StubCore(params)


def test_lmin_signal_scalar_broadcast():
    """Scalar ``lmin_signal=N`` is normalised to ``[N]*n_components``."""
    core = _minimal_core(lmin_signal=2, spins=(0, 2))
    assert core.params.lmin_signal == [2, 2]


def test_lmin_signal_per_component_list_accepted():
    """Per-component lists pass through unchanged when within spin floors."""
    core = _minimal_core(lmin_signal=[1, 2], spins=(0, 2))
    assert core.params.lmin_signal == [1, 2]


def test_lmin_signal_below_spin_floor_rejected():
    """``lmin_signal[i] < |spin|`` fails fast with a per-component message."""
    with pytest.raises(ValueError, match=r"lmin_signal\[1\]=1 invalid for spin-2"):
        _minimal_core(lmin_signal=[1, 1], spins=(0, 2))


def test_lmin_signal_length_mismatch_rejected():
    """Per-component lists must match ``len(spins)``."""
    with pytest.raises(ValueError, match=r"len\(lmin_signal\)=3 != len\(spins\)=2"):
        _minimal_core(lmin_signal=[2, 2, 2], spins=(0, 2))


def test_lmax_above_lmax_signal_clamped():
    """``lmax > lmax_signal`` is silently clamped to ``lmax_signal``.

    The legacy ``InputParams.lmax`` default (64) predates ADR 0009 and
    routinely exceeds ``lmax_signal=4*nside`` for small nside; clamping
    keeps existing analyses running while still pinning the inference
    window to the signal-cov band.
    """
    params = InputParams()
    cfg = {
        "nside": 4,
        "lmax": 64,
        "lmax_signal": 16,
        "spins": [0],
        "labels": ["T"],
    }
    params.update(cfg)
    core = _StubCore(params)
    assert core.params.lmax == 16


def test_lmin_above_lmax_rejected():
    """Explicit conflicts between ``lmin`` and ``lmax`` still raise."""
    params = InputParams()
    cfg = {
        "nside": 4,
        "lmin": 6,
        "lmax": 4,
        "lmax_signal": 16,
        "spins": [0],
        "labels": ["T"],
    }
    params.update(cfg)
    with pytest.raises(ValueError, match=r"lmax=4 < lmin=6"):
        _StubCore(params)


# =============================================================================
# ADR 0009 — foreground/template support via lmin_signal=0/1
# =============================================================================


def test_compute_00_contribution_includes_monopole():
    """A pure monopole template (``cl[0]=c0``) gives a constant ``c0/(4π)``.

    P_0(x)=1 for every pixel pair, so a foreground template carrying
    only a monopole produces a uniform offset on the signal matrix.
    Before ADR 0009 the loop hard-coded ``ell ≥ 2`` and silently dropped
    this contribution.
    """
    rng = np.random.default_rng(42)
    n_pix = 16
    theta = rng.uniform(0.3, np.pi - 0.3, n_pix)
    phi = rng.uniform(0, 2 * np.pi, n_pix)
    vec = np.column_stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )

    lmax = 6
    cl = np.zeros(lmax + 1, dtype=np.float64)
    c0 = 1.5
    cl[0] = c0
    legendre = np.empty(lmax + 1, dtype=np.float64)

    S = np.zeros((n_pix, n_pix), dtype=np.float64)
    compute_00_contribution(cl, S, vec, vec, legendre, mode=0)
    S_full = S + S.T - np.diag(np.diag(S))

    expected = c0 / (4 * np.pi)
    np.testing.assert_allclose(S_full, np.full_like(S_full, expected), rtol=1e-13)


def test_compute_00_contribution_dipole_template_matches_closed_form():
    """A pure dipole template (``cl[1]=c1``) reproduces ``3·c1·x_ij/(4π)``.

    With the per-component low-ℓ floor lifted, ``lmin_signal=1`` lets a
    foreground component carry its dipole into the signal matrix; the
    result must equal the closed-form Legendre sum on every pixel pair.
    """
    rng = np.random.default_rng(7)
    n_pix = 12
    theta = rng.uniform(0.3, np.pi - 0.3, n_pix)
    phi = rng.uniform(0, 2 * np.pi, n_pix)
    vec = np.column_stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )

    lmax = 4
    cl = np.zeros(lmax + 1, dtype=np.float64)
    c1 = 0.7
    cl[1] = c1
    legendre = np.empty(lmax + 1, dtype=np.float64)

    S = np.zeros((n_pix, n_pix), dtype=np.float64)
    compute_00_contribution(cl, S, vec, vec, legendre, mode=0)
    S_full = S + S.T - np.diag(np.diag(S))

    cos_pair = vec @ vec.T
    expected = c1 * 3.0 * cos_pair / (4 * np.pi)
    np.testing.assert_allclose(S_full, expected, rtol=1e-12)


def test_compute_00_contribution_default_skips_monopole():
    """With ``cl[0]=cl[1]=0`` the ℓ-from-0 loop is a no-op.

    Default CMB usage has ``lmin_signal=2``; cl arrays are zero-padded
    below ℓ=2. The new loop bound must produce results identical to the
    legacy ``ell ≥ 2`` summation when those entries are zero.
    """
    rng = np.random.default_rng(13)
    n_pix = 10
    theta = rng.uniform(0.3, np.pi - 0.3, n_pix)
    phi = rng.uniform(0, 2 * np.pi, n_pix)
    vec = np.column_stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )

    lmax = 6
    cl = np.zeros(lmax + 1, dtype=np.float64)
    cl[2:] = rng.uniform(0.1, 1.0, lmax - 1)
    legendre = np.empty(lmax + 1, dtype=np.float64)

    S = np.zeros((n_pix, n_pix), dtype=np.float64)
    compute_00_contribution(cl, S, vec, vec, legendre, mode=0)
    S_full = S + S.T - np.diag(np.diag(S))

    cos_pair = vec @ vec.T
    expected = np.zeros_like(S_full)
    P_prev = np.ones_like(cos_pair)  # P_0
    P_curr = cos_pair.copy()  # P_1
    for ell in range(2, lmax + 1):
        P_next = ((2 * ell - 1) * cos_pair * P_curr - (ell - 1) * P_prev) / ell
        P_prev, P_curr = P_curr, P_next
        expected += cl[ell] * (2 * ell + 1) / (4 * np.pi) * P_curr

    np.testing.assert_allclose(S_full, expected, rtol=1e-12)
