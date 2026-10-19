"""Tests for SymmetryMode wiring through Fisher and Spectra (ADR-0011)."""

import os

import healpy as hp
import numpy as np
import pytest

from cosmocore import Bins, active_pixel_index
from cosmocore.spectrum_key import SpectrumKind, SymmetryMode
from qube import Fisher, Spectra


@pytest.fixture
def fisher_t_qu(config_resolver):
    """Fisher instance over T + QU components — exercises cross-component
    spin-0 x spin-2 (TE, TB) and auto-pair spin-2 x spin-2 (EE, BB, EB)."""
    config_file = config_resolver("tests/data/nside4/TQU/config.yaml")
    fisher = Fisher(config_file)
    fisher.setup_fields()
    fisher.setup_cls(lmax=fisher.lmax_signal)
    os.unlink(config_file)
    return fisher


def test_fisher_default_symmetry_mode_is_symmetric(fisher_t_qu):
    """Fisher defaults to SYMMETRIC — pre-Slice-5 behaviour is the default."""
    assert fisher_t_qu.symmetry_mode is SymmetryMode.SYMMETRIC


def test_fisher_accepts_explicit_symmetry_mode(config_resolver):
    """Constructor accepts an explicit SymmetryMode kwarg and stores it."""
    config_file = config_resolver("tests/data/nside4/TQU/config.yaml")
    fisher = Fisher(config_file, symmetry_mode=SymmetryMode.DIRECTIONAL)
    os.unlink(config_file)
    assert fisher.symmetry_mode is SymmetryMode.DIRECTIONAL


def test_build_inputs_forwards_symmetry_mode(fisher_t_qu):
    """_build_multi_spectrum_inputs forwards the flag into SpectraManager.

    Verified by setting DIRECTIONAL on a T+QU collection: there are no
    spin-2 cross-component pairs here so the emitted keys are identical
    to SYMMETRIC, but the forwarding contract is exercised end-to-end
    (no crash, valid keys returned)."""
    fisher_t_qu.symmetry_mode = SymmetryMode.DIRECTIONAL
    cl_dict, keys = fisher_t_qu._build_multi_spectrum_inputs()
    assert len(keys) > 0
    # T + QU has no spin-2 cross-component pair, so no CG should appear.
    assert all(k.kind is not SpectrumKind.CG for k in keys)


def test_symmetric_t_qu_end_to_end_smoke(config_resolver):
    """End-to-end Fisher.run() smoke test on T+QU under default SYMMETRIC.

    Exercises the int-mode call sites in fisher.py (fixed in Slice 5 to pass
    is_cross to kind_to_legacy_mode) and the symmetry_mode threading through
    get_binned_derivative_matrix. T+QU has spin-0 x spin-2 cross pairs (TE,
    TB), which use a different encoding path than the spin-2 x spin-2
    cross-pair encoding affected by the is_cross fix — this test guards
    against a regression in the unaffected paths."""
    config_file = config_resolver("tests/data/nside4/TQU/config.yaml")
    fisher = Fisher(config_file)
    os.unlink(config_file)
    fisher.run()
    assert fisher.fisher is not None
    assert fisher.fisher.shape[0] == fisher.fisher.shape[1]


def test_spectra_inherits_symmetry_mode_from_fisher(config_resolver):
    """Spectra inherits symmetry_mode from the Fisher instance (ADR-0011) —
    the user sets the flag once on Fisher and Spectra can never diverge."""
    config_file = config_resolver("tests/data/nside4/TQU/config.yaml")
    fisher = Fisher(config_file)
    fisher.run()
    # Override on the Fisher instance after construction to verify the
    # propagation contract, then construct Spectra.
    fisher.symmetry_mode = SymmetryMode.DIRECTIONAL

    config_file2 = config_resolver("tests/data/nside4/TQU/config.yaml")
    spectra = Spectra(config_file2, fisher=fisher)
    os.unlink(config_file)
    os.unlink(config_file2)
    assert spectra.symmetry_mode is SymmetryMode.DIRECTIONAL


# === symmetry_mode coercion (Fisher.__init__) ===========================


@pytest.mark.parametrize(
    "value,expected",
    [
        ("SYMMETRIC", SymmetryMode.SYMMETRIC),
        ("symmetric", SymmetryMode.SYMMETRIC),
        ("Directional", SymmetryMode.DIRECTIONAL),
        ("DIRECTIONAL", SymmetryMode.DIRECTIONAL),
    ],
)
def test_fisher_coerces_symmetry_mode_string(config_resolver, value, expected):
    """String symmetry_mode (e.g. from YAML) is coerced to the enum so
    downstream `is SymmetryMode.*` identity checks don't silently misroute."""
    config_file = config_resolver("tests/data/nside4/TQU/config.yaml")
    fisher = Fisher(config_file, symmetry_mode=value)
    os.unlink(config_file)
    assert fisher.symmetry_mode is expected


def test_fisher_rejects_unknown_symmetry_mode_string(config_resolver):
    """Unknown symmetry_mode strings raise ValueError listing valid names."""
    config_file = config_resolver("tests/data/nside4/TQU/config.yaml")
    with pytest.raises(ValueError, match="unknown symmetry_mode"):
        Fisher(config_file, symmetry_mode="bogus")
    os.unlink(config_file)


def test_fisher_rejects_non_string_non_enum_symmetry_mode(config_resolver):
    """Non-string, non-enum symmetry_mode raises TypeError."""
    config_file = config_resolver("tests/data/nside4/TQU/config.yaml")
    with pytest.raises(TypeError, match="symmetry_mode must be"):
        Fisher(config_file, symmetry_mode=42)
    os.unlink(config_file)


# === Two spin-2 components (cross-component EB-like pair) ===============

TWO_QU_NSIDE = 8
TWO_QU_LMAX = 16
TWO_QU_LABELS = ["E1", "B1", "E2", "B2"]


def _two_qu_params(tmp_path):
    out = str(tmp_path)
    return {
        "nside": TWO_QU_NSIDE,
        "spins": [2, 2],
        "labels": TWO_QU_LABELS,
        "physical_labels": ["Q1", "U1", "Q2", "U2"],
        "do_cross": False,
        "maskfile": "/nonexistent/mask.fits",
        "output_geometry_file": f"{out}/geometry.dat",
        "ordering": "RING",
        "inputclfile": "/nonexistent/cls.txt",
        "input_convention": "Cl",
        "covmatfile1": "/nonexistent/ncvm1.bin",
        "covmatfile2": "/nonexistent/ncvm2.bin",
        "lmax": TWO_QU_LMAX,
        "lmax_signal": TWO_QU_LMAX,
        "load_inverted": False,
        "smoothing_type": "none",
        "fwhmarcmin": 0.0,
        "apply_pixwin": False,
        "beam_file": "",
        "outnoisecovmat1": f"{out}/rn1.bin",
        "outnoisecovmat2": f"{out}/rn2.bin",
        "feedback": 0,
        "outinvcovmatfile1": f"{out}/ic1.bin",
        "outinvcovmatfile2": f"{out}/ic2.bin",
        "outfilefisher": f"{out}/fisher.dat",
        "nsims": 1,
        "inputmapfile1": "/nonexistent/maps.npy",
        "inputmapfile2": "/nonexistent/maps.npy",
        "outcovmatfile": f"{out}/cov.dat",
        "outerrfile": f"{out}/err.dat",
        "remove_nb": False,
    }


def _two_qu_inputs():
    """Mask, noise and fiducial C_ell for two QU components (4 maps)."""
    npix = hp.nside2npix(TWO_QU_NSIDE)
    radius = np.arccos(1.0 - 2.0 * 0.4)
    pix = hp.query_disc(TWO_QU_NSIDE, hp.ang2vec(0.0, 0.0), radius)
    mask = np.zeros((npix, 4))
    mask[pix, :] = 1.0
    n_act = len(active_pixel_index(mask))
    rng = np.random.default_rng(1)
    noise = np.diag(rng.uniform(0.5, 1.5, n_act) * 1e-3)

    ell = np.arange(TWO_QU_LMAX + 1, dtype=np.float64)
    ee = np.zeros_like(ell)
    ee[2:] = 1.0 / (ell[2:] * (ell[2:] + 1.0))
    base = {"E": {"E": ee, "B": 0.05 * ee}, "B": {"E": 0.05 * ee, "B": 0.3 * ee}}
    cls_data = {}
    for a in TWO_QU_LABELS:
        for b in TWO_QU_LABELS:
            same_comp = a[1] == b[1]
            cls_data[a + b] = (1.0 if same_comp else 0.9) * base[a[0]][b[0]]
    return mask, noise, cls_data


def _distinct_eb_beam():
    """Injected beam whose T, E and B rows all differ, so b_E b_B != b_B b_B."""
    ell = np.arange(TWO_QU_LMAX + 1, dtype=np.float64)
    return np.vstack([np.exp(-ell / 40.0), np.exp(-ell / 30.0), np.exp(-ell / 20.0)])


@pytest.mark.parametrize(
    "mode,n_spectra",
    [(SymmetryMode.SYMMETRIC, 9), (SymmetryMode.DIRECTIONAL, 10)],
)
def test_two_qu_harmonic_fisher_runs(tmp_path, mode, n_spectra):
    """Two spin-2 components run on the harmonic path in both modes.

    SYMMETRIC folds the cross-pair CG into GC, so it carries one spectrum
    fewer than ``nfields (nfields + 1) / 2``. The Fisher axis must follow
    ``len(spectra_list)``, not the field-count formula."""
    mask, noise, cls_data = _two_qu_inputs()
    fisher = Fisher(
        _two_qu_params(tmp_path),
        basis="harmonic",
        symmetry_mode=mode,
        mask=mask,
        noise_cov1=noise,
        cls_data=cls_data,
    )
    fisher.set_binning(Bins.fromdeltal(2, TWO_QU_LMAX, 3))
    fisher.run()

    nbins = fisher.bins.nbins
    assert len(fisher.spectra_list) == n_spectra
    assert fisher.n_params == n_spectra * nbins
    assert fisher.fisher.shape == (n_spectra * nbins, n_spectra * nbins)
    assert np.all(np.isfinite(fisher.fisher))
    errors = fisher.get_error_bars()
    assert errors.shape == (n_spectra * nbins,)
    assert np.all(np.isfinite(errors)) and np.all(errors > 0)
    assert len(fisher.get_bandpower_slices()) == n_spectra
    window = fisher.get_bandpower_window_function()
    assert window.shape == (n_spectra * nbins, n_spectra * fisher.n_ell)


@pytest.mark.parametrize("mode", [SymmetryMode.SYMMETRIC, SymmetryMode.DIRECTIONAL])
def test_two_qu_beam_smoothing_follows_spectrum_keys(tmp_path, mode):
    """Each Fisher row block carries b_i(l) b_j(l) of its own SpectrumKey.

    Multi-character field labels (E1, B1, ...) used to be parsed by
    character, so no beam was found and every factor was 1. Under
    SYMMETRIC the label order also kept the dropped B1E2 slot, which
    shifted the cross-pair BB onto the B1E2 beam."""
    mask, noise, cls_data = _two_qu_inputs()
    beam = _distinct_eb_beam()
    fisher = Fisher(
        _two_qu_params(tmp_path),
        basis="harmonic",
        symmetry_mode=mode,
        mask=mask,
        noise_cov1=noise,
        cls_data=cls_data,
        beam=beam,
    )
    fisher.run()

    lmin = fisher.params.lmin
    slot_beam = {"G": beam[1], "C": beam[2]}
    blocks = fisher.beam_smoothing.reshape(len(fisher.spectra_list), fisher.n_ell)
    for key, block in zip(fisher.spectra_list, blocks, strict=True):
        slot_i, slot_j = (s.label for s in key.kind.slots)
        expected = (slot_beam[slot_i] * slot_beam[slot_j])[lmin : TWO_QU_LMAX + 1]
        np.testing.assert_allclose(block, expected, rtol=1e-12, err_msg=str(key))


@pytest.mark.parametrize(
    "mode,n_spectra",
    [(SymmetryMode.SYMMETRIC, 9), (SymmetryMode.DIRECTIONAL, 10)],
)
def test_two_qu_spectra_estimate_axis_follows_spectra_list(tmp_path, mode, n_spectra):
    """Spectra sizes its QML estimates by the Fisher's spectra_list."""
    mask, noise, cls_data = _two_qu_inputs()
    params = _two_qu_params(tmp_path)
    params["nsims"] = 2
    fisher = Fisher(
        params,
        basis="harmonic",
        symmetry_mode=mode,
        mask=mask,
        noise_cov1=noise,
        cls_data=cls_data,
    )
    fisher.set_binning(Bins.fromdeltal(2, TWO_QU_LMAX, 3))
    fisher.run()

    maps1 = np.random.default_rng(2).standard_normal((noise.shape[0], 2))
    spectra = Spectra(params, fisher=fisher, maps1=maps1)
    spectra.run()

    n_params = n_spectra * fisher.bins.nbins
    assert spectra.qml_results.shape == (2, n_params)
    assert np.all(np.isfinite(spectra.qml_results))
    assert spectra.get_covariance().shape == (n_params, n_params)
