"""Tests for SpectraManager.build_inputs (Slice 1, Task 1.6)."""

import healpy as hp
import numpy as np
import pytest

from cosmocore import SpectraManager, create_field
from cosmocore.spectrum_key import SpectrumKey, SpectrumKind


@pytest.fixture
def collection_T_QU():
    """Two components: T at index 0, EB (spin-2) at index 1."""
    nside, lmax = 4, 8
    npix = hp.nside2npix(nside)
    mask = np.ones(npix, dtype=np.float64)
    return [
        create_field(spin=0, nside=nside, lmax=lmax, mask=mask, labels="T"),
        create_field(spin=2, nside=nside, lmax=lmax, mask=mask, labels=["E", "B"]),
    ]


def _populate_fiducial_cls(sm: SpectraManager) -> None:
    """Populate fiducial Cls so build_inputs can resolve every label."""
    lmax = sm.fields[0].lmax
    ell = np.arange(lmax + 1)
    cls = {}
    for label in sm.labels:
        cls[label] = 1.0 / (ell + 1.0) ** 2
        cls[label][:2] = 0.0
    sm.set_cls(cls)


def test_build_inputs_emits_spectrum_keys(collection_T_QU):
    sm = SpectraManager(collection_T_QU)
    _populate_fiducial_cls(sm)

    cl_dict, keys = sm.build_inputs()

    assert all(isinstance(k, SpectrumKey) for k in keys)

    actual = {(k.comp_i, k.comp_j, k.kind) for k in keys}
    assert (0, 0, SpectrumKind.SS) in actual
    assert (1, 1, SpectrumKind.GG) in actual
    assert (1, 1, SpectrumKind.CC) in actual
    assert (1, 1, SpectrumKind.GC) in actual
    assert (0, 1, SpectrumKind.SG) in actual
    assert (0, 1, SpectrumKind.SC) in actual
    # Directional CG NOT emitted in symmetric default
    assert (1, 1, SpectrumKind.CG) not in actual


def test_build_inputs_dict_keyed_by_spectrum_key(collection_T_QU):
    sm = SpectraManager(collection_T_QU)
    _populate_fiducial_cls(sm)

    cl_dict, keys = sm.build_inputs()
    for key in keys:
        assert key in cl_dict
        assert isinstance(cl_dict[key], np.ndarray)


@pytest.fixture
def collection_QU_QU():
    """Two QU components (both spin-2) — the case where SymmetryMode applies."""
    nside, lmax = 4, 8
    npix = hp.nside2npix(nside)
    mask = np.ones(npix, dtype=np.float64)
    return [
        create_field(spin=2, nside=nside, lmax=lmax, mask=mask, labels=["E1", "B1"]),
        create_field(spin=2, nside=nside, lmax=lmax, mask=mask, labels=["E2", "B2"]),
    ]


def test_symmetric_does_not_emit_cg_for_cross_qu(collection_QU_QU):
    from cosmocore.spectrum_key import SymmetryMode

    sm = SpectraManager(collection_QU_QU)
    _populate_fiducial_cls(sm)

    cl_dict, keys = sm.build_inputs(symmetry_mode=SymmetryMode.SYMMETRIC)
    cross_kinds = {k.kind for k in keys if k.comp_i != k.comp_j}
    assert SpectrumKind.GC in cross_kinds
    assert SpectrumKind.CG not in cross_kinds


def test_directional_emits_cg_for_cross_qu(collection_QU_QU):
    from cosmocore.spectrum_key import SymmetryMode

    sm = SpectraManager(collection_QU_QU)
    _populate_fiducial_cls(sm)

    cl_dict, keys = sm.build_inputs(symmetry_mode=SymmetryMode.DIRECTIONAL)
    cross_kinds = {k.kind for k in keys if k.comp_i != k.comp_j}
    assert SpectrumKind.GC in cross_kinds
    assert SpectrumKind.CG in cross_kinds

    # Both GC (E_0 × B_1) and CG (B_0 × E_1) keys are populated independently
    # from the underlying spectra map.
    gc_key = next(
        k for k in keys if k.comp_i == 0 and k.comp_j == 1 and k.kind is SpectrumKind.GC
    )
    cg_key = next(
        k for k in keys if k.comp_i == 0 and k.comp_j == 1 and k.kind is SpectrumKind.CG
    )
    assert isinstance(cl_dict[gc_key], np.ndarray)
    assert isinstance(cl_dict[cg_key], np.ndarray)


def test_directional_has_one_more_cross_spectrum_per_qu_qu_pair(collection_QU_QU):
    """DIRECTIONAL keeps the underlying CG entry that SYMMETRIC filters out.
    For two QU components, that's +1 spectrum (the cross-pair CG)."""
    from cosmocore.spectrum_key import SymmetryMode

    sm = SpectraManager(collection_QU_QU)
    _populate_fiducial_cls(sm)

    _, keys_sym = sm.build_inputs(symmetry_mode=SymmetryMode.SYMMETRIC)
    _, keys_dir = sm.build_inputs(symmetry_mode=SymmetryMode.DIRECTIONAL)

    assert len(keys_dir) == len(keys_sym) + 1
    extra_keys = set(keys_dir) - set(keys_sym)
    assert len(extra_keys) == 1
    (extra,) = extra_keys
    assert extra.kind is SpectrumKind.CG and extra.comp_i != extra.comp_j


class _ConstantBeams:
    """Beam-manager stand-in: a constant beam per field-map label."""

    def __init__(self, values, lmax):
        self._beams = {k: np.full(lmax + 1, v) for k, v in values.items()}

    def get_beam_dict(self):
        return dict(self._beams)


@pytest.fixture
def collection_two_qu():
    """Two spin-2 components with multi-character field-map labels."""
    nside, lmax = 4, 8
    mask = np.ones(hp.nside2npix(nside), dtype=np.float64)
    return [
        create_field(spin=2, nside=nside, lmax=lmax, mask=mask, labels=["E1", "B1"]),
        create_field(spin=2, nside=nside, lmax=lmax, mask=mask, labels=["E2", "B2"]),
    ]


def test_smoothing_factors_multi_character_labels(collection_two_qu):
    """Each factor pairs the beams of its two field-map labels, whatever their length."""
    sm = SpectraManager(collection_two_qu)
    b = {"E1": 2.0, "B1": 3.0, "E2": 5.0, "B2": 7.0}
    factors = sm.compute_smoothing_factors(_ConstantBeams(b, 8))
    for label in sm.labels:
        pair = next((x, y) for x in b for y in b if x + y == label)
        np.testing.assert_array_equal(factors[label], b[pair[0]] * b[pair[1]])


@pytest.mark.parametrize("mode", ["SYMMETRIC", "DIRECTIONAL"])
def test_label_for_key_names_each_emitted_key(collection_two_qu, mode):
    """label_for_key maps every build_inputs key to its own spectrum label."""
    sm = SpectraManager(collection_two_qu)
    _populate_fiducial_cls(sm)
    _, keys = sm.build_inputs(symmetry_mode=mode)
    labels = [sm.label_for_key(k) for k in keys]
    expected = ["E1E1", "B1B1", "E1B1", "E2E2", "B2B2", "E2B2", "E1E2", "E1B2"]
    expected += ["B1E2", "B1B2"] if mode == "DIRECTIONAL" else ["B1B2"]
    assert labels == expected
