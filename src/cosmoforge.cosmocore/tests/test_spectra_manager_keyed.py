"""Tests for SpectraManager.build_keyed_inputs (Slice 1, Task 1.6)."""

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


def test_build_keyed_inputs_emits_spectrum_keys(collection_T_QU):
    sm = SpectraManager(collection_T_QU)
    _populate_fiducial_cls(sm)

    cl_dict, keys = sm.build_keyed_inputs()

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


def test_build_keyed_inputs_dict_keyed_by_spectrum_key(collection_T_QU):
    sm = SpectraManager(collection_T_QU)
    _populate_fiducial_cls(sm)

    cl_dict, keys = sm.build_keyed_inputs()
    for key in keys:
        assert key in cl_dict
        assert isinstance(cl_dict[key], np.ndarray)
