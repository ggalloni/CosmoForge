import numpy as np

from cosmocore.conventions.cmb import (
    BB,
    BE,
    BT,
    EB,
    EE,
    ET,
    TB,
    TE,
    TT,
    spectrum_key_to_label,
    to_cmb_canonical,
    to_label_dict,
)
from cosmocore.spectrum_key import SpectrumKey, SpectrumKind


def test_cmb_aliases_map_to_kinds():
    assert TT is SpectrumKind.SS
    assert EE is SpectrumKind.GG
    assert BB is SpectrumKind.CC
    assert EB is SpectrumKind.GC
    assert BE is SpectrumKind.CG
    assert TE is SpectrumKind.SG
    assert ET is SpectrumKind.GS
    assert TB is SpectrumKind.SC
    assert BT is SpectrumKind.CS


def test_to_cmb_canonical_swaps_qu_t_to_t_qu():
    """Declaration is [QU, T] -> cross is keyed (0, 1, GS) i.e. ET.
    to_cmb_canonical re-keys to (1, 0, SG) i.e. TE."""
    spins = (2, 0)
    raw_key = SpectrumKey(0, 1, SpectrumKind.GS, spins=spins)
    raw = {raw_key: np.array([1.0, 2.0, 3.0])}
    canonical = to_cmb_canonical(raw, spins=spins)
    expected_key = SpectrumKey(1, 0, SpectrumKind.SG, spins=spins)
    assert expected_key in canonical
    np.testing.assert_array_equal(canonical[expected_key], np.array([1.0, 2.0, 3.0]))


def test_to_cmb_canonical_passthrough_when_already_canonical():
    spins = (0, 2)
    key = SpectrumKey(0, 1, SpectrumKind.SG, spins=spins)
    raw = {key: np.array([1.0])}
    canonical = to_cmb_canonical(raw, spins=spins)
    assert canonical[key] is raw[key]


def test_to_cmb_canonical_swaps_cs_to_sc():
    """CS (cross-spin-2 to spin-0 polarisation) re-keys to SC (T to B)."""
    spins = (2, 0)
    raw_key = SpectrumKey(0, 1, SpectrumKind.CS, spins=spins)
    raw = {raw_key: np.array([7.0])}
    canonical = to_cmb_canonical(raw, spins=spins)
    expected_key = SpectrumKey(1, 0, SpectrumKind.SC, spins=spins)
    assert expected_key in canonical
    np.testing.assert_array_equal(canonical[expected_key], np.array([7.0]))


def test_to_cmb_canonical_passthrough_same_spin():
    """Same-spin pairs (TT, EE/BB/EB, etc.) pass through unchanged."""
    spins = (2, 2)
    keys = [
        SpectrumKey(0, 0, SpectrumKind.GG, spins=spins),
        SpectrumKey(0, 1, SpectrumKind.GC, spins=spins),
        SpectrumKey(0, 1, SpectrumKind.CC, spins=spins),
    ]
    raw = {k: np.array([1.0]) for k in keys}
    canonical = to_cmb_canonical(raw, spins=spins)
    for k in keys:
        assert k in canonical
        assert canonical[k] is raw[k]


def test_spectrum_key_to_label_tqu():
    """TQU labels: T at slot 0, E at slot 1, B at slot 2."""
    spins = (0, 2)
    labels = ["T", "E", "B"]
    expect = {
        (0, 0, SpectrumKind.SS): "TT",
        (1, 1, SpectrumKind.GG): "EE",
        (1, 1, SpectrumKind.CC): "BB",
        (1, 1, SpectrumKind.GC): "EB",
        (0, 1, SpectrumKind.SG): "TE",
        (0, 1, SpectrumKind.SC): "TB",
    }
    for (ci, cj, kind), label in expect.items():
        key = SpectrumKey(ci, cj, kind, spins=spins)
        assert spectrum_key_to_label(key, labels=labels, spins=spins) == label


def test_spectrum_key_to_label_multi_frequency():
    """Multi-frequency labels (T1/T2) yield unambiguous concatenated keys."""
    spins = (0, 0)
    labels = ["T1", "T2"]
    key = SpectrumKey(0, 1, SpectrumKind.SS, spins=spins)
    assert spectrum_key_to_label(key, labels=labels, spins=spins) == "T1T2"
    assert (
        spectrum_key_to_label(key, labels=labels, spins=spins, separator="x") == "T1xT2"
    )


def test_to_label_dict_from_spectrum_key_dict():
    spins = (0, 2)
    labels = ["T", "E", "B"]
    key_tt = SpectrumKey(0, 0, SpectrumKind.SS, spins=spins)
    key_ee = SpectrumKey(1, 1, SpectrumKind.GG, spins=spins)
    source = {key_tt: np.array([1.0, 2.0]), key_ee: np.array([3.0, 4.0])}
    out = to_label_dict(source, labels=labels, spins=spins)
    np.testing.assert_array_equal(out["TT"], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(out["EE"], np.array([3.0, 4.0]))


def test_to_label_dict_from_flat_array():
    """Flat array with spectra_list ordering becomes label-keyed dict.

    For TQU under SYMMETRIC mode, the order is TT, EE, BB, EB, TE, TB."""
    spins = (0, 2)
    labels = ["T", "E", "B"]
    nbins = 3
    spectra_list = [
        SpectrumKey(0, 0, SpectrumKind.SS, spins=spins),  # TT
        SpectrumKey(1, 1, SpectrumKind.GG, spins=spins),  # EE
        SpectrumKey(1, 1, SpectrumKind.CC, spins=spins),  # BB
        SpectrumKey(1, 1, SpectrumKind.GC, spins=spins),  # EB
        SpectrumKey(0, 1, SpectrumKind.SG, spins=spins),  # TE
        SpectrumKey(0, 1, SpectrumKind.SC, spins=spins),  # TB
    ]
    # 1D array, length 6 * 3 = 18, values 0..17
    flat = np.arange(18, dtype=float)
    out = to_label_dict(
        flat, labels=labels, spins=spins, spectra_list=spectra_list, n_bins=nbins
    )
    np.testing.assert_array_equal(out["TT"], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(out["EE"], [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(out["TB"], [15.0, 16.0, 17.0])

    # 2D array (n_sims, n_params) → per-key shape (n_sims, n_bins)
    flat2d = np.arange(2 * 18, dtype=float).reshape(2, 18)
    out2d = to_label_dict(
        flat2d, labels=labels, spins=spins, spectra_list=spectra_list, n_bins=nbins
    )
    assert out2d["TT"].shape == (2, 3)
    np.testing.assert_array_equal(out2d["TT"], flat2d[:, 0:3])
    np.testing.assert_array_equal(out2d["TB"], flat2d[:, 15:18])


def test_to_label_dict_array_requires_spectra_list_and_n_bins():
    import pytest

    flat = np.arange(6)
    with pytest.raises(ValueError, match="spectra_list and n_bins"):
        to_label_dict(flat, labels=["T"], spins=(0,))
