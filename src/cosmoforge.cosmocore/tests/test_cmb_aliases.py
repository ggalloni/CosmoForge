import numpy as np

from cosmocore.conventions.cmb import BB, BE, BT, EB, EE, ET, TB, TE, TT, to_cmb_canonical
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
