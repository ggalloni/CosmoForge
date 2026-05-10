from cosmocore.conventions.cmb import BB, BE, BT, EB, EE, ET, TB, TE, TT
from cosmocore.spectrum_key import SpectrumKind


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
