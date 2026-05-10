import pytest

from cosmocore.spectrum_key import Slot, SpectrumKind


class TestSpectrumKind:
    def test_has_nine_directional_values(self):
        names = {k.name for k in SpectrumKind}
        assert names == {"SS", "GG", "CC", "GC", "CG", "SG", "GS", "SC", "CS"}

    def test_each_kind_exposes_slot_pair(self):
        assert SpectrumKind.SS.slots == (Slot.S, Slot.S)
        assert SpectrumKind.SG.slots == (Slot.S, Slot.G)
        assert SpectrumKind.GS.slots == (Slot.G, Slot.S)
        assert SpectrumKind.GC.slots == (Slot.G, Slot.C)
        assert SpectrumKind.CG.slots == (Slot.C, Slot.G)

    def test_each_slot_carries_required_spin(self):
        assert Slot.S.spin == 0
        assert Slot.G.spin == 2
        assert Slot.C.spin == 2

    def test_kind_required_spins_derived_from_slots(self):
        assert SpectrumKind.SG.required_spins == (0, 2)
        assert SpectrumKind.GS.required_spins == (2, 0)
        assert SpectrumKind.GC.required_spins == (2, 2)


from cosmocore.spectrum_key import SpectrumKey


class TestSpectrumKey:
    def test_constructs_with_valid_kind_for_spins(self):
        spins = (0, 2)
        key = SpectrumKey(comp_i=0, comp_j=1, kind=SpectrumKind.SG, spins=spins)
        assert key.comp_i == 0 and key.comp_j == 1
        assert key.kind is SpectrumKind.SG

    def test_rejects_kind_inconsistent_with_comp_i_spin(self):
        spins = (2, 0)
        with pytest.raises(ValueError, match="kind SG.*comp_i.*spin 0"):
            SpectrumKey(0, 1, SpectrumKind.SG, spins=spins)

    def test_rejects_kind_inconsistent_with_comp_j_spin(self):
        spins = (0, 0)
        with pytest.raises(ValueError, match="kind SG.*comp_j.*spin 2"):
            SpectrumKey(0, 1, SpectrumKind.SG, spins=spins)

    def test_is_frozen(self):
        key = SpectrumKey(0, 0, SpectrumKind.SS, spins=(0,))
        with pytest.raises(Exception):
            key.comp_i = 99

    def test_is_hashable_and_equal_by_value(self):
        spins = (0, 2)
        a = SpectrumKey(0, 1, SpectrumKind.SG, spins=spins)
        b = SpectrumKey(0, 1, SpectrumKind.SG, spins=spins)
        assert a == b and hash(a) == hash(b)
        assert {a: "v"}[b] == "v"


from cosmocore.spectrum_key import kind_to_legacy_mode


@pytest.mark.parametrize(
    "kind,expected_mode",
    [
        (SpectrumKind.SS, 0),
        (SpectrumKind.GG, 0),
        (SpectrumKind.CC, 1),
        (SpectrumKind.GC, 2),
        (SpectrumKind.SG, 0),
        (SpectrumKind.SC, 1),
        (SpectrumKind.GS, 0),
        (SpectrumKind.CS, 1),
    ],
)
def test_kind_to_legacy_mode(kind, expected_mode):
    assert kind_to_legacy_mode(kind) == expected_mode


def test_kind_to_legacy_mode_rejects_cg_until_directional_landed():
    # CG has no slot in today's int-mode encoding; raise until Slice 5 lands.
    with pytest.raises(NotImplementedError):
        kind_to_legacy_mode(SpectrumKind.CG)
