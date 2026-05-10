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
