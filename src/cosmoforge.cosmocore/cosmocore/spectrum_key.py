from __future__ import annotations

from enum import Enum


class Slot(Enum):
    """Sub-mode within a spin-s component.

    S = scalar (spin-0 field's only slot; CMB alias: T).
    G = gradient (parity-even slot of a spin-2 field; CMB alias: E).
    C = curl (parity-odd slot of a spin-2 field; CMB alias: B).
    """

    S = ("S", 0)
    G = ("G", 2)
    C = ("C", 2)

    def __init__(self, label: str, spin: int):
        self.label = label
        self.spin = spin


class SpectrumKind(Enum):
    """Directional spectrum kind: an ordered slot pair (slot_i, slot_j).

    Nine values cover every (slot_i, slot_j) reachable from spin-0 and spin-2
    components. Asymmetry is preserved (SG != GS, GC != CG); whether to
    collapse to a symmetric average is a separate decision driven by the
    SymmetryMode flag on Spectra/Fisher.
    """

    SS = (Slot.S, Slot.S)
    GG = (Slot.G, Slot.G)
    CC = (Slot.C, Slot.C)
    GC = (Slot.G, Slot.C)
    CG = (Slot.C, Slot.G)
    SG = (Slot.S, Slot.G)
    GS = (Slot.G, Slot.S)
    SC = (Slot.S, Slot.C)
    CS = (Slot.C, Slot.S)

    def __init__(self, slot_i: Slot, slot_j: Slot):
        self.slots = (slot_i, slot_j)

    @property
    def required_spins(self) -> tuple[int, int]:
        return (self.slots[0].spin, self.slots[1].spin)
