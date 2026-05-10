from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class SpectrumKey:
    """Identifier for one cross/auto power spectrum.

    Passive identifier — does not symmetrise, canonicalise, or perform
    algebra. Constructor validates that `kind`'s required spins match the
    actual spins at `comp_i` and `comp_j`.
    """

    comp_i: int
    comp_j: int
    kind: SpectrumKind

    def __init__(
        self,
        comp_i: int,
        comp_j: int,
        kind: SpectrumKind,
        *,
        spins: tuple[int, ...] | None = None,
    ):
        object.__setattr__(self, "comp_i", comp_i)
        object.__setattr__(self, "comp_j", comp_j)
        object.__setattr__(self, "kind", kind)
        if spins is not None:
            self._validate(spins)

    def _validate(self, spins: tuple[int, ...]) -> None:
        required_i, required_j = self.kind.required_spins
        actual_i = spins[self.comp_i]
        actual_j = spins[self.comp_j]
        if actual_i != required_i:
            raise ValueError(
                f"kind {self.kind.name} requires comp_i to have spin "
                f"{required_i}; comp_{self.comp_i} has spin {actual_i}"
            )
        if actual_j != required_j:
            raise ValueError(
                f"kind {self.kind.name} requires comp_j to have spin "
                f"{required_j}; comp_{self.comp_j} has spin {actual_j}"
            )
