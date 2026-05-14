from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SymmetryMode(Enum):
    """How cross-component spin-2 × spin-2 EB-like spectra are treated.

    SYMMETRIC (default) — emits one combined GC spectrum per cross-pair; the
    Lambda model uses a single C_EB in both off-diagonal blocks. Standard
    cosmology (C_EB = 0) makes this numerically identical to DIRECTIONAL.

    DIRECTIONAL — emits both GC (E_i × B_j) and CG (B_i × E_j) as separate
    spectra. Lambda uses C_GC and C_CG independently. Opt-in for polarisation
    angle calibration diagnostics and parity-violation searches. See ADR-0011.
    """

    SYMMETRIC = "symmetric"
    DIRECTIONAL = "directional"


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


def _spin_pair_mode_to_kind(
    spin_i: int, spin_j: int, mode: int, *, is_cross: bool = False
):
    """Translate a legacy (spin_i, spin_j, mode) triple to a SpectrumKind.

    The mode→kind mapping for spin-2 × spin-2 depends on whether the
    pair is an auto-pair or a cross-component pair, because the underlying
    label enumeration differs:

    - Auto (i == j): PolarizationField.get_spectrum_labels returns
      ``[EE, BB, EB]`` → 3 modes, mapped to ``[GG, CC, GC]``.
    - Cross (i != j): PolarizationField.get_cross_spectrum_labels returns
      ``[E_iE_j, E_iB_j, B_iE_j, B_iB_j]`` → 4 modes, mapped to
      ``[GG, GC, CG, CC]``.

    All other spin pairs have a single mode count and ordering.
    """
    if (spin_i, spin_j) == (0, 0):
        return SpectrumKind.SS
    if (spin_i, spin_j) == (2, 2):
        if is_cross:
            return [
                SpectrumKind.GG,
                SpectrumKind.GC,
                SpectrumKind.CG,
                SpectrumKind.CC,
            ][mode]
        return [SpectrumKind.GG, SpectrumKind.CC, SpectrumKind.GC][mode]
    if (spin_i, spin_j) == (0, 2):
        return [SpectrumKind.SG, SpectrumKind.SC][mode]
    if (spin_i, spin_j) == (2, 0):
        return [SpectrumKind.GS, SpectrumKind.CS][mode]
    raise ValueError(f"unsupported spin pair ({spin_i}, {spin_j})")


def kind_to_legacy_mode(kind: SpectrumKind, *, is_cross: bool = False) -> int:
    """Bridge for migration: map SpectrumKind to the legacy int `mode`.

    For spin-2 × spin-2 the mode→kind mapping is context-dependent and must
    match ``_spin_pair_mode_to_kind`` (above) in this module:

    - Auto-pair (``is_cross=False``, default): ``[GG=0, CC=1, GC=2]``. CG has
      no slot here — auto-pair derivative builders emit the symmetrised GC
      matrix and never distinguish E_i B_j from B_i E_j.
    - Cross-pair (``is_cross=True``): ``[GG=0, GC=1, CG=2, CC=3]`` — the
      4-entry ordering used by ``PolarizationField.get_cross_spectrum_labels``
      and DIRECTIONAL-mode derivative construction.

    All other spin pairs have a single ordering independent of ``is_cross``.
    """
    spin_i, spin_j = kind.required_spins
    if (spin_i, spin_j) == (0, 0):
        return 0
    if (spin_i, spin_j) == (2, 2):
        if is_cross:
            try:
                return {
                    SpectrumKind.GG: 0,
                    SpectrumKind.GC: 1,
                    SpectrumKind.CG: 2,
                    SpectrumKind.CC: 3,
                }[kind]
            except KeyError:
                raise ValueError(
                    f"kind {kind.name} not valid for spin-2 × spin-2 cross-pair"
                )
        try:
            return {SpectrumKind.GG: 0, SpectrumKind.CC: 1, SpectrumKind.GC: 2}[kind]
        except KeyError:
            raise NotImplementedError(
                f"kind {kind.name} not supported in auto-pair int-mode encoding "
                "(CG is cross-pair only; pass is_cross=True)"
            )
    if (spin_i, spin_j) == (0, 2):
        return {SpectrumKind.SG: 0, SpectrumKind.SC: 1}[kind]
    if (spin_i, spin_j) == (2, 0):
        return {SpectrumKind.GS: 0, SpectrumKind.CS: 1}[kind]
    raise ValueError(kind)
