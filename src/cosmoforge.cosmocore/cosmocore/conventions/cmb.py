"""CMB-friendly aliases for SpectrumKind.

Slots map to CMB letters: S->T, G->E, C->B.
"""

from cosmocore.spectrum_key import SpectrumKind

TT = SpectrumKind.SS
EE = SpectrumKind.GG
BB = SpectrumKind.CC
EB = SpectrumKind.GC
BE = SpectrumKind.CG
TE = SpectrumKind.SG
ET = SpectrumKind.GS
TB = SpectrumKind.SC
BT = SpectrumKind.CS

__all__ = ["TT", "EE", "BB", "EB", "BE", "TE", "ET", "TB", "BT"]
