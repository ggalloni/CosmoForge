"""CMB-friendly aliases for SpectrumKind, plus the post-hoc canonicalisation
helper that re-keys output dicts to T-first ordering.

Slots map to CMB letters: S->T, G->E, C->B.
"""

from cosmocore.spectrum_key import SpectrumKey, SpectrumKind

TT = SpectrumKind.SS
EE = SpectrumKind.GG
BB = SpectrumKind.CC
EB = SpectrumKind.GC
BE = SpectrumKind.CG
TE = SpectrumKind.SG
ET = SpectrumKind.GS
TB = SpectrumKind.SC
BT = SpectrumKind.CS

# Slot-pair swap used to flip a spin-2 x spin-0 key into spin-0 x spin-2
# ordering. Only the four mixed-spin kinds appear here; same-spin kinds
# (SS, GG, CC, GC, CG) never need flipping.
_KIND_SWAP = {
    SpectrumKind.GS: SpectrumKind.SG,
    SpectrumKind.CS: SpectrumKind.SC,
    SpectrumKind.SG: SpectrumKind.GS,
    SpectrumKind.SC: SpectrumKind.CS,
}


def to_cmb_canonical(result_dict, *, spins):
    """Re-key a result dict to CMB-conventional ordering (T-first, then E/B).

    For mixed-spin pairs (one spin-0 + one spin-2) where the declaration
    placed the spin-2 component first, the output key is swapped so the
    spin-0 (T) component appears first — matching the conventional CMB
    naming where TE / TB are written T-first. Same-spin pairs and pairs
    already in T-first order pass through unchanged.

    Parameters
    ----------
    result_dict : dict[SpectrumKey, Any]
        Output dictionary keyed by SpectrumKey.
    spins : tuple[int, ...]
        Spin of each component in the original field collection.

    Returns
    -------
    dict[SpectrumKey, Any]
        New dict with re-keyed entries; values are not copied.
    """
    out = {}
    for key, value in result_dict.items():
        spin_i = spins[key.comp_i]
        spin_j = spins[key.comp_j]
        if spin_i == 2 and spin_j == 0:
            new_kind = _KIND_SWAP.get(key.kind, key.kind)
            out[SpectrumKey(key.comp_j, key.comp_i, new_kind, spins=spins)] = value
        else:
            out[key] = value
    return out


__all__ = [
    "TT",
    "EE",
    "BB",
    "EB",
    "BE",
    "TE",
    "ET",
    "TB",
    "BT",
    "to_cmb_canonical",
]
