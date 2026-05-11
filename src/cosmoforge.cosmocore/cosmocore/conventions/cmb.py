"""CMB-friendly aliases for SpectrumKind, plus the post-hoc canonicalisation
helper that re-keys output dicts to T-first ordering and label-conversion
helpers for user-facing output.

Slots map to CMB letters: S->T, G->E, C->B.
"""

from __future__ import annotations

import numpy as np

from cosmocore.spectrum_key import Slot, SpectrumKey, SpectrumKind

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


# Slot index within a single component: which entry in ``params.labels``
# (relative to that component's offset) does each slot occupy?
_SLOT_INDEX_WITHIN_COMPONENT = {Slot.S: 0, Slot.G: 0, Slot.C: 1}


def _slot_offsets(spins) -> list[int]:
    """Cumulative slot count before each component — maps comp_i to the
    starting index in a flat ``labels`` list."""
    offsets = []
    cum = 0
    for s in spins:
        offsets.append(cum)
        cum += 2 if s == 2 else 1
    return offsets


def spectrum_key_to_label(key: SpectrumKey, *, labels, spins, separator: str = "") -> str:
    """Convert a SpectrumKey to a concatenated physical label.

    Parameters
    ----------
    key : SpectrumKey
        The spectrum identifier.
    labels : sequence[str]
        Per-slot label list — typically ``params.labels``. Must contain
        one entry per slot across all components (e.g. ``["T", "E", "B"]``
        for ``spins=(0, 2)``).
    spins : sequence[int]
        Per-component spin list (e.g. ``(0, 2)`` for TQU).
    separator : str, optional
        Separator inserted between the two slot labels. Default empty
        string — yields ``"TT"``, ``"EE"``, ``"E1B2"``, etc. Pass ``"x"``
        for ``"T1xT2"`` etc.

    Returns
    -------
    str
        ``labels[i] + separator + labels[j]`` where ``i`` and ``j`` are
        the slot indices selected by ``key``.

    Examples
    --------
    >>> spins = (0, 2)
    >>> labels = ["T", "E", "B"]
    >>> spectrum_key_to_label(
    ...     SpectrumKey(0, 0, SpectrumKind.SS, spins=spins),
    ...     labels=labels, spins=spins
    ... )
    'TT'
    >>> spectrum_key_to_label(
    ...     SpectrumKey(1, 1, SpectrumKind.GC, spins=spins),
    ...     labels=labels, spins=spins
    ... )
    'EB'
    """
    offsets = _slot_offsets(spins)
    slot_i, slot_j = key.kind.slots
    idx_i = offsets[key.comp_i] + _SLOT_INDEX_WITHIN_COMPONENT[slot_i]
    idx_j = offsets[key.comp_j] + _SLOT_INDEX_WITHIN_COMPONENT[slot_j]
    return f"{labels[idx_i]}{separator}{labels[idx_j]}"


def to_label_dict(source, *, labels, spins, spectra_list=None, n_bins=None):
    """Convert a SpectrumKey-keyed dict or a flat array into a label-keyed dict.

    Parameters
    ----------
    source : dict or numpy.ndarray
        Either a SpectrumKey-keyed dict (``{key: value, ...}``) or a flat
        array. For a 1-D array of shape ``(n_spectra * n_bins,)`` the
        result entries have shape ``(n_bins,)``. For a 2-D array of shape
        ``(n_sims, n_spectra * n_bins)`` the result entries have shape
        ``(n_sims, n_bins)``. The last axis is partitioned into
        ``n_spectra`` contiguous blocks of ``n_bins`` columns each, in
        the order given by ``spectra_list``.
    labels : sequence[str]
        Per-slot label list (typically ``params.labels``).
    spins : sequence[int]
        Per-component spin list.
    spectra_list : list[SpectrumKey], optional
        Required when ``source`` is an array; ignored when ``source`` is a
        dict. Specifies the spectrum order in the flat layout.
    n_bins : int, optional
        Required when ``source`` is an array. Number of bins per spectrum.

    Returns
    -------
    dict[str, Any]
        Label-keyed dict. For multi-frequency / multi-component setups
        the user's per-slot labels (e.g. ``["T100", "T143"]``) make the
        result keys unambiguous (``"T100T143"`` etc.).
    """
    if isinstance(source, dict):
        return {
            spectrum_key_to_label(k, labels=labels, spins=spins): v
            for k, v in source.items()
        }

    if spectra_list is None or n_bins is None:
        raise ValueError("spectra_list and n_bins are required when source is an array")

    array = np.asarray(source)
    out: dict[str, np.ndarray] = {}
    for spec_idx, key in enumerate(spectra_list):
        label = spectrum_key_to_label(key, labels=labels, spins=spins)
        start = spec_idx * n_bins
        end = start + n_bins
        if array.ndim == 1:
            out[label] = array[start:end]
        else:
            out[label] = array[..., start:end]
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
    "spectrum_key_to_label",
    "to_cmb_canonical",
    "to_label_dict",
]
