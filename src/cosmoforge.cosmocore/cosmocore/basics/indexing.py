"""Spectrum index conversion utilities."""

from __future__ import annotations

from functools import lru_cache


@lru_cache
def spec2idx(i, j, nfields):
    """
    Convert field indices to spectrum index for compressed storage.

    Parameters
    ----------
    i, j : int
        Field indices for the spectrum.
    nfields : int
        Total number of fields.

    Returns
    -------
    int
        Linear index for spectrum storage in compressed format.

    Notes
    -----
    Auto-spectra (i==j) are stored first, followed by cross-spectra in
    upper triangular order. Uses LRU cache for performance.
    """
    if i == j:
        return i  # auto
    elif i < j:
        return nfields + (i * (2 * nfields - i - 1)) // 2 + (j - i - 1)
    else:
        return spec2idx(j, i, nfields)


@lru_cache
def idx2spec(idx, nfields):
    """
    Convert spectrum index back to field indices.

    Parameters
    ----------
    idx : int
        Linear spectrum index in compressed storage.
    nfields : int
        Total number of fields.

    Returns
    -------
    tuple of int
        Field indices (i, j) corresponding to the spectrum index.

    Raises
    ------
    ValueError
        If index is out of bounds for the given number of fields.

    Notes
    -----
    Inverse operation of spec2idx. Uses LRU cache for performance.
    """
    if idx < nfields:
        return idx, idx
    idx_cross = idx - nfields
    total_cross = nfields * (nfields - 1) // 2
    if idx_cross < 0 or idx_cross >= total_cross:
        raise ValueError(f"Index {idx} out of bounds for nfields={nfields}")
    i = 0
    while idx_cross >= nfields - i - 1:
        idx_cross -= nfields - i - 1
        i += 1
    j = i + idx_cross + 1
    return i, j
