"""Active pixels and the active-pixel index.

The QML estimator works on *active* (unmasked) pixels: the noise covariance and
the observed maps are **reduced** to them and ordered by the active-pixel index.
Injecting either object (ADR-0017) therefore means knowing that ordering, so it
lives here — a pure function of the mask, importable and runnable without the
framework, so a caller never has to build a ``Fisher`` to learn it.

See CONTEXT.md ("Sky geometry") for the vocabulary; ADR-0017 for the seam.
"""

import numpy as np

__all__ = ["ACTIVE_THRESHOLD", "active_pixels", "active_pixel_index"]

#: A pixel is active where its mask value exceeds this. Apodized masks are
#: thresholded, not weighted.
ACTIVE_THRESHOLD = 0.5


def active_pixels(mask: np.ndarray) -> list[np.ndarray]:
    """Return the active pixels of each component, in column order.

    Parameters
    ----------
    mask : numpy.ndarray
        Analysis mask, shape ``(npix, ncomponents)`` — one column per component
        (T, Q, U), pixel-indexed per the analysis ordering. A 1-D array is
        treated as a single column.

    Returns
    -------
    list of numpy.ndarray
        One ``intp`` index array per component, sorted ascending.

    Examples
    --------
    >>> mask = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> [a.tolist() for a in active_pixels(mask)]
    [[0, 2], [1, 2]]
    """
    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim == 1:
        mask = mask[:, np.newaxis]
    return [np.flatnonzero(mask[:, i] > ACTIVE_THRESHOLD) for i in range(mask.shape[1])]


def active_pixel_index(mask: np.ndarray) -> np.ndarray:
    """Return the active-pixel index: the ordering of every reduced array.

    Each row of a reduced array corresponds to one entry of this index, which
    gives that row's position in the flattened full-pixel ``(ncomponents *
    npix,)`` vector. Use it to reduce your own arrays without constructing a
    ``Fisher``::

        index = active_pixel_index(mask)
        maps = maps_full.reshape(ncomponents * npix, nsims)[index]
        cov = cov_full[np.ix_(index, index)]

    Parameters
    ----------
    mask : numpy.ndarray
        Analysis mask, shape ``(npix, ncomponents)``. See :func:`active_pixels`.

    Returns
    -------
    numpy.ndarray
        1-D ``intp`` array of length ``n_active`` (the total across components).

    Examples
    --------
    >>> mask = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> active_pixel_index(mask).tolist()
    [0, 2, 4, 5]
    """
    npix = np.shape(mask)[0]
    return np.concatenate(
        [component + i * npix for i, component in enumerate(active_pixels(mask))]
    )
