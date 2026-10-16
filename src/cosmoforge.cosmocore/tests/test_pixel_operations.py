"""Test pixel operations functionality from cosmocore."""

import numpy as np
import pytest

from cosmocore import compute_pointings


def test_compute_pointings():
    """RING and NESTED both give unit vectors, and differ from each other."""
    nside = 2
    active = [np.array([0, 5, 10])]

    result, _, _ = compute_pointings(nside, active, "RING")
    vectors = result[0]
    assert vectors.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-10)

    result_nested, _, _ = compute_pointings(nside, active, "NESTED")
    vectors_nested = result_nested[0]
    assert vectors_nested.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(vectors_nested, axis=1), 1.0, atol=1e-10)
    assert not np.allclose(vectors, vectors_nested)


def test_count_nonzero_mask():
    """Deprecated in 1.3.0, removed in 1.4.0; behaviour pinned until then."""
    import warnings

    from cosmocore.signal_kernels import count_nonzero_mask

    with pytest.warns(DeprecationWarning, match="active_pixels"):
        count_nonzero_mask(np.ones(12))
    warnings.simplefilter("ignore", DeprecationWarning)

    # Test counting non-zero pixels in mask
    nside = 4
    npix = 12 * nside**2
    mask = np.ones(npix, dtype=np.float64)

    # All pixels are 1 (active), should count all
    count = count_nonzero_mask(mask)
    assert count == npix

    # Test with half pixels masked
    mask_half = np.zeros(npix, dtype=np.float64)
    mask_half[: npix // 2] = 1.0
    count_half = count_nonzero_mask(mask_half)
    assert count_half == npix // 2

    # Test with all zeros
    mask_zeros = np.zeros(npix, dtype=np.float64)
    count_zeros = count_nonzero_mask(mask_zeros)
    assert count_zeros == 0


def test_wigner_d_matrix_is_deprecated_and_still_agrees_with_wigner_d_small():
    from cosmocore.basics import wigner_d_matrix, wigner_d_small

    ell, s, theta = 3, 2, 0.7
    out = np.empty(2 * ell + 1)
    with pytest.warns(DeprecationWarning, match="wigner_d_small"):
        wigner_d_matrix(ell, s, np.cos(theta), np.sin(theta), out)
    expected = [
        wigner_d_small(ell, m, s, np.cos(theta), np.sin(theta))
        for m in range(-ell, ell + 1)
    ]
    np.testing.assert_array_equal(out, expected)
