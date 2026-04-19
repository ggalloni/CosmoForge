"""Test pixel operations functionality from cosmocore."""

import numpy as np

from cosmocore import compute_pointings


def test_compute_pointings():
    """Test compute_pointings function."""
    nside = 2

    # Setup for single field
    npixs = [3]  # 3 active pixels in field 0
    active_pixels = np.array([[0, 5, 10]])  # Active pixel indices

    # Create empty point_vectors, theta_vectors, phi_vectors tuples
    point_vectors = (np.zeros((3, 3)),)  # (n_active, 3) for field 0
    theta_vectors = (np.zeros(3),)  # (n_active,) for field 0
    phi_vectors = (np.zeros(3),)  # (n_active,) for field 0

    # Test RING ordering
    result, _, _ = compute_pointings(
        nside, npixs, point_vectors, theta_vectors, phi_vectors, active_pixels, "RING"
    )

    # Check that we get normalized unit vectors
    vectors = result[0]
    assert vectors.shape == (3, 3)

    # Check normalization
    for i in range(3):
        norm = np.sqrt(np.sum(vectors[i, :] ** 2))
        assert abs(norm - 1.0) < 1e-10

    # Test NESTED ordering
    point_vectors_nested = (np.zeros((3, 3)),)
    theta_vectors_nested = (np.zeros(3),)
    phi_vectors_nested = (np.zeros(3),)
    result_nested, _, _ = compute_pointings(
        nside,
        npixs,
        point_vectors_nested,
        theta_vectors_nested,
        phi_vectors_nested,
        active_pixels,
        "NESTED",
    )

    # Should be different from RING (in general)
    vectors_nested = result_nested[0]
    assert vectors_nested.shape == (3, 3)

    # Check normalization for NESTED too
    for i in range(3):
        norm = np.sqrt(np.sum(vectors_nested[i, :] ** 2))
        assert abs(norm - 1.0) < 1e-10


def test_count_nonzero_mask():
    """Test count_nonzero_mask function."""
    from cosmocore.pixel import count_nonzero_mask

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
