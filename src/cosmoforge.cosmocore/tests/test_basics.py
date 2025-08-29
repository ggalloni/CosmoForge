import numpy as np
import pytest

from cosmocore import (
    cross_index,
    idx2spec,
    invert_lower_triangular,
)


def test_cross_index():
    """Test cross-correlation index function."""
    # Test basic functionality
    result = cross_index(0, 1, 3)
    assert isinstance(result, int)

    # Test symmetry (should swap i, j if i > j)
    result1 = cross_index(0, 1, 3)
    result2 = cross_index(1, 0, 3)
    assert result1 == result2


def test_idx2spec():
    """Test inverse spectrum index conversion."""
    nfields = 3

    # Test auto-spectra
    assert idx2spec(0, nfields) == (0, 0)
    assert idx2spec(1, nfields) == (1, 1)
    assert idx2spec(2, nfields) == (2, 2)

    # Test error case - out of bounds
    with pytest.raises(ValueError, match="Index .* out of bounds"):
        idx2spec(100, nfields)


def test_invert_lower_triangular():
    """Test lower triangular matrix inversion."""
    # Create a simple 2x2 lower triangular matrix
    L = np.array([[2, 0], [1, 3]], dtype=np.float64)
    L_inv = invert_lower_triangular(L)

    # Verify that L @ L_inv = I (with reasonable tolerance)
    identity = L @ L_inv
    expected_identity = np.eye(2)
    np.testing.assert_allclose(identity, expected_identity, rtol=1e-12, atol=1e-12)
