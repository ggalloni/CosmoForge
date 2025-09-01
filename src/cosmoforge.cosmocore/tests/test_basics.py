import numpy as np
import pytest

from cosmocore import (
    idx2spec,
    spec2idx,
)
from cosmocore.basics import _project_and_norm


def test_idx2spec():
    """Test inverse spectrum index conversion."""
    nfields = 3

    # Test auto-spectra indices
    assert idx2spec(0, nfields) == (0, 0)
    assert idx2spec(1, nfields) == (1, 1)
    assert idx2spec(2, nfields) == (2, 2)

    # Test cross-spectra indices (this will cover lines 228-233)
    # For nfields=3, cross indices start at 3
    assert idx2spec(3, nfields) == (0, 1)  # First cross spectrum
    assert idx2spec(4, nfields) == (0, 2)  # Second cross spectrum
    assert idx2spec(5, nfields) == (1, 2)  # Third cross spectrum

    # Test with more fields to exercise the while loop
    nfields = 4
    assert idx2spec(4, nfields) == (0, 1)  # First cross
    assert idx2spec(5, nfields) == (0, 2)  # Second cross
    assert idx2spec(6, nfields) == (0, 3)  # Third cross
    assert idx2spec(7, nfields) == (1, 2)  # Fourth cross
    assert idx2spec(8, nfields) == (1, 3)  # Fifth cross
    assert idx2spec(9, nfields) == (2, 3)  # Sixth cross

    # Test error case - out of bounds (covers line 192)
    with pytest.raises(ValueError, match="Index .* out of bounds"):
        idx2spec(100, nfields)


def test_spec2idx_idx2spec_consistency():
    """Test that spec2idx and idx2spec are inverse operations."""
    nfields = 3

    # Test auto-spectra
    for i in range(nfields):
        idx = spec2idx(i, i, nfields)
        assert idx2spec(idx, nfields) == (i, i)

    # Test cross-spectra
    for i in range(nfields):
        for j in range(i + 1, nfields):
            idx = spec2idx(i, j, nfields)
            assert idx2spec(idx, nfields) == (i, j)
            idx = spec2idx(j, i, nfields)
            assert idx2spec(idx, nfields) == (i, j)


def test_project_and_norm():
    """Test the _project_and_norm function to cover epsilon bump logic."""
    # Test normal case
    vx, vy, vz = 1.0, 1.0, 1.0
    px, py, pz = _project_and_norm(vx, vy, vz)

    # Result should be normalized
    norm = np.sqrt(px * px + py * py + pz * pz)
    assert abs(norm - 1.0) < 1e-10

    # Test edge case where projection gives near-zero norm (covers lines 271-274)
    # When v is parallel to z-axis, cross product with z gives near-zero
    vx, vy, vz = 0.0, 0.0, 1.0  # Parallel to z-axis
    px, py, pz = _project_and_norm(vx, vy, vz)

    # Should still be normalized after epsilon bump
    norm = np.sqrt(px * px + py * py + pz * pz)
    assert abs(norm - 1.0) < 1e-10
    assert pz == 0.0  # z-component should be zero (projection onto xy-plane)
