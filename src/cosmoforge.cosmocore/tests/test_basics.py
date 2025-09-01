import pytest

from cosmocore import (
    idx2spec,
)


def test_idx2spec():
    """Test inverse spectrum index conversion."""
    nfields = 3

    # Test auto-spectra indices
    assert idx2spec(0, nfields) == (0, 0)
    assert idx2spec(1, nfields) == (1, 1)
    assert idx2spec(2, nfields) == (2, 2)

    # Test error case - out of bounds
    with pytest.raises(ValueError, match="Index .* out of bounds"):
        idx2spec(100, nfields)
