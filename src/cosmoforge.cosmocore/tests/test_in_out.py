"""Opt-in persistence (ADR-0015): the write helpers no-op on a falsy path.

A falsy path (``None`` or ``""``) means "do not persist". A real path writes as
before. The gate lives in the helpers so every present and future call site is
covered uniformly.
"""

import healpy as hp
import numpy as np
import pytest

from cosmocore.in_out import (
    output_geometry,
    read_mask,
    write_covmat_reduced,
    write_out_matrix,
    writecl,
)

# Each helper paired with the extra (non-path) args that make it write.
HELPERS = [
    (write_covmat_reduced, (np.eye(3),)),
    (write_out_matrix, (np.eye(3),)),
    (writecl, (np.arange(6.0).reshape(3, 2),)),
    (output_geometry, ([2], [np.zeros((2, 3))], np.array([[0, 1]]))),
]


@pytest.mark.parametrize("helper, args", HELPERS)
@pytest.mark.parametrize("path", [None, "", "   "])
def test_helper_noop_on_falsy_path(helper, args, path, tmp_path, monkeypatch):
    """A None/empty/blank path writes nothing and does not raise."""
    monkeypatch.chdir(tmp_path)
    helper(path, *args)
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize("helper, args", HELPERS)
def test_helper_writes_on_real_path(helper, args, tmp_path):
    """A real path still produces a non-empty file."""
    target = tmp_path / "artifact.out"
    helper(str(target), *args)
    assert target.exists()
    assert target.stat().st_size > 0


def test_read_mask_honours_nested_ordering(tmp_path):
    """A mask written NESTED and read with nest=True round-trips unchanged.

    Without the nest flag, hp.read_map force-converts to RING, permuting the
    pixels and silently delivering wrong sky positions (ADR-0017 §Decision.6).
    """
    nside = 2
    npix = hp.nside2npix(nside)
    original = np.arange(npix, dtype=np.float64)  # distinct per pixel
    target = tmp_path / "mask_nest.fits"
    hp.write_map(str(target), original, nest=True, dtype=np.float64)

    buf = np.empty((1, npix), dtype=np.float64)
    result = read_mask(str(target), buf, nest=True)

    np.testing.assert_array_equal(result.ravel(), original)
