"""The active-pixel index (ADR-0017): the published ordering IS the ordering."""

import healpy as hp
import numpy as np
import pytest

from cosmocore import active_pixel_index, active_pixels
from cosmocore.core import Core
from cosmocore.settings import InputParams


class ConcreteCore(Core):
    def compute(self):
        return None

    def run(self):
        return None


def _params(nside, spins, labels):
    params = InputParams()
    params.nside = nside
    params.lmax = 8
    params.lmax_signal = 16
    params.spins = spins
    params.labels = labels
    params.nfields = len(labels)
    params.ordering = "RING"
    return params


def _toy_mask():
    """nside=1 (npix=12): component 0 active at [0, 2], component 1 at [1, 2]."""
    mask = np.zeros((12, 2))
    mask[[0, 2], 0] = 1.0
    mask[[1, 2], 1] = 1.0
    return mask


def test_active_pixels_per_component():
    per_component = active_pixels(_toy_mask())

    assert len(per_component) == 2
    np.testing.assert_array_equal(per_component[0], [0, 2])
    np.testing.assert_array_equal(per_component[1], [1, 2])


def test_active_pixel_index_offsets_each_component_by_npix():
    # npix = 12, so component 1's pixels land at 12 + [1, 2] = [13, 14].
    index = active_pixel_index(_toy_mask())
    np.testing.assert_array_equal(index, [0, 2, 13, 14])
    assert index.dtype == np.intp


def test_transposed_mask_is_refused():
    """A (ncomponents, npix) mask would silently reduce to the wrong pixels: it
    yields an index of the same length with plausible values, so no downstream
    shape check catches it. The row count must be a valid HEALPix npix."""
    transposed = _toy_mask().T  # (2, 12)

    with pytest.raises(ValueError, match="not a valid HEALPix npix|transposed"):
        active_pixel_index(transposed)
    with pytest.raises(ValueError, match="not a valid HEALPix npix|transposed"):
        active_pixels(transposed)


def test_1d_mask_is_treated_as_a_single_component():
    mask_1d = np.zeros(12)  # nside=1
    mask_1d[[0, 2]] = 1.0

    per_component = active_pixels(mask_1d)
    assert len(per_component) == 1
    np.testing.assert_array_equal(per_component[0], [0, 2])
    # A single component means no offset is applied.
    np.testing.assert_array_equal(active_pixel_index(mask_1d), [0, 2])
    np.testing.assert_array_equal(
        active_pixel_index(mask_1d), active_pixel_index(mask_1d[:, np.newaxis])
    )


def test_active_pixels_thresholds_at_half():
    mask = np.zeros(12)  # nside=1
    mask[[4, 5, 6]] = [0.5, 0.51, 1.0]  # 0.5 is NOT active (strict >)
    np.testing.assert_array_equal(active_pixels(mask)[0], [5, 6])


def test_published_index_matches_the_frameworks_ordering():
    """The whole point of ADR-0017: a caller reducing with the public index gets
    exactly the ordering Core builds internally."""
    nside = 8
    npix = hp.nside2npix(nside)

    rng = np.random.default_rng(0)
    mask_temp = (rng.random(npix) > 0.4).astype(np.float64)
    mask_pol = (rng.random(npix) > 0.6).astype(np.float64)
    mask = np.column_stack([mask_temp, mask_pol, mask_pol])

    core = ConcreteCore(_params(nside, [0, 2], ["T", "Q", "U"]), mask=mask)
    core.setup_fields()
    core.setup_geometry()

    framework = np.concatenate(
        [core.pixact[i] + i * npix for i in range(len(core.pixact))]
    )
    np.testing.assert_array_equal(active_pixel_index(mask), framework)


def test_different_temperature_and_polarisation_masks_are_supported():
    """Different T and P masks (the standard configuration) must not go ragged."""
    nside = 8
    npix = hp.nside2npix(nside)

    mask_temp = np.zeros(npix)
    mask_temp[: int(0.6 * npix)] = 1.0
    mask_pol = np.zeros(npix)
    mask_pol[: int(0.4 * npix)] = 1.0
    mask = np.column_stack([mask_temp, mask_pol, mask_pol])

    core = ConcreteCore(_params(nside, [0, 2], ["T", "Q", "U"]), mask=mask)
    core.setup_fields()
    pixact, point_vectors = core.setup_geometry()

    assert [len(p) for p in pixact] == [
        int(0.6 * npix),
        int(0.4 * npix),
        int(0.4 * npix),
    ]
    assert core.npixs == [int(0.6 * npix), int(0.4 * npix)]  # per FIELD
    assert len(point_vectors) == 2  # per FIELD


def test_spin2_mask_columns_must_agree():
    nside = 8
    npix = hp.nside2npix(nside)

    mask_q = np.ones(npix)
    mask_u = np.ones(npix)
    mask_u[0] = 0.0  # one pixel of difference is enough
    mask = np.column_stack([np.ones(npix), mask_q, mask_u])

    core = ConcreteCore(_params(nside, [0, 2], ["T", "Q", "U"]), mask=mask)
    with pytest.raises(ValueError, match="two components of one spin-2 field"):
        core.setup_fields()
