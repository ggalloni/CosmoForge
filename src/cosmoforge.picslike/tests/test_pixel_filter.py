"""``pixel_filter=`` through PICSLike (ADR-0020)."""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest
import yaml

from cosmocore.filters import harmonic_deprojection, hits_weighting
from picslike import PICSLike


def _mask_of(config_path):
    """The config's own mask, widened to one column per component."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    m = np.asarray(hp.read_map(config["maskfile"]), float)
    return np.repeat(m[:, None], len(config["physical_labels"]), axis=1)


def _shifted_logl(pl):
    values = np.asarray(pl.likelihood_result.log_likelihood_values, float)
    return values - values.max()


def test_invertible_filter_leaves_the_likelihood_shape_alone(fast_config_path):
    """Restriction by an invertible map shifts ``ln|C|`` by a constant only.

    The likelihood itself moves, so this compares the grid's *shape*: the no-op
    control at the PICSLike seam, matching the one at the Fisher seam.
    """
    mask = _mask_of(fast_config_path)
    rng = np.random.default_rng(5)
    hits = rng.uniform(0.5, 2.0, mask.shape[0])
    with pytest.warns(UserWarning, match="kept every singular value"):
        weighting = hits_weighting(hits, mask=mask, range_epsilon=1e-7)

    plain = PICSLike(fast_config_path, basis=False, mask=mask)
    plain.run()
    filtered = PICSLike(fast_config_path, basis=False, mask=mask, pixel_filter=weighting)
    filtered.run()

    np.testing.assert_allclose(_shifted_logl(plain), _shifted_logl(filtered), atol=1e-6)


def test_projector_reaches_the_likelihood(fast_config_path):
    """A rank-reducing filter must change the posterior, not just survive."""
    mask = _mask_of(fast_config_path)
    projector = harmonic_deprojection(mask=mask, spins=[0, 2], ells=[2, 3], slot="E")
    assert projector.rank < projector.n_pixels

    plain = PICSLike(fast_config_path, basis=False, mask=mask)
    plain.run()
    filtered = PICSLike(fast_config_path, basis=False, mask=mask, pixel_filter=projector)
    filtered.run()

    result = filtered.likelihood_result
    assert np.all(np.isfinite(result.log_likelihood_values))
    assert np.all(np.isfinite(result.chi_squared_values))
    assert not np.allclose(_shifted_logl(plain), _shifted_logl(filtered), atol=1e-6)
