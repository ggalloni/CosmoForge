"""Reusing a live PICSLike instance must not corrupt chi-squared.

Two pieces of per-evaluation state used to survive longer than they should:

* the harmonic (SMW) cache of ``(projected1, projected2, term1)``, which is
  derived from the basis's noise factorisation and so belongs to a single basis
  lifetime — a rebuild must not leave the next chi-squared evaluation pairing a
  stale projection with the new basis's kernel;
* the theory spectra themselves, which beam smoothing multiplied in place,
  re-smoothing the parameter grid's arrays on every evaluation.

Both were silent: no exception, just wrong numbers.
"""

import numpy as np
import pytest

from picslike import PICSLike

#: Noise rescaling between basis #1 and basis #2. Same shapes on both sides, so
#: a stale cache is silently wrong rather than loudly broken.
NOISE_SCALE = 3.0


@pytest.fixture
def cfg(sandboxed_config):
    return sandboxed_config("tests/data/nside4/TQU/fast_config.yaml")


def _prepared(config_path):
    """A PICSLike instance set up to the point just before the basis is built."""
    like = PICSLike(config_path)
    like.setup_parameter_grid()
    like.setup_fields()
    like.setup_geometry()
    like.setup_covariance_matrices()
    like.setup_cls(lmax=like.lmax_signal)
    like.setup_beams(lmax=like.lmax_signal)
    return like


def test_repeated_evaluation_is_stable(cfg):
    """The same parameter point evaluates to the same chi-squared every time."""
    like = _prepared(cfg)
    like.setup_computation_basis(method="harmonic")
    like.setup_maps()

    point = like.parameter_grid.grid_points[0]
    first, _ = like._compute_likelihood_point(point)
    second, _ = like._compute_likelihood_point(point)

    np.testing.assert_allclose(second, first, rtol=1e-12)


def test_basis_rebuild_invalidates_smw_cache(cfg):
    """A second ``setup_computation_basis`` re-derives the cached SMW data."""
    like = _prepared(cfg)
    like.setup_computation_basis(method="harmonic")
    like.setup_maps()

    point = like.parameter_grid.grid_points[0]
    like._compute_likelihood_point(point)  # populates the cache from basis #1

    # Rebuild on the live instance against a different noise model. The basis
    # consumed (and nulled) noise_cov1, so the noise has to be re-materialised.
    like.setup_covariance_matrices()
    like.noise_cov1 *= NOISE_SCALE
    like.setup_computation_basis(method="harmonic")
    chi2, log_like = like._compute_likelihood_point(point)

    ref = _prepared(cfg)
    ref.noise_cov1 *= NOISE_SCALE
    ref.setup_computation_basis(method="harmonic")
    ref.setup_maps()
    chi2_ref, log_like_ref = ref._compute_likelihood_point(point)

    np.testing.assert_allclose(chi2, chi2_ref, rtol=1e-10)
    np.testing.assert_allclose(log_like, log_like_ref, rtol=1e-10)
