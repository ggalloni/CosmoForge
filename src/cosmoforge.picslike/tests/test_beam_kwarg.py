"""``beam=`` injection on PICSLike (ADR-0017) holds at every grid point.

The traditional path (``basis=False``) once rebuilt the beam from
``smoothing_type`` at each grid point, so an injected beam was used at setup
and then silently replaced.
"""

import healpy as hp
import numpy as np

from cosmocore.settings import InputParams
from picslike import PICSLike


def _chi2(config_path, smoothing_type, beam=None):
    params = InputParams.read_parameter_file(config_path)
    params.smoothing_type = smoothing_type
    like = PICSLike(params, basis=False, beam=beam)
    like.run()
    return like.get_chi_squared()


def test_injected_beam_used_on_traditional_path(sandboxed_config):
    """An injected beam gives the chi-squared of the same beam read from file."""
    cfg = sandboxed_config("tests/data/nside4/TQU/fast_config.yaml")
    beam = hp.read_cl(InputParams.read_parameter_file(cfg).beam_file)

    from_file = _chi2(cfg, "file")
    injected = _chi2(cfg, "cosine_legacy", beam=beam.astype(np.float64))
    analytic = _chi2(cfg, "cosine_legacy")

    np.testing.assert_allclose(injected, from_file, rtol=1e-12)
    assert not np.allclose(injected, analytic)
