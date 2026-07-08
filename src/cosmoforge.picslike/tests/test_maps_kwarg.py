"""Tests for ``maps1=``/``maps2=`` injection on PICSLike (ADR-0017, A5 extension).

Maps injection was Spectra-only in A3; A5 hoists ``_resolve_maps`` to ``Core`` and
extends the seam to PICSLike (which reads observed maps identically). Dispatch +
validation are shared on ``Core``; here we assert PICSLike surfaces the kwargs and
resolves an injected array without touching disk.
"""

import inspect

import numpy as np
import pytest

from cosmocore.in_out import readcl
from cosmocore.settings import InputParams
from picslike import PICSLike


@pytest.mark.parametrize("name", ["maps1", "maps2"])
def test_maps_is_named_kwarg(name):
    """``maps1``/``maps2`` are explicit signature parameters on PICSLike."""
    assert name in inspect.signature(PICSLike.__init__).parameters


def test_injected_maps_matches_file_path(fast_config_path):
    """Injecting ``maps1`` yields the same maps as reading from the maps file."""
    ref = PICSLike(fast_config_path)
    ref.setup_fields()
    ref.setup_geometry()
    ref.setup_maps()
    ref_maps1 = ref.maps1.copy()

    inj = PICSLike(fast_config_path, maps1=ref_maps1.copy())
    inj.params.inputmapfile1 = "/nonexistent/sims.npy"
    inj.setup_fields()
    inj.setup_geometry()
    inj.setup_maps()

    np.testing.assert_array_equal(inj.maps1, ref_maps1)


def test_picslike_setup_fully_in_memory(fast_config_path):
    """PICSLike runs its full setup (fields → maps) from injected arrays only,
    with no mask/cov/cls/beam/map file on disk."""
    params = InputParams.read_parameter_file(fast_config_path)
    lmax_signal = 4 * params.nside

    import healpy as hp

    beam = hp.read_cl(params.beam_file).astype(np.float64)
    cls = readcl(params.inputclfile, params, lmax=lmax_signal)
    mask = np.ones((12 * params.nside**2, params.nfields), dtype=np.float64)

    probe = PICSLike(params, mask=mask, noise_cov1=np.eye(1), beam=beam, cls_data=cls)
    probe.setup_fields()
    probe.setup_geometry()
    n = int(probe.collection.total_active_pixels)
    noise_cov1 = np.eye(n) * 0.1
    maps1 = np.random.default_rng(0).standard_normal((n, params.nsims))

    for key in (
        "maskfile",
        "covmatfile1",
        "inputclfile",
        "fiducialfile",
        "beam_file",
        "inputmapfile1",
    ):
        setattr(params, key, "/nonexistent/" + key)

    like = PICSLike(
        params,
        mask=mask,
        noise_cov1=noise_cov1,
        cls_data=cls,
        fiducial_cls=cls,
        beam=beam,
        maps1=maps1,
    )
    like.setup_fields()
    like.setup_geometry()
    like.setup_covariance_matrices()
    like.setup_cls(lmax=lmax_signal)
    like.setup_beams(lmax=lmax_signal)
    like.setup_maps()
    np.testing.assert_array_equal(like.maps1, maps1)
