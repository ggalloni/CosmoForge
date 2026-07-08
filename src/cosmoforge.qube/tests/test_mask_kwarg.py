"""Tests for the ``mask=`` injection kwarg on Fisher/Spectra (ADR-0017).

The mask array injection is owned by ``Core`` (validated in cosmocore); here we
assert the orchestration classes surface it as an *explicit, named* constructor
kwarg (IDE-discoverable per ADR-0017 §Decision.4) and that an injected mask lets
the field/geometry setup run with no mask file on disk.
"""

import inspect

import numpy as np
import pytest

from cosmocore.settings import InputParams
from qube import Fisher, Spectra


def _params(config_resolver):
    """nside4/T config with the mask file pointed at a nonexistent path."""
    cfg = config_resolver("tests/data/nside4/T/config.yaml")
    params = InputParams.read_parameter_file(cfg)
    params.maskfile = "/nonexistent/mask.fits"
    return params


def _full_sky_mask(params):
    return np.ones(12 * params.nside**2, dtype=np.float64)


@pytest.mark.parametrize("cls", [Fisher, Spectra])
def test_mask_is_named_kwarg(cls):
    """``mask`` is an explicit signature parameter, not hidden in **kwargs."""
    assert "mask" in inspect.signature(cls.__init__).parameters


@pytest.mark.parametrize("cls", [Fisher, Spectra])
def test_injected_mask_reaches_geometry_without_file(config_resolver, cls):
    """Fisher/Spectra(params, mask=arr) run setup with no mask file present."""
    params = _params(config_resolver)
    obj = cls(params, mask=_full_sky_mask(params))
    obj.setup_fields()
    active, _ = obj.setup_geometry()
    assert active is not None
