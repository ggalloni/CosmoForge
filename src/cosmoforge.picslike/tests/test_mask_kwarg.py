"""Tests for the ``mask=`` injection kwarg on PICSLike (ADR-0017).

Mask array injection is owned by ``Core`` (validated in cosmocore); here we
assert PICSLike surfaces it as an *explicit, named* constructor kwarg
(IDE-discoverable per ADR-0017 §Decision.4) and that an injected mask lets the
field/geometry setup run with no mask file on disk.
"""

import inspect

import numpy as np

from cosmocore.settings import InputParams
from picslike import PICSLike


def test_mask_is_named_kwarg():
    """``mask`` is an explicit signature parameter, not hidden in **kwargs."""
    assert "mask" in inspect.signature(PICSLike.__init__).parameters


def test_injected_mask_reaches_geometry_without_file(config_resolver):
    """PICSLike(params, mask=arr) runs setup with no mask file present."""
    cfg = config_resolver("tests/data/nside4/TQU/config.yaml")
    params = InputParams.read_parameter_file(cfg)
    params.maskfile = "/nonexistent/mask.fits"
    mask = np.ones((12 * params.nside**2, params.nfields), dtype=np.float64)

    like = PICSLike(params, mask=mask)
    like.setup_fields()
    active, _ = like.setup_geometry()
    assert active is not None
