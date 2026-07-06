"""Opt-in persistence (ADR-0015): the write helpers no-op on a falsy path.

A falsy path (``None`` or ``""``) means "do not persist". A real path writes as
before. The gate lives in the helpers so every present and future call site is
covered uniformly.
"""

import numpy as np
import pytest

from cosmocore.in_out import (
    output_geometry,
    write_covmat_reduced,
    write_out_matrix,
    writecl,
)


def _call(helper, path):
    """Invoke ``helper`` with ``path`` and otherwise-valid minimal arguments."""
    if helper is write_covmat_reduced:
        write_covmat_reduced(path, np.eye(3))
    elif helper is write_out_matrix:
        write_out_matrix(path, np.eye(3))
    elif helper is writecl:
        writecl(path, np.arange(6.0).reshape(3, 2))
    elif helper is output_geometry:
        output_geometry(
            path,
            [2],
            [np.zeros((2, 3))],
            np.array([[0, 1]]),
        )
    else:  # pragma: no cover
        raise AssertionError(helper)


ALL_HELPERS = [write_covmat_reduced, write_out_matrix, writecl, output_geometry]


@pytest.mark.parametrize("helper", ALL_HELPERS)
@pytest.mark.parametrize("path", [None, "", "   "])
def test_helper_noop_on_falsy_path(helper, path, tmp_path, monkeypatch):
    """A None/empty/blank path writes nothing and does not raise."""
    monkeypatch.chdir(tmp_path)
    _call(helper, path)
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize("helper", ALL_HELPERS)
def test_helper_writes_on_real_path(helper, tmp_path):
    """A real path still produces a non-empty file."""
    target = tmp_path / "artifact.out"
    _call(helper, str(target))
    assert target.exists()
    assert target.stat().st_size > 0
