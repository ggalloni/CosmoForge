"""MPI smoke test for PICSLike pipeline.

Run under ``mpirun -n 2 uv run pytest test_mpi.py`` to exercise the
broadcast and shared-memory paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmocore._mpi import MPI
from picslike import PICSLike


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


def test_picslike_pipeline_under_mpi(comm, fast_config_path):
    """PICSLike.run() exercises point_vectors + maps shared-memory broadcast."""
    pl = PICSLike(fast_config_path)
    pl.run()

    if comm.Get_rank() == 0:
        result = pl.likelihood_result
        assert result is not None
        assert np.all(np.isfinite(result.log_likelihood_values))
        assert np.all(np.isfinite(result.chi_squared_values))
