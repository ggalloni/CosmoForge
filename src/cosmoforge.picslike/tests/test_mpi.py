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


def test_broadcast_drops_stale_smw_cache(fast_config_path):
    """The broadcast is the worker ranks' only SMW-cache invalidation point.

    ``setup_maps`` and ``setup_computation_basis`` — the two invalidation sites
    on rank 0 — are both rank-0 only inside ``run()``. Workers populate the
    cache during ``compute()`` and receive their basis and maps here, so a
    broadcast that left the cache standing would pair a projection from the
    previous basis with the new one.
    """
    pl = PICSLike(fast_config_path)
    pl.run()

    pl._smw_data_cache = ("stale", "stale", "stale")
    pl._broadcast_variables()

    assert pl._smw_data_cache is None
