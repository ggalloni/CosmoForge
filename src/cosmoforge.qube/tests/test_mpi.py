"""MPI smoke tests for qube pipelines.

Run under ``mpirun -n 2 uv run pytest test_mpi.py`` to exercise the
size>1 broadcast and shared-memory paths. Under single-rank
invocation these still run and validate the serial path.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from cosmocore._mpi import MPI
from qube import Fisher, Spectra


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


def test_fisher_pipeline_under_mpi(comm, local_path, config_resolver):
    """Fisher.run() must complete on every rank and match the serial reference."""
    config_file = config_resolver("tests/data/nside4/T/config.yaml")
    fisher = Fisher(config_file)
    fisher.run()
    os.unlink(config_file)

    if comm.Get_rank() == 0:
        assert fisher.fisher is not None
        ref_path = os.path.join(
            local_path, "tests", "data", "nside4", "T", "ref_fisher.dat"
        )
        ref = np.loadtxt(ref_path, dtype=np.float64)
        beam = fisher.beam_smoothing
        fisher_raw = fisher.fisher / np.outer(beam, beam)
        np.testing.assert_allclose(fisher_raw, ref, atol=1e-3, rtol=1e-5)


def test_spectra_pipeline_under_mpi(comm, config_resolver):
    """Spectra.run() exercises the point_vectors shared-memory broadcast."""
    config_file_f = config_resolver("tests/data/nside4/T/config.yaml")
    fisher = Fisher(config_file_f)
    fisher.run()
    os.unlink(config_file_f)

    config_file_s = config_resolver("tests/data/nside4/T/config.yaml")
    qml = Spectra(config_file_s, fisher=fisher)
    qml.run()
    os.unlink(config_file_s)

    if comm.Get_rank() == 0:
        ps = qml.get_power_spectra()
        assert ps is not None
        assert np.all(np.isfinite(ps))


def test_cross_spectra_under_mpi(comm, config_resolver):
    """do_cross=true exercises the maps2 / noise_cov2 broadcast paths."""
    config_file_f = config_resolver("tests/data/nside4/QU/cross_config.yaml")
    fisher = Fisher(config_file_f)
    fisher.run()
    os.unlink(config_file_f)

    config_file_s = config_resolver("tests/data/nside4/QU/cross_config.yaml")
    qml = Spectra(config_file_s, fisher=fisher)
    qml.run()
    os.unlink(config_file_s)

    if comm.Get_rank() == 0:
        ps = qml.get_power_spectra()
        assert ps is not None
        assert np.all(np.isfinite(ps))


def test_bandpower_window_collective(comm, config_resolver):
    """get_bandpower_window_function must not deadlock and returns rank-0-only."""
    config_file = config_resolver("tests/data/nside4/T/config.yaml")
    fisher = Fisher(config_file)
    fisher.run()
    os.unlink(config_file)

    W = fisher.get_bandpower_window_function()

    if comm.Get_rank() == 0:
        assert W is not None
        assert W.ndim == 2
        assert np.all(np.isfinite(W))
    else:
        assert W is None
