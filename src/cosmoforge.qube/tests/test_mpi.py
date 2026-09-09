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


def test_effective_ells_collective(comm, config_resolver):
    """get_effective_ells must enter the collective per-ℓ Fisher on every rank.

    A ``if rank != 0: return`` in front of the collective would hang the
    workers inside _compute_per_ell_fisher (ADR-0019). Calling it on all ranks
    must complete; rank 0 returns the centroid, workers return None.
    """
    from cosmocore import Bins

    config_file = config_resolver("tests/data/nside4/T/config.yaml")
    fisher = Fisher(config_file)
    fisher.set_binning(Bins.fromdeltal(2, 8, 3))
    fisher.run()
    os.unlink(config_file)

    ell_eff = fisher.get_effective_ells()

    if comm.Get_rank() == 0:
        assert ell_eff is not None
        assert np.all(np.isfinite(ell_eff))
    else:
        assert ell_eff is None


def _qu_mask(local_path):
    import healpy as hp

    path = os.path.join(local_path, "tests", "data", "nside4", "QU", "mask.fits")
    return np.repeat(np.asarray(hp.read_map(path), float)[:, None], 2, axis=1)


def test_pixel_filter_survives_the_shared_memory_broadcast(
    comm, local_path, config_resolver
):
    """``W`` travels through ``_shared_array``, never as a pickle (ADR-0020).

    Every rank must end holding the same factors, and a projector's ``U`` must
    still *be* its ``W``: raw and pre-filtered inputs land in identical
    coordinates only while that identity holds.
    """
    from cosmocore.filters import harmonic_deprojection

    mask = _qu_mask(local_path)
    projector = harmonic_deprojection(mask=mask, spins=[2], ells=[2, 3], slot="E")

    config_file = config_resolver("tests/data/nside4/QU/config.yaml")
    fisher = Fisher(config_file, basis=False, mask=mask, pixel_filter=projector)
    fisher.run()
    os.unlink(config_file)

    f = fisher.pixel_filter
    assert f is not None
    assert f.W.shape == (projector.n_pixels, projector.rank)
    assert f.U is f.W
    assert f.rank == projector.rank
    assert f.fingerprint == projector.fingerprint

    checksums = comm.allgather((float(f.W.sum()), float(np.abs(f.W).sum())))
    assert len(set(checksums)) == 1


def test_filtered_spectra_under_mpi(comm, local_path, config_resolver):
    """The worker ranks estimate from restricted maps and derivatives."""
    from cosmocore.filters import harmonic_deprojection

    mask = _qu_mask(local_path)
    projector = harmonic_deprojection(mask=mask, spins=[2], ells=[2, 3], slot="E")

    config_file_f = config_resolver("tests/data/nside4/QU/config.yaml")
    fisher = Fisher(config_file_f, basis=False, mask=mask, pixel_filter=projector)
    fisher.run()
    os.unlink(config_file_f)

    config_file_s = config_resolver("tests/data/nside4/QU/config.yaml")
    qml = Spectra(config_file_s, fisher=fisher)
    qml.run()
    os.unlink(config_file_s)

    assert qml.pixel_filter is not None
    assert qml.pixel_filter.U is qml.pixel_filter.W

    if comm.Get_rank() == 0:
        spectra = qml.get_power_spectra()
        assert spectra is not None
        assert np.all(np.isfinite(spectra))
