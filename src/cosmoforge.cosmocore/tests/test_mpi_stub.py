"""Tests for the no-op MPI stub used when ``mpi4py`` is unavailable.

These cover the stub's public surface in isolation. End-to-end coverage
of the stub through :class:`cosmocore.mpi_utils.MPISharedMemoryMixin`
would require uninstalling ``mpi4py`` for the test run; left to CI
matrix when a no-MPI variant is added.
"""

from __future__ import annotations

import numpy as np

from cosmocore import _mpi_stub as MPI


def test_comm_world_single_rank():
    comm = MPI.COMM_WORLD
    assert comm.Get_rank() == 0
    assert comm.Get_size() == 1


def test_bcast_returns_input():
    assert MPI.COMM_WORLD.bcast({"a": 1}, root=0) == {"a": 1}


def test_Bcast_is_noop_in_place():
    arr = np.arange(5, dtype=np.float64)
    before = arr.copy()
    MPI.COMM_WORLD.Bcast(arr, root=0)
    assert np.array_equal(arr, before)


def test_Reduce_copies_send_into_recv():
    send = np.array([1.0, 2.0, 3.0])
    recv = np.zeros_like(send)
    MPI.COMM_WORLD.Reduce(send, recv, op=MPI.SUM, root=0)
    assert np.array_equal(recv, send)


def test_gather_wraps_single_rank_payload():
    assert MPI.COMM_WORLD.gather("payload", root=0) == ["payload"]


def test_Barrier_returns_none():
    assert MPI.COMM_WORLD.Barrier() is None


def test_Split_with_UNDEFINED_yields_COMM_NULL():
    assert MPI.COMM_WORLD.Split(MPI.UNDEFINED, 0) is MPI.COMM_NULL


def test_Split_with_color_yields_communicator():
    sub = MPI.COMM_WORLD.Split(0, 0)
    assert sub.Get_rank() == 0
    assert sub.Get_size() == 1


def test_Split_type_returns_communicator():
    sub = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    assert sub.Get_rank() == 0


def test_Win_Allocate_shared_yields_local_buffer():
    win = MPI.Win.Allocate_shared(64, 8, comm=MPI.COMM_WORLD)
    buf, itemsize = win.Shared_query(0)
    assert len(memoryview(buf)) == 64
    assert itemsize == 1
    win.Free()


def test_HAS_MPI_reflects_mpi4py_availability():
    from cosmocore._mpi import HAS_MPI

    try:
        import mpi4py  # noqa: F401
    except ImportError:
        assert HAS_MPI is False
    else:
        assert HAS_MPI is True
