"""Direct MPI tests for cosmocore MPISharedMemoryMixin.

These tests exercise the size>1 paths when run under mpirun (e.g.
``mpirun -n 2 uv run pytest test_mpi.py``). Under single-rank
invocation they still run, validating the size==1 behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmocore import MPISharedMemoryMixin
from cosmocore._mpi import MPI


class _Helper(MPISharedMemoryMixin):
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()


@pytest.fixture
def helper():
    h = _Helper()
    yield h
    h.close()


def test_shared_array_data_consistency(helper):
    """All ranks must see identical contents in the shared buffer."""
    arr = np.arange(120, dtype=np.float64).reshape(10, 12) if helper.rank == 0 else None
    shared = helper._shared_array(arr)

    expected = np.arange(120, dtype=np.float64).reshape(10, 12)
    np.testing.assert_array_equal(shared, expected)
    assert shared.shape == (10, 12)
    assert shared.dtype == np.float64


def test_shared_array_dtype_propagation(helper):
    """Shape and dtype broadcast must work for non-default dtypes."""
    arr = np.array([1, 2, 3, 4, 5], dtype=np.int32) if helper.rank == 0 else None
    shared = helper._shared_array(arr)
    np.testing.assert_array_equal(shared, np.array([1, 2, 3, 4, 5], dtype=np.int32))
    assert shared.dtype == np.int32


def test_shared_array_multiple_buffers(helper):
    """Multiple shared windows must remain independent."""
    a_src = np.full(50, 7.0) if helper.rank == 0 else None
    b_src = np.full((4, 5), -3.0) if helper.rank == 0 else None
    a = helper._shared_array(a_src)
    b = helper._shared_array(b_src)
    np.testing.assert_array_equal(a, np.full(50, 7.0))
    np.testing.assert_array_equal(b, np.full((4, 5), -3.0))


def test_bcast_array(helper):
    """Buffer-based broadcast distributes data without pickling."""
    arr = np.linspace(0, 1, 16).reshape(4, 4) if helper.rank == 0 else None
    out = helper._bcast_array(arr)
    np.testing.assert_array_equal(out, np.linspace(0, 1, 16).reshape(4, 4))


def test_cleanup_is_idempotent(helper):
    """close() must be safe to call multiple times."""
    helper._shared_array(np.arange(10, dtype=np.float64) if helper.rank == 0 else None)
    helper.close()
    helper.close()  # second call must not raise
    assert not getattr(helper, "_shared_wins", [])


def test_context_manager():
    """Using the mixin as a context manager releases resources on exit."""
    with _Helper() as h:
        h._shared_array(np.zeros(8) if h.rank == 0 else None)
        assert len(h._shared_wins) == 1
    assert not getattr(h, "_shared_wins", [])
