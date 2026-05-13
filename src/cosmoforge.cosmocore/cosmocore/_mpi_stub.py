"""No-op MPI stub used when mpi4py is not installed.

Exposes the subset of the ``mpi4py.MPI`` surface that this codebase
actually touches: ``COMM_WORLD``, ``COMM_NULL``, ``COMM_TYPE_SHARED``,
``SUM``, ``UNDEFINED``, ``Comm``, ``Win``. The semantics collapse to a
single-rank, single-node world where every collective is a local
identity operation. This lets the package import and the production
classes initialise without an MPI runtime; cluster users install the
``mpi`` extra and get the real bindings.
"""

from __future__ import annotations

import numpy as np

UNDEFINED = -32766
SUM = "SUM"
COMM_TYPE_SHARED = "SHARED"


class _NoOpComm:
    """Single-rank communicator stub.

    All collectives are local identity ops because there is only one
    rank. Type-checks against :data:`COMM_NULL` use object identity, so
    a split that yields no communicator returns :data:`COMM_NULL`
    rather than an instance.
    """

    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def Barrier(self) -> None:
        return None

    def bcast(self, obj, root=0):
        return obj

    def Bcast(self, arr, root=0) -> None:
        return None

    def gather(self, obj, root=0):
        return [obj]

    def Reduce(self, send, recv, op=None, root=0) -> None:
        if recv is not None:
            recv[...] = np.asarray(send)

    def Split_type(self, comm_type, key=0, info=None):
        return _NoOpComm()

    def Split(self, color, key=0):
        if color == UNDEFINED:
            return COMM_NULL
        return _NoOpComm()

    def Free(self) -> None:
        return None


class _NullComm:
    """Sentinel returned by :meth:`_NoOpComm.Split` when color=UNDEFINED."""

    def Free(self) -> None:
        return None


COMM_WORLD = _NoOpComm()
COMM_NULL = _NullComm()
Comm = _NoOpComm


class _NoOpWin:
    """Shared-memory window stub backed by a local heap buffer."""

    def __init__(self, nbytes: int):
        self._buf = np.empty(max(int(nbytes), 1), dtype=np.uint8)

    def Shared_query(self, rank: int):
        return (self._buf.data, 1)

    def Free(self) -> None:
        return None


class Win:
    """Stub mirroring ``mpi4py.MPI.Win``'s class-method factory."""

    @staticmethod
    def Allocate_shared(size_bytes, itemsize, comm=None):
        return _NoOpWin(size_bytes)
