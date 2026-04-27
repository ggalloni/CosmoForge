"""MPI utilities for shared memory and array broadcasting."""

from __future__ import annotations

import numpy as np
from mpi4py import MPI


class MPISharedMemoryMixin:
    """Mixin providing shared memory array distribution for MPI classes.

    Requires ``self.comm`` and ``self.rank``. ``_shared_array`` places
    one copy of each array per node in shared memory; intra-node ranks
    get a zero-copy view, and a single buffer-based ``Bcast`` seeds
    each node's local rank 0. ``_bcast_array`` is the buffer-based
    broadcast fallback. Call :meth:`close` (or use as a context
    manager) to release windows and split communicators.
    """

    # ------------------------------------------------------------------
    # Shared memory (intra-node, zero-copy)
    # ------------------------------------------------------------------

    def _setup_shared_comm(self):
        """Create intra- and inter-node communicators, if not already done."""
        if hasattr(self, "_shared_comm"):
            return

        self._shared_comm = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
        self._shared_wins: list[MPI.Win] = []

        # Inter-node communicator: node-local rank 0 only. Using
        # self.rank as the key keeps global rank 0 as inter-comm root.
        node_rank = self._shared_comm.Get_rank()
        color = 0 if node_rank == 0 else MPI.UNDEFINED
        self._inter_node_comm = self.comm.Split(color, self.rank)

    def _shared_array(self, arr: np.ndarray | None = None) -> np.ndarray:
        """Share a numpy array via MPI shared memory (intra-node, zero-copy).

        Each node's local rank 0 allocates the shared buffer and is
        seeded with the data (either directly on the global root or via
        a buffer-based inter-node ``Bcast``). All other ranks attach
        read-only — no inter-rank copies on a node.

        Parameters
        ----------
        arr : numpy.ndarray or None
            Array to share. Only global rank 0 needs to pass the actual
            array; other ranks may pass ``None`` (shape and dtype are
            broadcast from rank 0).
        """
        self._setup_shared_comm()
        comm_node = self._shared_comm
        inter_comm = self._inter_node_comm

        is_root = self.rank == 0 and arr is not None
        shape = self.comm.bcast(arr.shape if is_root else None, root=0)
        dtype = self.comm.bcast(arr.dtype if is_root else None, root=0)
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

        alloc = nbytes if comm_node.Get_rank() == 0 else 0
        win = MPI.Win.Allocate_shared(alloc, np.dtype(dtype).itemsize, comm=comm_node)
        self._shared_wins.append(win)

        buf, _ = win.Shared_query(0)
        shared = np.ndarray(shape, dtype=dtype, buffer=buf)

        # Seed each node's shared buffer: global root copies in its
        # array, then node-roots exchange via buffer-based Bcast.
        if comm_node.Get_rank() == 0:
            if self.rank == 0:
                shared[:] = arr
            inter_comm.Bcast(shared, root=0)

        comm_node.Barrier()
        return shared

    def _cleanup_shared(self) -> None:
        """Free all shared-memory windows and split communicators.

        Idempotent — safe to call multiple times.
        """
        for win in getattr(self, "_shared_wins", []):
            try:
                win.Free()
            except Exception:
                pass
        self._shared_wins = []

        inter = getattr(self, "_inter_node_comm", None)
        if inter is not None and inter != MPI.COMM_NULL:
            try:
                inter.Free()
            except Exception:
                pass
        if hasattr(self, "_inter_node_comm"):
            del self._inter_node_comm

        shared_comm = getattr(self, "_shared_comm", None)
        if shared_comm is not None:
            try:
                shared_comm.Free()
            except Exception:
                pass
            finally:
                del self._shared_comm

    def close(self) -> None:
        """Release all MPI shared-memory resources owned by this instance."""
        self._cleanup_shared()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    # ------------------------------------------------------------------
    # Buffer-based broadcast (fallback / non-shared use)
    # ------------------------------------------------------------------

    def _bcast_array(self, arr: np.ndarray | None = None) -> np.ndarray:
        """Broadcast a numpy array using buffer-based MPI.

        Uses ``comm.Bcast`` (uppercase) which sends raw memory buffers
        instead of ``comm.bcast`` (lowercase) which serializes via the
        standard library.  This avoids the ~2 GB message-size limit
        that affects serialization-based broadcasts in many MPI
        implementations.

        Parameters
        ----------
        arr : numpy.ndarray or None
            Array to broadcast.  Only rank 0 needs to pass the actual
            array; other ranks may pass ``None``.
        """
        is_root = self.rank == 0 and arr is not None
        shape = self.comm.bcast(arr.shape if is_root else None, root=0)
        dtype = self.comm.bcast(arr.dtype if is_root else None, root=0)
        if self.rank != 0:
            arr = np.empty(shape, dtype=dtype)
        self.comm.Bcast(arr, root=0)
        return arr
