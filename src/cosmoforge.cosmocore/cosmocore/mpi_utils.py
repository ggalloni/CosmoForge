"""MPI utilities for shared memory and array broadcasting."""

from __future__ import annotations

import numpy as np
from mpi4py import MPI


class MPISharedMemoryMixin:
    """Mixin providing shared memory array distribution for MPI classes.

    Requires the class to have ``self.comm`` (MPI communicator) and
    ``self.rank`` (process rank) attributes.

    For single-node runs, ``_shared_array`` places one copy of each
    array in shared memory so every rank on the node gets a zero-copy
    view — no data duplication, no message-size limits.

    ``_bcast_array`` is kept as a fallback for multi-node topologies
    where shared memory is not available across nodes.
    """

    # ------------------------------------------------------------------
    # Shared memory (intra-node, zero-copy)
    # ------------------------------------------------------------------

    def _setup_shared_comm(self):
        """Create intra-node communicator for shared memory, if not already done."""
        if not hasattr(self, "_shared_comm"):
            self._shared_comm = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
            self._shared_wins: list[MPI.Win] = []

    def _shared_array(self, arr: np.ndarray | None = None) -> np.ndarray:
        """Share a numpy array via MPI shared memory (intra-node, zero-copy).

        Rank 0 on each node allocates and fills the shared buffer.
        Other ranks attach to it read-only.  No data is copied between
        ranks on the same node.

        Parameters
        ----------
        arr : numpy.ndarray or None
            Array to share.  Only rank 0 needs to pass the actual array;
            other ranks may pass ``None`` (the shape and dtype are
            broadcast from rank 0).
        """
        self._setup_shared_comm()
        comm_node = self._shared_comm

        is_root = self.rank == 0 and arr is not None
        shape = self.comm.bcast(arr.shape if is_root else None, root=0)
        dtype = self.comm.bcast(arr.dtype if is_root else None, root=0)
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

        alloc = nbytes if comm_node.Get_rank() == 0 else 0
        win = MPI.Win.Allocate_shared(alloc, np.dtype(dtype).itemsize, comm=comm_node)
        self._shared_wins.append(win)

        buf, _ = win.Shared_query(0)
        shared = np.ndarray(shape, dtype=dtype, buffer=buf)

        if comm_node.Get_rank() == 0:
            shared[:] = arr

        comm_node.Barrier()
        return shared

    def _cleanup_shared(self):
        """Free all shared memory windows and the intra-node communicator."""
        for win in getattr(self, "_shared_wins", []):
            win.Free()
        self._shared_wins = []
        if hasattr(self, "_shared_comm"):
            self._shared_comm.Free()
            del self._shared_comm

    # ------------------------------------------------------------------
    # Buffer-based broadcast (fallback for multi-node)
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
