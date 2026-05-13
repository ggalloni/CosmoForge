"""MPI dispatch: real ``mpi4py.MPI`` when installed, no-op stub otherwise.

Every production module that needs MPI primitives should import from
this module rather than directly from ``mpi4py``. The stub branch lets
single-process users install and run without an MPI runtime; the real
branch is bit-for-bit equivalent to the original direct import.
"""

from __future__ import annotations

try:
    from mpi4py import MPI

    HAS_MPI = True
except ImportError:
    from . import _mpi_stub as MPI  # type: ignore[no-redef]

    HAS_MPI = False

__all__ = ["MPI", "HAS_MPI"]
