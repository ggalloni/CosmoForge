"""Per-stage RSS memory profiling for benchmark scripts.

Combines a polling-thread peak monitor with stage-boundary entry/exit
sampling. Peak RSS catches transient allocations that are released
before the next boundary; entry/exit deltas attribute residual growth
to specific stages.

Allocator caching (pymalloc, NumPy's allocator) means stage-exit RSS
typically does not return to pre-stage levels. The peak is the
authoritative number; the delta is informative but not authoritative.

The optional :class:`StageProfiler` adds named sub-stage emission and
a per-array inventory snapshot at each boundary; ``overhead_mb``
contrasts measured RSS growth against the sum of declared
``np.ndarray`` sizes reachable from the host object, exposing
BLAS / Cython workspace that doesn't show up in Python.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psutil

_BYTES_PER_MB = 1024.0 * 1024.0


@dataclass
class ArrayInfo:
    """One ``np.ndarray`` attribution: dotted path, shape, dtype, nbytes."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int

    @property
    def mb(self) -> float:
        return self.nbytes / _BYTES_PER_MB

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "nbytes": self.nbytes,
            "MB": round(self.mb, 3),
        }


@dataclass
class MemSample:
    """One stage measurement.

    The ``inventory`` / ``inventory_total_mb`` / ``inventory_delta_mb`` /
    ``overhead_mb`` fields are populated only when a :class:`StageProfiler`
    produced the sample; plain ``measure()`` calls leave them ``None``.

    ``inventory_delta_mb`` = inventory total at exit minus inventory total
    at entry — i.e. how much declared-array memory was *newly added*
    during this stage. ``overhead_mb`` = (peak − entry) − inventory_delta;
    positive means BLAS workspace / Python transients beyond the declared
    arrays, negative means the allocator released memory mid-stage.
    """

    name: str
    rss_at_entry_mb: float
    rss_at_exit_mb: float
    rss_peak_mb: float
    inventory: list[ArrayInfo] | None = None
    inventory_total_mb: float | None = None
    inventory_entry_mb: float | None = None
    inventory_delta_mb: float | None = None
    overhead_mb: float | None = None
    rss_delta_mb: float = field(init=False)

    def __post_init__(self) -> None:
        self.rss_delta_mb = self.rss_at_exit_mb - self.rss_at_entry_mb

    def to_dict(self) -> dict:
        out: dict = {
            "name": self.name,
            "rss_at_entry_mb": self.rss_at_entry_mb,
            "rss_at_exit_mb": self.rss_at_exit_mb,
            "rss_peak_mb": self.rss_peak_mb,
            "rss_delta_mb": self.rss_delta_mb,
        }
        if self.inventory is not None:
            out["inventory"] = [a.to_dict() for a in self.inventory]
            out["inventory_total_mb"] = self.inventory_total_mb
            out["inventory_entry_mb"] = self.inventory_entry_mb
            out["inventory_delta_mb"] = self.inventory_delta_mb
            out["overhead_mb"] = self.overhead_mb
        return out


class MemMonitor:
    """Background thread polling RSS at a fixed interval.

    Used as a context manager. ``reset()`` rewinds the running max so the
    next ``measure()`` block sees only its own peak.
    """

    def __init__(self, interval_s: float = 0.1):
        self.interval_s = interval_s
        self._proc = psutil.Process()
        self._max_rss = self._proc.memory_info().rss
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> MemMonitor:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._max_rss = max(self._max_rss, self._proc.memory_info().rss)
            time.sleep(self.interval_s)

    def reset(self) -> None:
        self._max_rss = self._proc.memory_info().rss

    @property
    def max_rss_mb(self) -> float:
        return self._max_rss / _BYTES_PER_MB

    @property
    def current_rss_mb(self) -> float:
        return self._proc.memory_info().rss / _BYTES_PER_MB


@contextmanager
def measure(monitor: MemMonitor, name: str, samples: list[MemSample]):
    """Wrap a stage; append one ``MemSample`` to ``samples`` on exit."""
    monitor.reset()
    rss_in = monitor.current_rss_mb
    yield
    rss_out = monitor.current_rss_mb
    samples.append(
        MemSample(
            name=name,
            rss_at_entry_mb=rss_in,
            rss_at_exit_mb=rss_out,
            rss_peak_mb=monitor.max_rss_mb,
        )
    )


_PRIMITIVE_TYPES = (int, float, bool, str, bytes, complex, type(None))


def inventory_arrays(
    obj: Any,
    prefix: str = "",
    *,
    max_depth: int = 5,
    min_mb: float = 0.0,
    _visited: set[int] | None = None,
) -> list[ArrayInfo]:
    """Walk an object's attribute graph; return ``ArrayInfo`` for each
    ``np.ndarray`` reachable within ``max_depth`` whose size is at least
    ``min_mb`` megabytes.

    Recurses into ``list``/``tuple``/``dict`` containers and into objects
    exposing ``__dict__``. Skips dunder attributes and primitives. Cycle
    protection via an ``id()`` set.
    """
    if _visited is None:
        _visited = set()
    if max_depth < 0:
        return []
    if isinstance(obj, _PRIMITIVE_TYPES):
        return []
    obj_id = id(obj)
    if obj_id in _visited:
        return []
    _visited.add(obj_id)

    if isinstance(obj, np.ndarray):
        nbytes = int(obj.nbytes)
        if nbytes / _BYTES_PER_MB >= min_mb:
            return [
                ArrayInfo(
                    path=prefix or "<root>",
                    shape=tuple(obj.shape),
                    dtype=str(obj.dtype),
                    nbytes=nbytes,
                )
            ]
        return []

    out: list[ArrayInfo] = []
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(
                inventory_arrays(
                    v,
                    f"{prefix}[{i}]",
                    max_depth=max_depth - 1,
                    min_mb=min_mb,
                    _visited=_visited,
                )
            )
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{prefix}[{k!r}]" if prefix else f"[{k!r}]"
            out.extend(
                inventory_arrays(
                    v,
                    child,
                    max_depth=max_depth - 1,
                    min_mb=min_mb,
                    _visited=_visited,
                )
            )
        return out

    if hasattr(obj, "__dict__"):
        try:
            attrs = vars(obj)
        except TypeError:
            return out
        for k, v in attrs.items():
            if k.startswith("__"):
                continue
            child = f"{prefix}.{k}" if prefix else k
            out.extend(
                inventory_arrays(
                    v,
                    child,
                    max_depth=max_depth - 1,
                    min_mb=min_mb,
                    _visited=_visited,
                )
            )
    return out


class StageProfiler:
    """Emit per-stage :class:`MemSample` entries from a long-running pipeline.

    Hosts attach an instance to themselves (e.g.
    ``fisher._profiler = StageProfiler(monitor, samples, [fisher])``) and
    bracket substeps with ``with self._profiler.stage("name"):``. Each
    block appends one MemSample carrying entry / exit / peak RSS plus an
    inventory of large ``np.ndarray`` objects reachable from
    ``snapshot_targets`` at exit time.

    ``overhead_mb = (peak − entry) − inventory_total_mb`` is the headline
    diagnostic: positive means BLAS / transient allocations not visible in
    Python; negative means the allocator released memory before exit.
    """

    def __init__(
        self,
        monitor: MemMonitor,
        samples: list[MemSample],
        snapshot_targets: list[Any] = (),
        *,
        inventory_min_mb: float = 1.0,
        inventory_max_depth: int = 5,
    ):
        self.monitor = monitor
        self.samples = samples
        self.snapshot_targets: list[Any] = list(snapshot_targets)
        self.inventory_min_mb = inventory_min_mb
        self.inventory_max_depth = inventory_max_depth

    def add_target(self, target: Any) -> None:
        """Add another root object to walk on each stage exit."""
        self.snapshot_targets.append(target)

    def _snapshot_inventory(self) -> tuple[list[ArrayInfo], float]:
        inventory: list[ArrayInfo] = []
        visited: set[int] = set()
        for tgt in self.snapshot_targets:
            tgt_prefix = type(tgt).__name__.lower()
            inventory.extend(
                inventory_arrays(
                    tgt,
                    prefix=tgt_prefix,
                    max_depth=self.inventory_max_depth,
                    min_mb=self.inventory_min_mb,
                    _visited=visited,
                )
            )
        inventory.sort(key=lambda a: a.nbytes, reverse=True)
        total_mb = sum(a.nbytes for a in inventory) / _BYTES_PER_MB
        return inventory, total_mb

    @contextmanager
    def stage(self, name: str):
        self.monitor.reset()
        rss_in = self.monitor.current_rss_mb
        _, inv_entry_mb = self._snapshot_inventory()
        try:
            yield
        finally:
            rss_out = self.monitor.current_rss_mb
            peak = self.monitor.max_rss_mb
            inventory, inv_total_mb = self._snapshot_inventory()
            inv_delta_mb = inv_total_mb - inv_entry_mb
            # Overhead = how much peak RSS growth was NOT accounted for by
            # newly-declared arrays during this stage.
            overhead_mb = (peak - rss_in) - inv_delta_mb
            self.samples.append(
                MemSample(
                    name=name,
                    rss_at_entry_mb=rss_in,
                    rss_at_exit_mb=rss_out,
                    rss_peak_mb=peak,
                    inventory=inventory,
                    inventory_total_mb=inv_total_mb,
                    inventory_entry_mb=inv_entry_mb,
                    inventory_delta_mb=inv_delta_mb,
                    overhead_mb=overhead_mb,
                )
            )
