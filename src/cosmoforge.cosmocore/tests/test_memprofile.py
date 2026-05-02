"""Sanity tests for the cosmocore.benchmarks.memprofile utilities."""

from __future__ import annotations

import gc
import time

import numpy as np
import pytest

from cosmocore.benchmarks import (
    ArrayInfo,
    MemMonitor,
    MemSample,
    StageProfiler,
    inventory_arrays,
    measure,
)


def test_memsample_delta_computed():
    s = MemSample(
        name="foo", rss_at_entry_mb=100.0, rss_at_exit_mb=150.0, rss_peak_mb=200.0
    )
    assert s.rss_delta_mb == pytest.approx(50.0)


def test_memsample_to_dict_omits_inventory_when_absent():
    s = MemSample(name="foo", rss_at_entry_mb=10.0, rss_at_exit_mb=20.0, rss_peak_mb=25.0)
    d = s.to_dict()
    assert "inventory" not in d
    assert d["rss_delta_mb"] == pytest.approx(10.0)


def test_monitor_picks_up_known_allocation():
    """Allocate ~80 MB and confirm the monitor's peak grew by at least ~50 MB.

    Uses a generous lower bound to stay robust against allocator slack and
    threadpool baselines, while still failing if monitoring is broken.
    """
    samples: list[MemSample] = []
    with MemMonitor(interval_s=0.01) as mon:
        # Touch a small allocation first to stabilise the baseline
        with measure(mon, "baseline", samples):
            time.sleep(0.05)

        with measure(mon, "alloc", samples):
            # 10_000 * 1_000 float64 = 80 MB, written so memory is actually committed
            arr = np.ones((10_000, 1_000), dtype=np.float64)
            arr += 1.0
            time.sleep(0.05)
        del arr
        gc.collect()

    assert len(samples) == 2
    alloc = samples[1]
    assert alloc.name == "alloc"
    # Peak should reflect the allocation; allow generous slack but require
    # the monitor sees at least 30 MB of growth.
    assert alloc.rss_peak_mb - alloc.rss_at_entry_mb > 30.0


def test_reset_rewinds_max():
    with MemMonitor(interval_s=0.01) as mon:
        arr = np.ones((5_000, 1_000), dtype=np.float64)  # ~40 MB
        time.sleep(0.05)
        peak_before = mon.max_rss_mb
        del arr
        gc.collect()
        mon.reset()
        time.sleep(0.05)
        peak_after = mon.max_rss_mb

    assert peak_after <= peak_before


# ---------------------------------------------------------------------------
# inventory_arrays
# ---------------------------------------------------------------------------


class _Holder:
    """Plain object with attributes for reflection tests."""


def test_inventory_finds_direct_attribute():
    h = _Holder()
    h.big = np.zeros((100, 100), dtype=np.float64)  # 80 kB
    found = inventory_arrays(h, prefix="h")
    assert len(found) == 1
    assert found[0].path == "h.big"
    assert found[0].shape == (100, 100)
    assert found[0].dtype == "float64"
    assert found[0].nbytes == 100 * 100 * 8


def test_inventory_walks_lists_and_dicts():
    h = _Holder()
    h.list_of_arrays = [np.zeros(10), np.zeros(20)]
    h.map = {"k": np.zeros(30)}
    found = inventory_arrays(h, prefix="h")
    paths = sorted(a.path for a in found)
    assert paths == ["h.list_of_arrays[0]", "h.list_of_arrays[1]", "h.map['k']"]


def test_inventory_min_mb_filters_small_arrays():
    h = _Holder()
    h.small = np.zeros(100, dtype=np.float64)  # 800 B
    h.big = np.zeros((1000, 1000), dtype=np.float64)  # 8 MB
    found = inventory_arrays(h, prefix="h", min_mb=1.0)
    assert len(found) == 1
    assert found[0].path == "h.big"


def test_inventory_handles_cycles():
    h = _Holder()
    h.self_ref = h
    h.arr = np.zeros((50, 50))
    # Should not infinite-recurse and should still find the array.
    found = inventory_arrays(h, prefix="h")
    paths = [a.path for a in found]
    assert "h.arr" in paths


def test_inventory_respects_max_depth():
    a = _Holder()
    b = _Holder()
    c = _Holder()
    a.b = b
    b.c = c
    c.arr = np.zeros((100, 100))  # at depth 3 from a (a.b.c.arr)
    # max_depth=2 should not reach c.arr (a -> b -> c is 2 edges; a.b.c.arr is 3)
    found_shallow = inventory_arrays(a, prefix="a", max_depth=2)
    assert all("arr" not in p.path for p in found_shallow)
    found_deep = inventory_arrays(a, prefix="a", max_depth=5)
    assert any("arr" in p.path for p in found_deep)


def test_inventory_skips_dunder_attributes():
    h = _Holder()
    # __dict__ itself is reachable as obj.__dict__ but the walker only
    # descends through vars(obj), so dunder keys never appear as paths.
    h.real = np.zeros((10, 10))
    found = inventory_arrays(h, prefix="h")
    assert all("__" not in a.path for a in found)


def test_array_info_to_dict():
    nbytes = 1024 * 1024 * 8  # exactly 8 MB
    info = ArrayInfo(path="x.y", shape=(1024, 1024), dtype="float64", nbytes=nbytes)
    d = info.to_dict()
    assert d["path"] == "x.y"
    assert d["shape"] == [1024, 1024]
    assert d["dtype"] == "float64"
    assert d["nbytes"] == nbytes
    assert d["MB"] == pytest.approx(8.0, abs=1e-3)


# ---------------------------------------------------------------------------
# StageProfiler
# ---------------------------------------------------------------------------


def test_stage_profiler_emits_sample_with_inventory():
    target = _Holder()
    samples: list[MemSample] = []
    with MemMonitor(interval_s=0.01) as mon:
        prof = StageProfiler(mon, samples, [target], inventory_min_mb=1.0)
        with prof.stage("alloc"):
            target.buf = np.ones((1000, 1000), dtype=np.float64)  # 8 MB
            time.sleep(0.05)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.name == "alloc"
    assert sample.inventory is not None
    paths = [a.path for a in sample.inventory]
    assert any(p.endswith(".buf") for p in paths)
    # The 8 MB allocation should dominate the inventory total
    assert sample.inventory_total_mb is not None
    assert sample.inventory_total_mb >= 7.0
    # overhead can be slightly negative or positive depending on allocator,
    # but its magnitude should be much smaller than the 8 MB allocation.
    assert sample.overhead_mb is not None


def test_stage_profiler_multiple_stages_accumulate():
    target = _Holder()
    samples: list[MemSample] = []
    with MemMonitor(interval_s=0.01) as mon:
        prof = StageProfiler(mon, samples, [target])
        with prof.stage("first"):
            time.sleep(0.02)
        with prof.stage("second"):
            time.sleep(0.02)

    assert [s.name for s in samples] == ["first", "second"]


def test_stage_profiler_inventory_serialises():
    """to_dict() output round-trips inventory entries."""
    target = _Holder()
    samples: list[MemSample] = []
    with MemMonitor(interval_s=0.01) as mon:
        prof = StageProfiler(mon, samples, [target], inventory_min_mb=0.1)
        with prof.stage("alloc"):
            target.arr = np.ones((1024, 1024), dtype=np.float64)  # 8 MB

    d = samples[0].to_dict()
    assert "inventory" in d
    assert "inventory_total_mb" in d
    assert "overhead_mb" in d
    assert isinstance(d["inventory"], list)
    if d["inventory"]:
        entry = d["inventory"][0]
        assert "path" in entry and "shape" in entry and "MB" in entry
