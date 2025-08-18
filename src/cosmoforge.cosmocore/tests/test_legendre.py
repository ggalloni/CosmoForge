#!/usr/bin/env python3
"""
Comprehensive benchmark and test for all Legendre polynomial optimizations.
Includes 00, 22, and 02 cases with unified and individual implementations.
"""

import time

import numpy as np
from numba import njit

from cosmocore import (
    legendre_00,
    legendre_00_inplace,
    legendre_02,
    legendre_02_inplace,
    legendre_22,
    legendre_22_inplace,
    legendre_unified,
    legendre_unified_inplace,
)


@njit(cache=True)
def original_legendre_00(scalar_prod, lmax):
    """Original 00 implementation."""
    legendre = np.ones(lmax, dtype=np.float64)
    legendre[0] = scalar_prod
    legendre[1] = 1.5 * scalar_prod * scalar_prod - 0.5
    for ell in range(3, lmax + 1):
        legendre[ell - 1] = (
            scalar_prod * (2 * ell - 1) * legendre[ell - 2]
            - (ell - 1) * legendre[ell - 3]
        ) / ell
    return legendre


@njit(cache=True)
def original_legendre_22(scalar_prod, lmax):
    """Original 22 implementation."""
    legendre = np.zeros(lmax, dtype=np.float64)
    legendre[1] = 3.0
    for ell in range(3, lmax + 1):
        legendre[ell - 1] = (
            scalar_prod * (2 * ell - 1) * legendre[ell - 2]
            - (ell + 1) * legendre[ell - 3]
        ) / (ell - 2)
    return legendre


@njit(cache=True)
def original_legendre_02(scalar_prod, lmax):
    """Original 02 implementation from compute_02_contribution."""
    legendre = np.zeros(lmax, dtype=np.float64)
    legendre[1] = 3.0 * (1.0 - scalar_prod * scalar_prod)
    for ell in range(3, lmax + 1):
        legendre[ell - 1] = (
            scalar_prod * (2 * ell - 1) * legendre[ell - 2]
            - (ell + 1) * legendre[ell - 3]
        ) / (ell - 2)
    return legendre


def benchmark_function(func, args, name, n_runs=1000):
    """Benchmark a function."""
    # Warm up JIT
    for _ in range(10):
        func(*args)

    start_time = time.perf_counter()
    for _ in range(n_runs):
        result = func(*args)
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / n_runs * 1e6  # microseconds
    print(f"{name:35s}: {avg_time:8.2f} µs/call")
    return result


def benchmark_inplace(func, scalar_prod, buffer, name, n_runs=1000, **kwargs):
    """Benchmark in-place function."""
    # Warm up JIT
    for _ in range(10):
        func(scalar_prod, buffer, **kwargs)

    start_time = time.perf_counter()
    for _ in range(n_runs):
        func(scalar_prod, buffer, **kwargs)
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / n_runs * 1e6
    print(f"{name:35s}: {avg_time:8.2f} µs/call")


def benchmark_inplace_22(
    func, scalar_prod, buffer, buffer_f1, buffer_f2, name, n_runs=1000
):
    """Benchmark in-place function for 22 case with f1, f2 buffers."""
    # Warm up JIT
    for _ in range(10):
        func(scalar_prod, buffer, buffer_f1, buffer_f2)

    start_time = time.perf_counter()
    for _ in range(n_runs):
        func(scalar_prod, buffer, buffer_f1, buffer_f2)
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / n_runs * 1e6
    print(f"{name:35s}: {avg_time:8.2f} µs/call")


def correctness():
    """Test that all implementations produce the same results."""
    print("Testing correctness...")
    scalar_prod = 0.5
    lmax = 50
    rtol = 1e-14

    # Test 00 case
    orig_00 = original_legendre_00(scalar_prod, lmax)
    opt_00 = legendre_00(scalar_prod, lmax)
    unified_00 = legendre_unified(scalar_prod, lmax, "00")

    buffer_00 = np.empty(lmax, dtype=np.float64)
    legendre_00_inplace(scalar_prod, buffer_00)

    buffer_unified_00 = np.empty(lmax, dtype=np.float64)
    legendre_unified_inplace(scalar_prod, buffer_unified_00, "00")

    np.testing.assert_allclose(orig_00, opt_00, rtol=rtol)
    np.testing.assert_allclose(orig_00, unified_00, rtol=rtol)
    np.testing.assert_allclose(orig_00, buffer_00, rtol=rtol)
    np.testing.assert_allclose(orig_00, buffer_unified_00, rtol=rtol)
    print("✓ 00 case: All implementations match")

    # Test 22 case
    orig_22 = original_legendre_22(scalar_prod, lmax)
    opt_22 = legendre_22(scalar_prod, lmax)
    unified_22 = legendre_unified(scalar_prod, lmax, "22")

    buffer_22 = np.empty(lmax, dtype=np.float64)
    buffer_f1 = np.empty(lmax, dtype=np.float64)
    buffer_f2 = np.empty(lmax, dtype=np.float64)
    legendre_22_inplace(scalar_prod, buffer_22, buffer_f1, buffer_f2)

    buffer_unified_22 = np.empty(lmax, dtype=np.float64)
    legendre_unified_inplace(scalar_prod, buffer_unified_22, "22")

    np.testing.assert_allclose(orig_22, opt_22, rtol=rtol)
    np.testing.assert_allclose(orig_22, unified_22, rtol=rtol)
    np.testing.assert_allclose(orig_22, buffer_22, rtol=rtol)
    np.testing.assert_allclose(orig_22, buffer_unified_22, rtol=rtol)
    print("✓ 22 case: All implementations match")

    # Test 02 case
    orig_02 = original_legendre_02(scalar_prod, lmax)
    opt_02 = legendre_02(scalar_prod, lmax)
    unified_02 = legendre_unified(scalar_prod, lmax, "02")

    buffer_02 = np.empty(lmax, dtype=np.float64)
    legendre_02_inplace(scalar_prod, buffer_02)

    buffer_unified_02 = np.empty(lmax, dtype=np.float64)
    legendre_unified_inplace(scalar_prod, buffer_unified_02, "02")

    np.testing.assert_allclose(orig_02, opt_02, rtol=rtol)
    np.testing.assert_allclose(orig_02, unified_02, rtol=rtol)
    np.testing.assert_allclose(orig_02, buffer_02, rtol=rtol)
    np.testing.assert_allclose(orig_02, buffer_unified_02, rtol=rtol)
    print("✓ 02 case: All implementations match")

    print("All correctness tests passed!\n")


def test_all_legendre():
    print("Comprehensive Legendre Polynomial Benchmark")
    print("=" * 60)

    correctness()

    # Benchmark parameters
    scalar_prod = 0.5
    lmax_values = [2, 5, 10, 30, 50, 100, 200]
    n_runs = 1000

    for lmax in lmax_values:
        print(f"Performance Benchmark: lmax = {lmax}")
        print("-" * 40)

        # 00 Case benchmarks
        print("P_l polynomials (00 case):")
        benchmark_function(original_legendre_00, (scalar_prod, lmax), "Original", n_runs)
        benchmark_function(legendre_00, (scalar_prod, lmax), "Optimized", n_runs)
        benchmark_function(legendre_unified, (scalar_prod, lmax, "00"), "Unified", n_runs)

        buffer = np.empty(lmax, dtype=np.float64)
        buffer_f1 = np.empty(lmax, dtype=np.float64)
        buffer_f2 = np.empty(lmax, dtype=np.float64)
        benchmark_inplace(
            legendre_00_inplace, scalar_prod, buffer, "Optimized (in-place)", n_runs
        )
        benchmark_inplace(
            legendre_unified_inplace,
            scalar_prod,
            buffer,
            "Unified (in-place)",
            n_runs,
            spin_case="00",
        )

        # 22 Case benchmarks
        print("P_l^{22} polynomials (22 case):")
        benchmark_function(original_legendre_22, (scalar_prod, lmax), "Original", n_runs)
        benchmark_function(legendre_22, (scalar_prod, lmax), "Optimized", n_runs)
        benchmark_function(legendre_unified, (scalar_prod, lmax, "22"), "Unified", n_runs)

        benchmark_inplace_22(
            legendre_22_inplace,
            scalar_prod,
            buffer,
            buffer_f1,
            buffer_f2,
            "Optimized (in-place)",
            n_runs,
        )
        benchmark_inplace(
            legendre_unified_inplace,
            scalar_prod,
            buffer,
            "Unified (in-place)",
            n_runs,
            spin_case="22",
        )

        # 02 Case benchmarks
        print("P_l^{02} polynomials (02 case):")
        benchmark_function(original_legendre_02, (scalar_prod, lmax), "Original", n_runs)
        benchmark_function(legendre_02, (scalar_prod, lmax), "Optimized", n_runs)
        benchmark_function(legendre_unified, (scalar_prod, lmax, "02"), "Unified", n_runs)

        benchmark_inplace(
            legendre_02_inplace, scalar_prod, buffer, "Optimized (in-place)", n_runs
        )
        benchmark_inplace(
            legendre_unified_inplace,
            scalar_prod,
            buffer,
            "Unified (in-place)",
            n_runs,
            spin_case="02",
        )

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    test_all_legendre()
