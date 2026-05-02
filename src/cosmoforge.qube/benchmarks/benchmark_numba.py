"""
Benchmark 4: Numba JIT acceleration

Compares execution time of key Numba-accelerated functions:
- Legendre polynomial evaluation
- Signal matrix construction

Run twice:
  uv run python benchmark_numba.py                    # JIT enabled (default)
  NUMBA_DISABLE_JIT=1 uv run python benchmark_numba.py  # Pure Python

Usage: uv run python benchmark_numba.py
"""

import os
import time

import numpy as np

# Check if JIT is disabled
jit_disabled = os.environ.get("NUMBA_DISABLE_JIT", "0") == "1"
mode = "pure_python" if jit_disabled else "jit"
print(f"Mode: {mode}")
print(f"NUMBA_DISABLE_JIT = {os.environ.get('NUMBA_DISABLE_JIT', 'not set')}")


def benchmark_legendre():
    """Benchmark Legendre polynomial evaluation."""
    import healpy as hp

    from cosmocore.basics import legendre_plm

    nside = 16
    lmax = 32
    npix = 12 * nside**2

    # Use real pixel positions
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    plm = np.zeros((lmax + 1, lmax + 1), dtype=np.float64)

    # Warmup (JIT compilation on first call)
    t0 = time.perf_counter()
    legendre_plm(cos_theta[0], sin_theta[0], plm)
    t_warmup = time.perf_counter() - t0

    # Timed run: evaluate for all pixels
    n_repeats = 3
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        for ipix in range(npix):
            legendre_plm(cos_theta[ipix], sin_theta[ipix], plm)
        times.append(time.perf_counter() - t0)

    return {
        "function": "legendre_plm",
        "description": f"P_lm for {npix} pixels, lmax={lmax}",
        "warmup": t_warmup,
        "mean": np.mean(times),
        "std": np.std(times),
        "n_repeats": n_repeats,
        "nside": nside,
        "lmax": lmax,
        "npix": npix,
    }


def benchmark_signal_matrix():
    """Benchmark signal matrix construction using Fisher setup."""
    from qube import Fisher

    nside = 16
    lmax = 32

    # Use the existing config to set up a Fisher instance (setup only, no traces)
    config_file = "sims/config.yaml"
    fisher = Fisher(config_file)
    fisher.setup_fields()
    fisher.setup_geometry()
    fisher.setup_covariance_matrices()
    fisher.setup_cls(lmax=fisher.lmax_signal)
    fisher.setup_beams(lmax=fisher.lmax_signal)

    from cosmocore.pixel import compute_signal_matrix

    # Match the production call site (qube/fisher.py:204): the kernel's
    # internal `cl * legendre[1:]` only broadcasts if the legendre buffer
    # length matches the cls length, which means `lmax` here must equal
    # `lmax_signal` (typically 4*nside), not the analysis `lmax`.
    lmax_signal = fisher.lmax_signal

    ntot = sum(fisher.collection.n_active)
    S = np.zeros((ntot, ntot), dtype=np.float64)

    # Warmup
    t0 = time.perf_counter()
    compute_signal_matrix(S, lmax=lmax_signal, fields=fisher.collection)
    t_warmup = time.perf_counter() - t0

    # Timed run
    n_repeats = 3
    times = []
    for _ in range(n_repeats):
        S[:] = 0
        t0 = time.perf_counter()
        compute_signal_matrix(S, lmax=lmax_signal, fields=fisher.collection)
        times.append(time.perf_counter() - t0)

    return {
        "function": "compute_signal_matrix",
        "description": f"S matrix ({ntot}x{ntot}), lmax_signal={lmax_signal}",
        "warmup": t_warmup,
        "mean": np.mean(times),
        "std": np.std(times),
        "n_repeats": n_repeats,
        "nside": nside,
        "lmax": lmax,
        "lmax_signal": lmax_signal,
        "n_pix_active": ntot,
    }


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("\n=== Legendre polynomial benchmark ===")
    leg_results = benchmark_legendre()
    print(f"  Warmup (includes JIT compilation): {leg_results['warmup']:.3f}s")
    print(
        f"  Mean execution time: {leg_results['mean']:.4f}s "
        f"(+/- {leg_results['std']:.4f}s)"
    )

    print("\n=== Signal matrix benchmark ===")
    try:
        sig_results = benchmark_signal_matrix()
        print(f"  Warmup (includes JIT compilation): {sig_results['warmup']:.3f}s")
        print(
            f"  Mean execution time: {sig_results['mean']:.4f}s "
            f"(+/- {sig_results['std']:.4f}s)"
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        sig_results = {"error": str(e)}

    # Save results
    import json

    from _bench_utils import RESULTS_DIR, save_results

    results = {
        "mode": mode,
        "legendre": leg_results,
        "signal_matrix": sig_results,
    }
    out_path = save_results(f"benchmark_numba_{mode}", results)
    print(f"\nResults saved to {out_path}")

    # If both result files exist, print comparison
    jit_path = RESULTS_DIR / "benchmark_numba_jit_results.json"
    py_path = RESULTS_DIR / "benchmark_numba_pure_python_results.json"
    if jit_path.exists() and py_path.exists():
        with open(jit_path) as f:
            jit = json.load(f).get("results", {})
        with open(py_path) as f:
            py = json.load(f).get("results", {})

        print(f"\n{'=' * 50}")
        print(f"{'Function':<25} {'JIT':>10} {'Python':>10} {'Speedup':>10}")
        print(f"{'=' * 50}")
        for key in ["legendre", "signal_matrix"]:
            if "error" in jit.get(key, {}) or "error" in py.get(key, {}):
                continue
            t_jit = jit[key]["mean"]
            t_py = py[key]["mean"]
            speedup = t_py / t_jit if t_jit > 0 else float("inf")
            print(f"{key:<25} {t_jit:>9.4f}s {t_py:>9.4f}s {speedup:>9.1f}x")


if __name__ == "__main__":
    main()
