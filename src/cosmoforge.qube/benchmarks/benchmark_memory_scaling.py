"""Benchmark: per-stage RSS memory profile across (field, nside) at given fsky.

Mirrors ``benchmark_pixel_direct_only.py`` cell-by-cell so the timing and
memory JSONs can be read side by side. Writes one JSON file containing,
for each cell:
  * peak RSS across the whole pipeline (headline number)
  * per-stage peak RSS, entry/exit RSS and delta for "fisher" (run) and
    "spectra" (run) stages
  * the same metadata block as the timing benchmarks

The auto-selector picks the basis path; this matches what users will run
in production. Single-rank / full-thread, matching
``benchmark_pixel_direct_scaling.sbatch``.

Usage:
  uv run python -u benchmark_memory_scaling.py
  uv run python -u benchmark_memory_scaling.py --fsky 0.01 \\
      --nsides 16,32,64,128 --fields T,QU
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

from _bench_utils import save_results
from benchmark_pixel_direct_only import (
    FIELD_DEFS as _PIXEL_FIELD_DEFS,
)
from benchmark_pixel_direct_only import (
    generate_test_inputs,
    write_temp_config,
)

# Extend the pixel benchmark's field defs with the joint TQU case, used
# for the harmonic heavy run. (spins, labels, physical_labels)
FIELD_DEFS = {
    **_PIXEL_FIELD_DEFS,
    "TQU": ([0, 2], ["T", "E", "B"], ["T", "Q", "U"]),
}

from cosmocore import Bins
from cosmocore.benchmarks import MemMonitor, StageProfiler, measure
from qube import Fisher, Spectra

DEFAULT_FSKY = 0.10
DEFAULT_NSIDES = [16, 32, 64]
DEFAULT_FIELDS = ["T", "QU"]
TARGET_NBINS = 6
POLL_INTERVAL_S = 0.05


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fsky", type=float, default=DEFAULT_FSKY)
    p.add_argument(
        "--nsides",
        type=str,
        default=",".join(str(n) for n in DEFAULT_NSIDES),
        help="Comma-separated nside values to sweep.",
    )
    p.add_argument(
        "--fields",
        type=str,
        default=",".join(DEFAULT_FIELDS),
        help="Comma-separated field labels to run (T, QU).",
    )
    p.add_argument("--target-nbins", type=int, default=TARGET_NBINS)
    p.add_argument(
        "--lmax",
        type=int,
        default=None,
        help="Override lmax. Default: 2*nside per cell.",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=POLL_INTERVAL_S,
        help="MemMonitor polling interval in seconds.",
    )
    p.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=("auto", "harmonic", "pixel"),
        help="Computation basis to force; 'auto' lets the selector decide.",
    )
    p.add_argument(
        "--no-cache-derivatives",
        action="store_true",
        help=(
            "Disable Fisher's derivative caching (cache_derivatives=False). "
            "Skips holding the n_params dC blocks across the trace loop; "
            "Spectra cannot reuse them so it recomputes from scratch."
        ),
    )
    p.add_argument("--suffix", type=str, default="")
    return p.parse_args()


def _profile_cell(config_file, bins, poll_interval, method, cache_derivatives):
    """Run Fisher.run() and Spectra.run() under a polled MemMonitor.

    Returns (timings, mem_samples_dicts, peak_overall_mb).
    """
    samples: list = []
    timings: dict = {}

    with MemMonitor(interval_s=poll_interval) as mon:
        baseline_mb = mon.current_rss_mb

        fisher = Fisher(
            config_file,
            compression={"method": method},
            cache_derivatives=cache_derivatives,
        )
        # Sub-stage profiler: emits one MemSample per `with self._stage(...)`
        # block inside Fisher.run() / .compute(), each carrying a per-array
        # inventory snapshot of arrays reachable from `fisher` at exit.
        fisher._profiler = StageProfiler(
            mon, samples, snapshot_targets=[fisher], inventory_min_mb=1.0
        )
        if bins is not None:
            fisher.set_binning(bins)

        with measure(mon, "fisher_run", samples):
            t0 = time.perf_counter()
            fisher.run()
            timings["fisher_total_s"] = time.perf_counter() - t0

        n_kept = (
            fisher.basis_manager.n_kept
            if hasattr(fisher, "basis_manager") and fisher.basis_manager
            else None
        )
        timings["n_modes"] = n_kept
        timings["n_pix"] = sum(fisher.collection.n_active)

        spectra = Spectra(config_file, fisher=fisher, compression={"method": method})
        spectra._profiler = StageProfiler(
            mon, samples, snapshot_targets=[spectra], inventory_min_mb=1.0
        )
        if bins is not None:
            spectra.set_binning(bins)

        with measure(mon, "spectra_run", samples):
            t0 = time.perf_counter()
            spectra.run()
            timings["spectra_total_s"] = time.perf_counter() - t0

        timings["nsims"] = spectra.params.nsims
        peak_overall_mb = mon.max_rss_mb

    return (
        timings,
        [s.to_dict() for s in samples],
        peak_overall_mb,
        baseline_mb,
    )


def main():
    args = _parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    fsky = args.fsky
    nside_values = [int(s.strip()) for s in args.nsides.split(",") if s.strip()]
    field_labels = [s.strip() for s in args.fields.split(",") if s.strip()]
    target_nbins = args.target_nbins

    for label in field_labels:
        if label not in FIELD_DEFS:
            raise ValueError(f"Unknown field '{label}'. Choose from {list(FIELD_DEFS)}")

    print(
        f"Memory benchmark: fsky={fsky}, nsides={nside_values}, "
        f"fields={field_labels}, target_nbins={target_nbins}, "
        f"method={args.method}, "
        f"cache_derivatives={not args.no_cache_derivatives}, "
        f"poll_interval={args.poll_interval}s"
    )

    results: dict = {}

    for nside in nside_values:
        lmax = args.lmax if args.lmax is not None else 2 * nside
        lmax_sim = max(4 * nside, 2 * lmax)
        delta_ell = max(2, (lmax - 1) // target_nbins)

        for field_label in field_labels:
            spins, labels, physical_labels = FIELD_DEFS[field_label]
            print(f"\n{'=' * 60}")
            print(
                f"{field_label} nside={nside}, lmax={lmax}, "
                f"delta_ell={delta_ell}, fsky={fsky}"
            )
            print(f"{'=' * 60}")

            cov, mask, sim_maps, fwhmarcmin = generate_test_inputs(
                nside, lmax, spins, physical_labels, lmax_sim=lmax_sim, fsky=fsky
            )

            run_label = f"{field_label}_nside{nside}_auto"
            with tempfile.TemporaryDirectory(
                dir="sims", prefix=f"memb_{run_label}_"
            ) as tmpdir:
                config_file = write_temp_config(
                    tmpdir,
                    nside,
                    lmax,
                    spins,
                    labels,
                    physical_labels,
                    cov,
                    mask,
                    sim_maps,
                    fwhmarcmin=fwhmarcmin,
                )
                bins = Bins.fromdeltal(2, lmax, delta_ell)
                try:
                    timings, mem_samples, peak_mb, baseline_mb = _profile_cell(
                        config_file,
                        bins,
                        args.poll_interval,
                        method=args.method,
                        cache_derivatives=not args.no_cache_derivatives,
                    )
                    print(f"  baseline RSS:    {baseline_mb:>10.1f} MB")
                    print(f"  peak RSS:        {peak_mb:>10.1f} MB")
                    for s in mem_samples:
                        print(
                            f"    {s['name']:<14} "
                            f"entry={s['rss_at_entry_mb']:>8.1f} MB  "
                            f"peak={s['rss_peak_mb']:>8.1f} MB  "
                            f"delta={s['rss_delta_mb']:>+8.1f} MB"
                        )
                    results[run_label] = {
                        "nside": nside,
                        "fsky": fsky,
                        "lmax": lmax,
                        "lmax_sim": lmax_sim,
                        "delta_ell": delta_ell,
                        "method": args.method,
                        "cache_derivatives": not args.no_cache_derivatives,
                        "spins": spins,
                        "field_label": field_label,
                        "baseline_rss_mb": baseline_mb,
                        "peak_rss_mb": peak_mb,
                        "stages": mem_samples,
                        **timings,
                    }
                except Exception as e:
                    print(f"  FAILED: {e}")
                    results[run_label] = {"error": str(e)}

    out_name = f"benchmark_memory_scaling_fsky{fsky:.3f}".replace(".", "p")
    out_name += f"_{args.method}"
    if args.lmax is not None:
        out_name += f"_lmax{args.lmax}"
    if args.no_cache_derivatives:
        out_name += "_nocache"
    if args.suffix:
        out_name += f"_{args.suffix}"
    extra_md = {
        "poll_interval_s": args.poll_interval,
        "method": args.method,
        "cache_derivatives": not args.no_cache_derivatives,
    }
    out_path = save_results(out_name, results, extra_metadata=extra_md)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'=' * 80}")
    print(
        f"{'Config':<26} {'n_modes':>9} {'n_pix':>7} {'peak RSS':>11} "
        f"{'fisher peak':>13} {'spectra peak':>14}"
    )
    print(f"{'=' * 80}")
    for key, t in results.items():
        if "error" in t:
            print(f"{key:<26} FAILED: {t['error']}")
            continue
        stages = {s["name"]: s for s in t["stages"]}
        f_peak = stages.get("fisher_run", {}).get("rss_peak_mb", float("nan"))
        s_peak = stages.get("spectra_run", {}).get("rss_peak_mb", float("nan"))
        print(
            f"{key:<26} {t.get('n_modes', '?'):>9} {t['n_pix']:>7} "
            f"{t['peak_rss_mb']:>9.0f} MB "
            f"{f_peak:>11.0f} MB {s_peak:>12.0f} MB"
        )


if __name__ == "__main__":
    main()
