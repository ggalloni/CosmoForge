"""Per-stage RSS memory profile with one (field, nside) cell per process.

Referee-driven fix: benchmark_memory_scaling.py sweeps all cells inside a
single Python process, so each cell's "baseline RSS" (and therefore its
absolute peak RSS) inherits allocator-retained memory from every cell that
ran before it — for the largest cell this inflated the baseline to ~13 GiB
and was misreported as "Python interpreter baseline" in the paper.

This driver launches benchmark_memory_scaling.py once per cell in a fresh
subprocess, so the baseline is a genuinely cold interpreter and the peak is
the isolated single-run number. Results from the per-cell JSONs are merged
into a single benchmark_memory_isolated_* file with the same schema.

Usage (same knobs as benchmark_memory_scaling.py):
    uv run python benchmark_memory_isolated.py --fsky 0.1 --nsides 16,32,64 \
        --fields T,QU --method pixel [--no-cache-derivatives] [--suffix tag]

Any flag not listed in --help (--target-nbins, --poll-interval, ...) is
forwarded to benchmark_memory_scaling.py verbatim, so see that script's
--help for the full set.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _bench_utils import RESULTS_DIR, save_results

SCRIPT = Path(__file__).resolve().parent / "benchmark_memory_scaling.py"


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--fsky", type=float, default=0.1)
    p.add_argument("--nsides", type=str, default="16,32,64")
    p.add_argument("--fields", type=str, default="T,QU")
    p.add_argument("--method", type=str, default="auto")
    # --lmax is parsed because the child embeds it in its output filename and
    # we need that to find the file again. Everything else the child accepts
    # (--target-nbins, --poll-interval, ...) is forwarded verbatim.
    p.add_argument("--lmax", type=int, default=None)
    p.add_argument("--no-cache-derivatives", action="store_true")
    p.add_argument("--suffix", type=str, default="")
    return p.parse_known_args()


def _cell_result_path(fsky, method, lmax, nocache, cell_suffix):
    name = f"benchmark_memory_scaling_fsky{fsky:.3f}".replace(".", "p")
    name += f"_{method}"
    if lmax is not None:
        name += f"_lmax{lmax}"
    if nocache:
        name += "_nocache"
    name += f"_{cell_suffix}"
    return RESULTS_DIR / f"{name}_results.json"


def main():
    args, passthrough = _parse_args()
    nsides = [int(s) for s in args.nsides.split(",") if s.strip()]
    fields = [s.strip() for s in args.fields.split(",") if s.strip()]

    merged: dict = {}
    for nside in nsides:
        for field in fields:
            cell_suffix = f"iso_{field}_n{nside}"
            if args.suffix:
                cell_suffix += f"_{args.suffix}"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--fsky",
                str(args.fsky),
                "--nsides",
                str(nside),
                "--fields",
                field,
                "--method",
                args.method,
                "--suffix",
                cell_suffix,
            ]
            if args.lmax is not None:
                cmd += ["--lmax", str(args.lmax)]
            if args.no_cache_derivatives:
                cmd += ["--no-cache-derivatives"]
            cmd += passthrough

            cell_path = _cell_result_path(
                args.fsky,
                args.method,
                args.lmax,
                args.no_cache_derivatives,
                cell_suffix,
            )
            # The path is rebuilt from the child's naming convention, so a
            # child that exits 0 while writing some other name would leave us
            # reading an earlier run's file. Clear it first: then a missing
            # file can only mean this run did not produce one. This driver
            # exists to fix a provenance defect; it must not invent one.
            cell_path.unlink(missing_ok=True)

            print(f"\n### isolated cell: {field} nside={nside} (fresh process)")
            proc = subprocess.run(cmd, cwd=str(SCRIPT.parent))

            if proc.returncode != 0:
                key = f"{field}_nside{nside}_{args.method}"
                merged[key] = {"error": f"subprocess exit {proc.returncode}"}
                continue
            with open(cell_path) as f:
                cell = json.load(f)
            merged.update(cell["results"])

    out_name = f"benchmark_memory_isolated_fsky{args.fsky:.3f}".replace(".", "p")
    out_name += f"_{args.method}"
    if args.no_cache_derivatives:
        out_name += "_nocache"
    if args.suffix:
        out_name += f"_{args.suffix}"
    out_path = save_results(
        out_name, merged, extra_metadata={"isolation": "one cell per process"}
    )
    print(f"\nMerged isolated results saved to {out_path}")

    print(f"\n{'Config':<26} {'baseline RSS':>14} {'peak RSS':>11}")
    for key, t in merged.items():
        if "error" in t:
            print(f"{key:<26} ERROR: {t['error']}")
            continue
        print(f"{key:<26} {t['baseline_rss_mb']:>11.0f} MB {t['peak_rss_mb']:>8.0f} MB")


if __name__ == "__main__":
    main()
