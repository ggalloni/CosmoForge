"""
Benchmark: Harmonic vs pixel-direct basis at small fsky.

The pixel-direct path becomes competitive when n_pix^3 (with the per-bin
prefactor) drops below n_modes^3. At fixed nside, n_pix shrinks with
the mask while n_modes ~ lmax^2 grows, so the crossover sits at small
fsky and increasing lmax — the regime this benchmark targets.

Geometry: polar cap centred on the north pole, fsky ~ 0.1.
Sweep: lmax in {8, 16, 24, 32, 48}, T-only and QU.
Methods:
  - harmonic:     V-based, full SMW pipeline
  - pixel_direct: pixel basis in direct mode (use_direct=True, no V,
                  full pixel-space ops)
  - auto:         factory picks harmonic vs pixel-direct via the cost
                  model (n_modes^3 vs (n_bins+1)*n_pix^3)

Each (field, lmax, method) triple runs in a fresh Python subprocess so
every measurement pays a full Numba JIT warmup. This eliminates the
within-process JIT-cache bias that would otherwise systematically favour
methods invoked later in a single-process run.

Usage:
    mpirun -n 1 uv run python -u benchmark_pixel_vs_harmonic.py
        # driver: spawns one subprocess per config, accumulates JSON
    uv run python benchmark_pixel_vs_harmonic.py \\
        --field T --lmax 8 --method harmonic
        # worker: runs a single config, prints one-line JSON to stdout
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import healpy as hp
import numpy as np
import yaml

from cosmocore import Bins
from qube import Fisher, Spectra


def _gaussian_fwhm_for_lmax(lmax, beam_at_lmax=0.5):
    """Return FWHM in arcmin for a Gaussian beam with b(lmax) = beam_at_lmax."""
    sigma = np.sqrt(-2 * np.log(beam_at_lmax) / (lmax * (lmax + 1)))
    return float(np.degrees(sigma * np.sqrt(8 * np.log(2))) * 60)


def _polar_cap_mask(nside, n_fields, fsky):
    """Return a mask selecting a polar cap with the requested sky fraction."""
    radius = np.arccos(1.0 - 2.0 * fsky)
    pole_vec = hp.ang2vec(0.0, 0.0)
    cap_pixels = hp.query_disc(nside, pole_vec, radius)
    mask = np.zeros((n_fields, hp.nside2npix(nside)))
    mask[:, cap_pixels] = 1.0
    return mask


def generate_test_inputs(nside, lmax, spins, physical_labels, lmax_sim, fsky):
    npix = 12 * nside**2
    n_fields = len(physical_labels)
    mask = _polar_cap_mask(nside, n_fields, fsky)

    raw_cls = np.loadtxt("sims/tau0.06_dls_r_likelihood_r0.0.txt")
    ells_file = raw_cls[:, 0].astype(int)
    dl2cl = np.ones(len(ells_file))
    dl2cl[ells_file > 0] = (
        2 * np.pi / (ells_file[ells_file > 0] * (ells_file[ells_file > 0] + 1))
    )

    cl_tt = np.zeros(lmax_sim + 1)
    cl_ee = np.zeros(lmax_sim + 1)
    cl_bb = np.zeros(lmax_sim + 1)
    cl_te = np.zeros(lmax_sim + 1)
    for i, ell_val in enumerate(ells_file):
        if ell_val <= lmax_sim:
            cl_tt[ell_val] = raw_cls[i, 1] * dl2cl[i]
            cl_ee[ell_val] = raw_cls[i, 2] * dl2cl[i]
            cl_bb[ell_val] = raw_cls[i, 3] * dl2cl[i]
            cl_te[ell_val] = raw_cls[i, 4] * dl2cl[i]

    # Use a beam fixed to the simulation lmax so the same input maps work
    # across the entire sweep — only the analysis cutoff (params.lmax) varies.
    fwhm_arcmin = _gaussian_fwhm_for_lmax(lmax_sim)
    fwhm_rad = np.radians(fwhm_arcmin / 60.0)
    beam = hp.gauss_beam(fwhm_rad, lmax=lmax_sim)

    # Set noise so C = N + S is uniformly well-conditioned across the sweep
    # AND the Fisher matrix remains invertible at the highest ℓ where the beam
    # suppresses signal most heavily. Use sigma² >= signal pixel-variance so
    # noise dominates and condition numbers stay near 1.
    cl_for_sigma = cl_ee if 2 in spins else cl_tt
    ell_arr = np.arange(lmax_sim + 1)
    sig_var_per_pix = np.sum((2 * ell_arr + 1) / (4 * np.pi) * cl_for_sigma * beam**2)
    sigma = float(np.sqrt(sig_var_per_pix))
    cov = np.eye(n_fields * npix) * sigma**2

    nsims = 10
    field_index_map = {"T": 0, "Q": 1, "U": 2, "E": 1, "B": 2}

    sim_maps = np.empty((n_fields, npix, nsims), dtype=np.float64)
    for i in range(nsims):
        np.random.seed(42 + i)
        alms = hp.synalm([cl_tt, cl_ee, cl_bb, cl_te], lmax=lmax_sim, new=True)
        if 2 in spins:
            hp.almxfl(alms[1], beam, inplace=True)
            hp.almxfl(alms[2], beam, inplace=True)
        else:
            for a in alms:
                hp.almxfl(a, beam, inplace=True)
        sim_tqu = hp.alm2map(alms, nside=nside, lmax=lmax_sim)
        for j, label in enumerate(physical_labels):
            sim_maps[j, :, i] = sim_tqu[field_index_map[label]]

    return cov, mask, sim_maps, fwhm_arcmin


def write_temp_config(
    tmpdir,
    nside,
    lmax,
    spins,
    labels,
    physical_labels,
    cov,
    mask,
    sim_maps,
    fwhmarcmin,
):
    nsims = sim_maps.shape[2]
    cov_file = os.path.join(tmpdir, "ncvm.bin")
    cov.tofile(cov_file)
    mask_file = os.path.join(tmpdir, "mask.fits")
    hp.write_map(mask_file, mask, overwrite=True)
    sim_file = os.path.join(tmpdir, "sims.npy")
    np.save(sim_file, sim_maps)

    config = {
        "nside": nside,
        "spins": spins,
        "labels": labels,
        "physical_labels": physical_labels,
        "do_cross": False,
        "maskfile": mask_file,
        "output_geometry_file": os.path.join(tmpdir, "geometry.dat"),
        "ordering": "RING",
        "inputclfile": os.path.abspath("sims/tau0.06_dls_r_likelihood_r0.0.txt"),
        "input_convention": "Dl",
        "covmatfile1": cov_file,
        "covmatfile2": cov_file,
        "lmax": lmax,
        "load_inverted": True,
        "calibration": 1.0,
        "smoothing_type": "gaussian",
        "fwhmarcmin": fwhmarcmin,
        "apply_pixwin": False,
        "beam_file": "",
        "outnoisecovmat1": os.path.join(tmpdir, "reduced_ncvm1.bin"),
        "outnoisecovmat2": os.path.join(tmpdir, "reduced_ncvm2.bin"),
        "feedback": 4,
        "outinvcovmatfile1": os.path.join(tmpdir, "invcov1.bin"),
        "outinvcovmatfile2": os.path.join(tmpdir, "invcov2.bin"),
        "outfilefisher": os.path.join(tmpdir, "fisher.dat"),
        "nsims": nsims,
        "inputmapfile1": sim_file,
        "inputmapfile2": sim_file,
        "outcovmatfile": os.path.join(tmpdir, "cov_matrix.dat"),
        "outerrfile": os.path.join(tmpdir, "errors.dat"),
        "remove_nb": False,
        "nspectra": len(labels) * (len(labels) + 1) // 2,
    }
    config_file = os.path.join(tmpdir, "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return config_file


def _compression_dict(method):
    """Build the basis-construction config for a method label.

    The benchmark labels are ``harmonic``, ``pixel_direct``, ``auto``.
    The label ``pixel_direct`` maps to ``method='pixel'`` plus
    ``use_direct=True`` so the basis runs the no-V direct pixel-space
    path that the auto selector picks at high lmax.
    """
    if method == "pixel_direct":
        return {"method": "pixel", "use_direct": True}
    return {"method": method}


def benchmark_fisher(config_file, method, bins=None):
    fisher = Fisher(config_file, compression=_compression_dict(method))
    if bins is not None:
        fisher.set_binning(bins)
    t0 = time.perf_counter()
    fisher.run()
    t_total = time.perf_counter() - t0
    dim = (
        fisher.basis_manager.dim
        if hasattr(fisher, "basis_manager") and fisher.basis_manager
        else None
    )
    return {
        "total": t_total,
        "n_modes": dim,
        "n_pix": sum(fisher.collection.n_active),
    }, fisher


def benchmark_spectra(config_file, fisher, method, bins=None):
    spectra = Spectra(config_file, fisher=fisher, compression=_compression_dict(method))
    if bins is not None:
        spectra.set_binning(bins)
    t0 = time.perf_counter()
    spectra.run()
    t_qml = time.perf_counter() - t0
    return {"qml_total": t_qml, "nsims": spectra.params.nsims}


# =========================================================================
# Configurations: fixed nside + fsky, sweep lmax across both methods.
# =========================================================================

NSIDE = 16
FSKY = 0.10
LMAX_VALUES = [8, 16, 24, 32, 48]
DELTA_ELL = 8  # Bandpower binning keeps the Fisher matrix small (~6×6) and
# well-conditioned even at small fsky where individual ℓ bins lack modes.

FIELDS = [
    # (label, spins, labels, physical_labels)
    ("T", [0], ["T"], ["T"]),
    ("QU", [2], ["E", "B"], ["Q", "U"]),
]

METHODS = ["harmonic", "pixel_direct", "auto"]


def _field_spec(field_label):
    """Resolve a CLI ``--field`` label to (spins, labels, physical_labels)."""
    for label, spins, labels, physical in FIELDS:
        if label == field_label:
            return spins, labels, physical
    raise SystemExit(
        f"unknown field label {field_label!r}; choose from {[f[0] for f in FIELDS]}"
    )


def run_one_config(field_label, lmax, method):
    """Run a single (field, lmax, method) configuration and return its timings.

    Each call regenerates inputs locally; intended to be invoked from a
    fresh Python subprocess so every JIT-compiled kernel pays its full
    warmup cost. Sharing inputs across methods inside one process would
    cache the JIT compilations and bias the second/third method.
    """
    spins, labels, physical_labels = _field_spec(field_label)
    cov, mask, sim_maps, fwhmarcmin = generate_test_inputs(
        NSIDE, lmax, spins, physical_labels, lmax_sim=4 * NSIDE, fsky=FSKY
    )
    delta_ell_used = min(DELTA_ELL, max(2, (lmax - 1) // 2))
    with tempfile.TemporaryDirectory(
        dir="sims", prefix=f"bench_{field_label}_lmax{lmax}_{method}_"
    ) as tmpdir:
        config_file = write_temp_config(
            tmpdir,
            NSIDE,
            lmax,
            spins,
            labels,
            physical_labels,
            cov,
            mask,
            sim_maps,
            fwhmarcmin=fwhmarcmin,
        )
        bins = Bins.fromdeltal(2, lmax, delta_ell_used)
        timings, fisher = benchmark_fisher(config_file, method, bins=bins)
        timings.update(
            nside=NSIDE,
            fsky=FSKY,
            lmax=lmax,
            method=method,
            spins=spins,
            field_label=field_label,
            delta_ell=DELTA_ELL,
        )
        try:
            qml_timings = benchmark_spectra(config_file, fisher, method, bins=bins)
            timings.update(qml_timings)
        except Exception as qml_err:
            timings["qml_total"] = None
            timings["qml_error"] = str(qml_err)
    return timings


def _run_worker(field_label, lmax, method):
    """Worker entry point: emit one JSON document on stdout."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        result = run_one_config(field_label, lmax, method)
    except Exception as e:
        result = {"error": str(e)}
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()


def _drive_subprocesses():
    """Driver entry point: spawn one cold subprocess per config, accumulate."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    results = {}
    script = os.path.abspath(__file__)
    for field_label, _spins, _labels, _physical in FIELDS:
        for lmax in LMAX_VALUES:
            header = f"{field_label} lmax={lmax} (nside={NSIDE}, fsky={FSKY})"
            print(f"\n{'=' * 60}\n{header}\n{'=' * 60}")
            for method in METHODS:
                run_label = f"{field_label}_lmax{lmax}_{method}"
                print(f"\n--- Method: {method} (cold subprocess) ---")
                t0 = time.perf_counter()
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        script,
                        "--field",
                        field_label,
                        "--lmax",
                        str(lmax),
                        "--method",
                        method,
                    ],
                    capture_output=True,
                    text=True,
                )
                wall = time.perf_counter() - t0
                if proc.returncode != 0:
                    print(
                        f"  SUBPROCESS FAILED (rc={proc.returncode}, wall={wall:.1f}s):"
                    )
                    print(proc.stderr[-2000:])
                    results[run_label] = {
                        "error": f"subprocess rc={proc.returncode}",
                        "stderr_tail": proc.stderr[-500:],
                    }
                    continue
                # Worker emits one JSON document on stdout — take the last
                # well-formed JSON line so any benign info prints don't break parsing.
                payload = None
                for line in reversed(proc.stdout.strip().splitlines()):
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            payload = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue
                if payload is None:
                    print(f"  NO JSON IN STDOUT (wall={wall:.1f}s)")
                    results[run_label] = {"error": "no JSON in worker stdout"}
                    continue
                payload["wall_subprocess_total"] = wall
                if "error" in payload:
                    print(f"  WORKER ERROR: {payload['error']} (wall={wall:.1f}s)")
                else:
                    print(f"  n_modes/dim = {payload.get('n_modes')}")
                    print(f"  n_pix          = {payload.get('n_pix')}")
                    print(
                        f"  Fisher run:    {payload.get('total', float('nan')):.2f}s "
                        f"(subprocess wall {wall:.1f}s)"
                    )
                    if payload.get("qml_total") is not None:
                        print(
                            f"  QML ({payload.get('nsims')} sims): "
                            f"{payload['qml_total']:.2f}s"
                        )
                results[run_label] = payload

    from _bench_utils import save_results

    out_path = save_results("benchmark_pixel_vs_harmonic", results)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'=' * 78}")
    print(
        f"{'Config':<22} {'Method':<10} {'n_modes':>9} {'n_pix':>7} "
        f"{'Fisher':>10} {'QML/sim':>10}"
    )
    print(f"{'=' * 78}")
    for key, t in results.items():
        if "error" in t:
            print(f"{key:<22} FAILED: {t['error']}")
            continue
        nsims = t.get("nsims") or 0
        qml_per_sim = (
            t["qml_total"] / nsims
            if (t.get("qml_total") is not None and nsims > 0)
            else 0
        )
        print(
            f"{key:<22} {t['method']:<10} {t.get('n_modes', '?'):>9} "
            f"{t['n_pix']:>7} {t.get('total', float('nan')):>9.1f}s "
            f"{qml_per_sim:>9.3f}s"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field", type=str, default=None, help="(worker mode) field label, e.g. T or QU"
    )
    parser.add_argument(
        "--lmax", type=int, default=None, help="(worker mode) analysis lmax"
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=METHODS,
        help="(worker mode) basis method",
    )
    args = parser.parse_args()

    is_worker = any(x is not None for x in (args.field, args.lmax, args.method))
    if is_worker:
        if not all((args.field, args.lmax, args.method)):
            parser.error("worker mode requires --field, --lmax, and --method together")
        _run_worker(args.field, args.lmax, args.method)
    else:
        _drive_subprocesses()


if __name__ == "__main__":
    main()
