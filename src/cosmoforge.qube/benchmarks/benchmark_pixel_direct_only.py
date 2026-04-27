"""Benchmark: pixel-direct mode alone, sweeping nside at configurable fsky.

Restricted to ``method="auto"``. At small fsky and lmax = 2 * nside the auto
factory selects the pixel-direct path on every cell, so this measures
pixel-direct's pure scaling without the harmonic/pixel V-based comparison.
Use this when you already know (e.g. from benchmark_pixel_vs_harmonic.py)
that auto wins in this regime and you want timing for the auto path alone.

Sweep: nside in NSIDE_VALUES (env-overridable), fsky configurable via CLI.
Binning: delta_ell scales so each run has ~TARGET_NBINS bandpower bins.

Memory & time hints (cluster, threaded BLAS, fsky=0.01):
  nside= 64:   n_pix(QU) ~  983,  total cell <  10 s
  nside=128:   n_pix(QU) ~ 3932,  total cell ~ minutes
  nside=256:   n_pix(QU) ~15728,  total cell ~ 0.5–1 hour
  nside=512:   n_pix(QU) ~62914,  total cell ~ several hours, 30+ GB RAM

Inputs are written in *reduced* form (n_active × n_active) and consumed via
``load_reduced=True``. This avoids the full 12*nside² × 12*nside² noise
materialisation that would OOM at nside ≥ 128 (the full HEALPix-resolution
matrix is 300+ GB for T at nside=128, even though the active block is tiny).

Usage:
  uv run python -u benchmark_pixel_direct_only.py
  uv run python -u benchmark_pixel_direct_only.py --fsky 0.01 --nsides 16,32,64,128

  # Two invocations at fsky=0.01 without clobbering output:
  uv run python -u benchmark_pixel_direct_only.py --fsky 0.01 \\
      --nsides 16,32,64,128,256 --suffix both_fields
  uv run python -u benchmark_pixel_direct_only.py --fsky 0.01 \\
      --nsides 512 --fields T --suffix T_only_extreme
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import healpy as hp
import numpy as np
import yaml
from _bench_utils import save_results

from cosmocore import Bins
from qube import Fisher, Spectra


def _gaussian_fwhm_for_lmax(lmax, beam_at_lmax=0.5):
    """Return FWHM in arcmin for a Gaussian beam with b(lmax) = beam_at_lmax."""
    sigma = np.sqrt(-2 * np.log(beam_at_lmax) / (lmax * (lmax + 1)))
    return float(np.degrees(sigma * np.sqrt(8 * np.log(2))) * 60)


def _polar_cap_mask(nside, n_fields, fsky):
    radius = np.arccos(1.0 - 2.0 * fsky)
    pole_vec = hp.ang2vec(0.0, 0.0)
    cap_pixels = hp.query_disc(nside, pole_vec, radius)
    mask = np.zeros((n_fields, hp.nside2npix(nside)))
    mask[:, cap_pixels] = 1.0
    return mask


def generate_test_inputs(nside, lmax, spins, physical_labels, lmax_sim, fsky):
    """Generate inputs for a benchmark cell.

    Returns the *reduced* noise covariance (n_active × n_active) instead of
    the full HEALPix-resolution one. At small fsky the full matrix is mostly
    zero anyway and would OOM at nside ≥ 128 (e.g. T at nside=128 is a
    300+ GB allocation). The reduced matrix is paired with the
    ``load_reduced`` config flag so Core reads it via
    ``read_covmat_reduced`` and never materialises the full version.
    """
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

    fwhm_arcmin = _gaussian_fwhm_for_lmax(lmax_sim)
    fwhm_rad = np.radians(fwhm_arcmin / 60.0)
    beam = hp.gauss_beam(fwhm_rad, lmax=lmax_sim)

    cl_for_sigma = cl_ee if 2 in spins else cl_tt
    ell_arr = np.arange(lmax_sim + 1)
    sig_var_per_pix = np.sum((2 * ell_arr + 1) / (4 * np.pi) * cl_for_sigma * beam**2)
    sigma = float(np.sqrt(sig_var_per_pix))

    # Reduced noise covariance: only the n_active × n_active block.
    # Layout matches Core's expectation. n_fields equals the number of
    # *physical* maps (e.g. 2 for QU = Q + U), so summing the active count
    # per physical map already accounts for the spin-2 Q+U doubling — no
    # extra factor of 2 needed.
    n_active_per_phys = [int(np.sum(mask[i] > 0)) for i in range(n_fields)]
    n_pix_active = sum(n_active_per_phys)
    cov_reduced = np.eye(n_pix_active, dtype=np.float64) * sigma**2

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

    return cov_reduced, mask, sim_maps, fwhm_arcmin


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
        # Read pre-reduced (n_active × n_active) covmat directly from disk;
        # avoids materialising the full 12*nside^2 noise matrix in memory.
        "load_reduced": True,
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


def benchmark_fisher(config_file, bins=None):
    fisher = Fisher(config_file, compression={"method": "auto"})
    if bins is not None:
        fisher.set_binning(bins)
    t0 = time.perf_counter()
    fisher.run()
    t_total = time.perf_counter() - t0
    n_kept = (
        fisher.basis_manager.n_kept
        if hasattr(fisher, "basis_manager") and fisher.basis_manager
        else None
    )
    return {
        "total": t_total,
        "n_modes": n_kept,
        "n_pix": sum(fisher.collection.n_active),
    }, fisher


def benchmark_spectra(config_file, fisher, bins=None):
    spectra = Spectra(config_file, fisher=fisher, compression={"method": "auto"})
    if bins is not None:
        spectra.set_binning(bins)
    t0 = time.perf_counter()
    spectra.run()
    t_qml = time.perf_counter() - t0
    return {"qml_total": t_qml, "nsims": spectra.params.nsims}


# =========================================================================
# Defaults (overridable via CLI)
# =========================================================================

DEFAULT_FSKY = 0.10
DEFAULT_NSIDES = [16, 32, 64]
DEFAULT_FIELDS = ["T", "QU"]
TARGET_NBINS = 6

FIELD_DEFS = {
    "T": ([0], ["T"], ["T"]),
    "QU": ([2], ["E", "B"], ["Q", "U"]),
}


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
    p.add_argument(
        "--target-nbins",
        type=int,
        default=TARGET_NBINS,
        help="Target number of bandpower bins per cell.",
    )
    p.add_argument(
        "--suffix",
        type=str,
        default="",
        help=(
            "Optional suffix appended to the output filename, useful when "
            "running multiple invocations at the same fsky (e.g. one for "
            "T+QU up to a moderate nside and one for T-only at extreme nside) "
            "without clobbering each other's results."
        ),
    )
    return p.parse_args()


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
        f"Pixel-direct benchmark: fsky={fsky}, "
        f"nsides={nside_values}, fields={field_labels}, target_nbins={target_nbins}"
    )

    results = {}

    for nside in nside_values:
        lmax = 2 * nside
        lmax_sim = 4 * nside
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
                nside,
                lmax,
                spins,
                physical_labels,
                lmax_sim=lmax_sim,
                fsky=fsky,
            )

            run_label = f"{field_label}_nside{nside}_auto"
            with tempfile.TemporaryDirectory(
                dir="sims", prefix=f"bench_{run_label}_"
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
                    timings, fisher = benchmark_fisher(config_file, bins=bins)
                    print(f"  n_modes/n_kept = {timings['n_modes']}")
                    print(f"  n_pix          = {timings['n_pix']}")
                    print(f"  Fisher run:    {timings['total']:.2f}s")
                    timings.update(
                        {
                            "nside": nside,
                            "fsky": fsky,
                            "lmax": lmax,
                            "lmax_sim": lmax_sim,
                            "delta_ell": delta_ell,
                            "method": "auto",
                            "spins": spins,
                            "field_label": field_label,
                        }
                    )
                    try:
                        qml_timings = benchmark_spectra(config_file, fisher, bins=bins)
                        print(
                            f"  QML ({qml_timings['nsims']} sims): "
                            f"{qml_timings['qml_total']:.2f}s"
                        )
                        timings.update(qml_timings)
                    except Exception as qml_err:
                        print(f"  QML SKIPPED: {qml_err}")
                        timings["qml_total"] = None
                        timings["qml_error"] = str(qml_err)
                    results[run_label] = timings
                except Exception as e:
                    print(f"  FAILED: {e}")
                    results[run_label] = {"error": str(e)}

    out_name = f"benchmark_pixel_direct_only_fsky{fsky:.3f}".replace(".", "p")
    if args.suffix:
        out_name += f"_{args.suffix}"
    out_path = save_results(out_name, results)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'=' * 80}")
    print(f"{'Config':<26} {'n_modes':>9} {'n_pix':>7} {'Fisher':>10} {'QML/sim':>10}")
    print(f"{'=' * 80}")
    for key, t in results.items():
        if "error" in t:
            print(f"{key:<26} FAILED: {t['error']}")
            continue
        if t.get("qml_total") is None:
            qml_per_sim = float("nan")
        else:
            qml_per_sim = t["qml_total"] / t["nsims"] if t["nsims"] > 0 else 0
        print(
            f"{key:<26} {t.get('n_modes', '?'):>9} "
            f"{t['n_pix']:>7} {t['total']:>9.1f}s "
            f"{qml_per_sim:>9.3f}s"
        )


if __name__ == "__main__":
    main()
