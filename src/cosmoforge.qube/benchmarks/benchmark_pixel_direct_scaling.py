"""Benchmark: pixel-direct scaling with nside at fixed fsky and lmax/nside.

Always operates in the regime where pixel-direct (auto) is the optimal
choice: fsky=0.10 with lmax = 2 * nside means n_modes >> n_pix at every
nside. The point is to measure how pixel-direct scales with n_pix and
compare against harmonic and pixel V-based at each scale.

Geometry: polar cap centred on the north pole, fsky ~ 0.10.
Sweep: nside in {16, 32, 64}; lmax_analysis = 2 * nside.
Binning: delta_ell scales so each run has ~6 bandpower bins.
Methods: harmonic, pixel (V-based, no compression), auto (pixel-direct).

Usage: mpirun -n 1 uv run python -u benchmark_pixel_direct_scaling.py
"""

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

    # Noise sigma sized so C = N + S is well-conditioned at every scale.
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
    cfg = {"method": method}
    if method == "pixel":
        cfg["epsilon"] = 1e-30
    return cfg


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
# Configurations
# =========================================================================

FSKY = 0.10
# nside=64 QU needs ~10+ GB RAM (n_pix=9830, ~770 MB per matrix); run on a
# workstation/cluster, not a laptop.
NSIDE_VALUES = [16, 32, 64]
TARGET_NBINS = 6  # Binning chosen so each run has ~6 bandpower bins.

FIELDS = [
    # (label, spins, labels, physical_labels)
    ("T", [0], ["T"], ["T"]),
    ("QU", [2], ["E", "B"], ["Q", "U"]),
]

METHODS = ["harmonic", "pixel", "auto"]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    results = {}

    for nside in NSIDE_VALUES:
        lmax = 2 * nside
        lmax_sim = 4 * nside
        delta_ell = max(2, (lmax - 1) // TARGET_NBINS)

        for field_label, spins, labels, physical_labels in FIELDS:
            print(f"\n{'=' * 60}")
            print(
                f"{field_label} nside={nside}, lmax={lmax}, "
                f"delta_ell={delta_ell}, fsky={FSKY}"
            )
            print(f"{'=' * 60}")

            cov, mask, sim_maps, fwhmarcmin = generate_test_inputs(
                nside,
                lmax,
                spins,
                physical_labels,
                lmax_sim=lmax_sim,
                fsky=FSKY,
            )

            for method in METHODS:
                run_label = f"{field_label}_nside{nside}_{method}"
                print(f"\n--- Method: {method} ---")
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
                        timings, fisher = benchmark_fisher(config_file, method, bins=bins)
                        print(f"  n_modes/dim = {timings['n_modes']}")
                        print(f"  n_pix          = {timings['n_pix']}")
                        print(f"  Fisher run:    {timings['total']:.2f}s")
                        timings.update(
                            {
                                "nside": nside,
                                "fsky": FSKY,
                                "lmax": lmax,
                                "lmax_sim": lmax_sim,
                                "delta_ell": delta_ell,
                                "method": method,
                                "spins": spins,
                                "field_label": field_label,
                            }
                        )
                        try:
                            qml_timings = benchmark_spectra(
                                config_file, fisher, method, bins=bins
                            )
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

    out_path = save_results("benchmark_pixel_direct_scaling", results)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'=' * 90}")
    print(
        f"{'Config':<26} {'Method':<10} {'n_modes':>9} {'n_pix':>7} "
        f"{'Fisher':>10} {'QML/sim':>10}"
    )
    print(f"{'=' * 90}")
    for key, t in results.items():
        if "error" in t:
            print(f"{key:<26} FAILED: {t['error']}")
            continue
        if t.get("qml_total") is None:
            qml_per_sim = float("nan")
        else:
            qml_per_sim = t["qml_total"] / t["nsims"] if t["nsims"] > 0 else 0
        print(
            f"{key:<26} {t['method']:<10} {t.get('n_modes', '?'):>9} "
            f"{t['n_pix']:>7} {t['total']:>9.1f}s "
            f"{qml_per_sim:>9.3f}s"
        )


if __name__ == "__main__":
    main()
