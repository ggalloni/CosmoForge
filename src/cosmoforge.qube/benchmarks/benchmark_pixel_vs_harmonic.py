"""
Benchmark: Harmonic vs pixel basis at small fsky.

The pixel basis becomes competitive when n_kept < n_modes. At fixed
nside, n_pix shrinks with the mask while n_modes ~ lmax^2 grows. The
crossover therefore sits at small fsky and increasing lmax — the regime
this benchmark targets.

Geometry: polar cap centred on the north pole, fsky ~ 0.1.
Sweep: lmax in {8, 16, 24, 32, 48}, T-only and QU.
Methods:
  - harmonic: V-based, full SMW pipeline
  - pixel:    V-based eigenvalue truncation (default epsilon=1e-6, noise_weighted)
  - auto:     factory picks harmonic when n_pix > n_modes, otherwise
              pixel in direct mode (no V, full pixel-space ops)

Usage: mpirun -n 1 uv run python -u benchmark_pixel_vs_harmonic.py
"""

import json
import os
import tempfile
import time

import healpy as hp
import numpy as np
import yaml

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

    sigma = 1.5
    cov = np.eye(n_fields * npix) * sigma**2
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

    fwhm_arcmin = _gaussian_fwhm_for_lmax(lmax)
    fwhm_rad = np.radians(fwhm_arcmin / 60.0)
    beam = hp.gauss_beam(fwhm_rad, lmax=lmax_sim)

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


def benchmark_fisher(config_file, method):
    fisher = Fisher(config_file, compression={"method": method})
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


def benchmark_spectra(config_file, fisher, method):
    spectra = Spectra(config_file, fisher=fisher, compression={"method": method})
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

FIELDS = [
    # (label, spins, labels, physical_labels)
    ("T", [0], ["T"], ["T"]),
    ("QU", [2], ["E", "B"], ["Q", "U"]),
]

METHODS = ["harmonic", "pixel", "auto"]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    results = {}
    lmax_sim = 4 * NSIDE

    for field_label, spins, labels, physical_labels in FIELDS:
        for lmax in LMAX_VALUES:
            print(f"\n{'=' * 60}")
            print(
                f"{field_label} lmax={lmax} (nside={NSIDE}, fsky={FSKY}, "
                f"{physical_labels})"
            )
            print(f"{'=' * 60}")

            cov, mask, sim_maps, fwhmarcmin = generate_test_inputs(
                NSIDE,
                lmax,
                spins,
                physical_labels,
                lmax_sim=lmax_sim,
                fsky=FSKY,
            )

            for method in METHODS:
                run_label = f"{field_label}_lmax{lmax}_{method}"
                print(f"\n--- Method: {method} ---")
                with tempfile.TemporaryDirectory(
                    dir="sims", prefix=f"bench_{run_label}_"
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
                    try:
                        timings, fisher = benchmark_fisher(config_file, method)
                        print(f"  n_modes/n_kept = {timings['n_modes']}")
                        print(f"  n_pix          = {timings['n_pix']}")
                        print(f"  Fisher run:    {timings['total']:.2f}s")
                        qml_timings = benchmark_spectra(config_file, fisher, method)
                        print(
                            f"  QML ({qml_timings['nsims']} sims): "
                            f"{qml_timings['qml_total']:.2f}s"
                        )
                        timings.update(qml_timings)
                        timings.update(
                            {
                                "nside": NSIDE,
                                "fsky": FSKY,
                                "lmax": lmax,
                                "method": method,
                                "spins": spins,
                                "field_label": field_label,
                            }
                        )
                        results[run_label] = timings
                    except Exception as e:
                        print(f"  FAILED: {e}")
                        results[run_label] = {"error": str(e)}

    output_file = "benchmark_pixel_vs_harmonic_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

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
        qml_per_sim = t["qml_total"] / t["nsims"] if t["nsims"] > 0 else 0
        print(
            f"{key:<22} {t['method']:<10} {t.get('n_modes', '?'):>9} "
            f"{t['n_pix']:>7} {t['total']:>9.1f}s "
            f"{qml_per_sim:>9.3f}s"
        )


if __name__ == "__main__":
    main()
