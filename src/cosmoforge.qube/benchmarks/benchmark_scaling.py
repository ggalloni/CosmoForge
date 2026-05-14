"""
Benchmark 1: Scaling with nside/lmax
Benchmark 2: Harmonic vs pixel basis comparison

Measures wall-clock time for Fisher pipeline stages across configurations.
Single MPI process, default BLAS threading.

Usage: uv run python benchmark_scaling.py
"""

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


def generate_test_inputs(
    nside, lmax, spins, physical_labels, lmax_sim, mask_gal_cut_deg=10
):
    """Generate minimal inputs for a benchmark run.

    Parameters
    ----------
    lmax_sim : int
        Maximum multipole for simulation generation.  Should be
        4 * nside_max so that every configuration sees a fully
        populated sky, mimicking real data.
    """
    npix = 12 * nside**2

    # Noise covariance (diagonal). Scale with nside to keep C = S + N
    # well-conditioned at all resolutions.
    n_fields = len(physical_labels)
    cov = np.zeros((n_fields * npix, n_fields * npix))
    sigma = 1.5
    for i in range(n_fields * npix):
        cov[i, i] = sigma**2

    # Mask
    gal_cut = np.radians(mask_gal_cut_deg)
    if mask_gal_cut_deg > 0:
        galactic_pixels = hp.query_strip(nside, np.pi / 2 - gal_cut, np.pi / 2 + gal_cut)
        mask = np.ones((n_fields, npix))
        mask[:, galactic_pixels] = 0.0
    else:
        mask = np.ones((n_fields, npix))

    # Read Cls up to lmax_sim so all signal multipoles are populated
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

    # Gaussian beam: FWHM chosen so b(lmax) ~ 0.5, keeping Fisher well-conditioned
    fwhm_arcmin = _gaussian_fwhm_for_lmax(lmax)
    fwhm_rad = np.radians(fwhm_arcmin / 60.0)
    beam = hp.gauss_beam(fwhm_rad, lmax=lmax_sim)

    nsims = 1000
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

    return cov, mask, sim_maps


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
    """Write temporary config and data files, return config path."""
    nsims = sim_maps.shape[2]

    # Write files
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
    """Run Fisher and return timing breakdown."""
    compression = {"method": method}

    fisher = Fisher(config_file, compression=compression)

    # Time the full run() — it handles all setup + computation internally
    t0 = time.perf_counter()
    fisher.run()
    t_total = time.perf_counter() - t0

    return {
        "total": t_total,
        "n_modes": fisher.basis_manager.dim
        if hasattr(fisher, "basis_manager") and fisher.basis_manager
        else None,
        "n_pix": sum(fisher.collection.n_active),
    }, fisher


def benchmark_spectra(config_file, fisher, method):
    """Run Spectra with pre-computed Fisher, return QML timing."""
    compression = {"method": method}

    spectra = Spectra(config_file, fisher=fisher, compression=compression)

    t0 = time.perf_counter()
    spectra.run()
    t_qml = time.perf_counter() - t0

    return {"qml_total": t_qml, "nsims": spectra.params.nsims}


# =========================================================================
# Benchmark configurations
# =========================================================================

CONFIGS = [
    # (label, nside, lmax, spins, labels, physical_labels)
    ("T_ns8", 8, 16, [0], ["T"], ["T"]),
    ("T_ns16", 16, 32, [0], ["T"], ["T"]),
    ("T_ns32", 32, 64, [0], ["T"], ["T"]),
    ("QU_ns8", 8, 16, [2], ["E", "B"], ["Q", "U"]),
    ("QU_ns16", 16, 32, [2], ["E", "B"], ["Q", "U"]),
    ("QU_ns32", 32, 64, [2], ["E", "B"], ["Q", "U"]),
]

METHODS = ["harmonic"]  # Add "pixel" for Benchmark 2 comparison


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    results = {}

    # Use 4 * max_nside so all configs see a fully populated sky
    max_nside = max(cfg[1] for cfg in CONFIGS)
    lmax_sim = 4 * max_nside

    for label, nside, lmax, spins, labels, physical_labels in CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"Configuration: {label} (nside={nside}, lmax={lmax}, {physical_labels})")
        print(f"{'=' * 60}")

        fwhmarcmin = _gaussian_fwhm_for_lmax(lmax)
        cov, mask, sim_maps = generate_test_inputs(
            nside, lmax, spins, physical_labels, lmax_sim=lmax_sim
        )

        for method in METHODS:
            run_label = f"{label}_{method}"
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

                try:
                    timings, fisher = benchmark_fisher(config_file, method)

                    print(f"  n_modes = {timings['n_modes']}")
                    print(f"  n_pix   = {timings['n_pix']}")
                    print(f"  Total Fisher:   {timings['total']:.2f}s")

                    qml_timings = benchmark_spectra(config_file, fisher, method)
                    print(
                        f"  QML ({qml_timings['nsims']} sims): "
                        f"{qml_timings['qml_total']:.2f}s"
                    )

                    timings.update(qml_timings)
                    timings["nside"] = nside
                    timings["lmax"] = lmax
                    timings["method"] = method
                    timings["spins"] = spins
                    results[run_label] = timings

                except Exception as e:
                    print(f"  FAILED: {e}")
                    results[run_label] = {"error": str(e)}

    # Save results
    from _bench_utils import save_results

    out_path = save_results("benchmark_scaling", results)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'=' * 70}")
    print(
        f"{'Config':<15} {'Method':<10} {'n_modes':>8} {'n_pix':>8} "
        f"{'Fisher':>10} {'QML/sim':>10}"
    )
    print(f"{'=' * 70}")
    for key, t in results.items():
        if "error" in t:
            print(f"{key:<15} FAILED: {t['error']}")
            continue
        qml_per_sim = t["qml_total"] / t["nsims"] if t["nsims"] > 0 else 0
        print(
            f"{key:<15} {t['method']:<10} {t.get('n_modes', '?'):>8} "
            f"{t['n_pix']:>8} {t['total']:>9.1f}s "
            f"{qml_per_sim:>9.3f}s"
        )


if __name__ == "__main__":
    main()
