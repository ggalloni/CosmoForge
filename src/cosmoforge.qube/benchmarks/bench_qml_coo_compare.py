"""
Compare QML timings: sparse-COO branch vs dense fallback.

Runs the full QU pipeline once (Fisher + Spectra), times the default sparse
path, then nulls the COO cache and re-times the dense path. Same maps, same
weighted data, same noise covariance — only the QML inner loop differs.

Defaults match the QU MPI benchmark (nside=32, lmax=64, nsims=10000). Override
via CLI for a faster local check, e.g.:

  uv run python bench_qml_coo_compare.py --nside 16 --lmax 32 --nsims 500
"""

import argparse
import os
import tempfile
import time

import healpy as hp
import numpy as np
import yaml

from qube import Spectra


def _gaussian_fwhm_for_lmax(lmax, beam_at_lmax=0.5):
    sigma = np.sqrt(-2 * np.log(beam_at_lmax) / (lmax * (lmax + 1)))
    return float(np.degrees(sigma * np.sqrt(8 * np.log(2))) * 60)


def _build_fixture(tmpdir, nside, lmax, nsims):
    """Generate maps, mask, noise cov, beam, config — same shape as benchmark_mpi.py."""
    spins = [2]
    physical_labels = ["Q", "U"]
    n_fields = len(physical_labels)
    npix = 12 * nside**2
    sigma = 1.5

    # Diagonal noise
    cov = np.zeros((n_fields * npix, n_fields * npix))
    np.fill_diagonal(cov, sigma**2)

    # Galactic-cut mask
    galactic_pixels = hp.query_strip(
        nside, np.pi / 2 - np.radians(10), np.pi / 2 + np.radians(10)
    )
    mask = np.ones((n_fields, npix))
    mask[:, galactic_pixels] = 0.0

    # Theory Cls
    raw_cls = np.loadtxt("sims/tau0.06_dls_r_likelihood_r0.0.txt")
    ells_file = raw_cls[:, 0].astype(int)
    dl2cl = np.ones(len(ells_file))
    dl2cl[ells_file > 0] = (
        2 * np.pi / (ells_file[ells_file > 0] * (ells_file[ells_file > 0] + 1))
    )
    lmax_sim = 4 * nside
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
    field_index_map = {"Q": 1, "U": 2}

    sim_maps = np.empty((n_fields, npix, nsims), dtype=np.float64)
    for i in range(nsims):
        np.random.seed(42 + i)
        alms = hp.synalm([cl_tt, cl_ee, cl_bb, cl_te], lmax=lmax_sim, new=True)
        hp.almxfl(alms[1], beam, inplace=True)
        hp.almxfl(alms[2], beam, inplace=True)
        sim_tqu = hp.alm2map(alms, nside=nside, lmax=lmax_sim)
        for j, label in enumerate(physical_labels):
            sim_maps[j, :, i] = sim_tqu[field_index_map[label]]

    cov.tofile(os.path.join(tmpdir, "ncvm.bin"))
    hp.write_map(os.path.join(tmpdir, "mask.fits"), mask, overwrite=True)
    np.save(os.path.join(tmpdir, "sims.npy"), sim_maps)

    config = {
        "nside": nside,
        "spins": spins,
        "labels": ["E", "B"],
        "physical_labels": physical_labels,
        "do_cross": False,
        "maskfile": os.path.join(tmpdir, "mask.fits"),
        "output_geometry_file": os.path.join(tmpdir, "geometry.dat"),
        "ordering": "RING",
        "inputclfile": os.path.abspath("sims/tau0.06_dls_r_likelihood_r0.0.txt"),
        "input_convention": "Dl",
        "covmatfile1": os.path.join(tmpdir, "ncvm.bin"),
        "covmatfile2": os.path.join(tmpdir, "ncvm.bin"),
        "lmax": lmax,
        "load_inverted": True,
        "smoothing_type": "gaussian",
        "fwhmarcmin": fwhm_arcmin,
        "apply_pixwin": False,
        "beam_file": "",
        "outnoisecovmat1": os.path.join(tmpdir, "reduced_ncvm1.bin"),
        "outnoisecovmat2": os.path.join(tmpdir, "reduced_ncvm2.bin"),
        "feedback": 1,
        "outinvcovmatfile1": os.path.join(tmpdir, "invcov1.bin"),
        "outinvcovmatfile2": os.path.join(tmpdir, "invcov2.bin"),
        "outfilefisher": os.path.join(tmpdir, "fisher.dat"),
        "nsims": nsims,
        "inputmapfile1": os.path.join(tmpdir, "sims.npy"),
        "inputmapfile2": os.path.join(tmpdir, "sims.npy"),
        "outcovmatfile": os.path.join(tmpdir, "cov_matrix.dat"),
        "outerrfile": os.path.join(tmpdir, "errors.dat"),
        "remove_nb": False,
        "nspectra": 3,
    }
    config_file = os.path.join(tmpdir, "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return config_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nside", type=int, default=32)
    p.add_argument("--lmax", type=int, default=64)
    p.add_argument("--nsims", type=int, default=10000)
    args = p.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print(f"\n{'=' * 60}")
    print(
        f"QML COO comparison: QU nside={args.nside} lmax={args.lmax} nsims={args.nsims}"
    )
    print(f"{'=' * 60}")

    tmpdir = tempfile.mkdtemp(dir="sims", prefix="bench_coo_")
    try:
        config_file = _build_fixture(tmpdir, args.nside, args.lmax, args.nsims)

        # --- Run 1: default (sparse-COO) ---
        qml_sparse = Spectra(config_file, compression={"method": "harmonic"})
        t0 = time.perf_counter()
        qml_sparse.run()
        t_total_sparse = time.perf_counter() - t0
        ps_sparse = qml_sparse.get_power_spectra()

        # Time only the QML step on the same fully-set-up Spectra: invalidate
        # results but reuse setup. We just rerun compute_qml_spectra().
        t0 = time.perf_counter()
        qml_sparse.compute_qml_spectra()
        t_qml_sparse = time.perf_counter() - t0
        path_sparse = qml_sparse._qml_path_used

        # --- Run 2: dense fallback ---
        qml_sparse.fisher_instance._cached_sparse_coo_data = None
        t0 = time.perf_counter()
        qml_sparse.compute_qml_spectra()
        t_qml_dense = time.perf_counter() - t0
        path_dense = qml_sparse._qml_path_used
        ps_dense = qml_sparse.get_power_spectra()

        rel_diff = np.max(np.abs(ps_sparse - ps_dense)) / np.max(np.abs(ps_dense))

        print("\n--- Results ---")
        print(f"  Full-pipeline sparse run:      {t_total_sparse:.2f} s")
        print(f"  QML only — {path_sparse:6} path: {t_qml_sparse:.2f} s")
        print(f"  QML only — {path_dense:6} path: {t_qml_dense:.2f} s")
        if t_qml_sparse > 0:
            print(f"  Speedup (dense / sparse):      {t_qml_dense / t_qml_sparse:.1f}x")
        print(f"  Max relative ps difference:    {rel_diff:.2e}")
    finally:
        import shutil

        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
