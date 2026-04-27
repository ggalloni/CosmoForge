"""
Benchmark 3: MPI scaling for Fisher + QML computation

Runs QU nside=32 lmax=64 with varying MPI ranks.
At this size, Fisher traces (~44s) and QML (~40s/1000 sims) dominate,
so MPI parallelism should show clear scaling.

Usage:
  mpirun -n 1  uv run python -u benchmark_mpi.py
  mpirun -n 2  uv run python -u benchmark_mpi.py
  mpirun -n 4  uv run python -u benchmark_mpi.py
  mpirun -n 8  uv run python -u benchmark_mpi.py
  mpirun -n 16 uv run python -u benchmark_mpi.py

Or run the wrapper that does all ranks automatically:
  bash run_mpi_benchmark.sh
"""

import json
import os
import tempfile
import time

import healpy as hp
import numpy as np
import yaml
from mpi4py import MPI

from qube import Fisher, Spectra


def _gaussian_fwhm_for_lmax(lmax, beam_at_lmax=0.5):
    """Return FWHM in arcmin for a Gaussian beam with b(lmax) = beam_at_lmax."""
    sigma = np.sqrt(-2 * np.log(beam_at_lmax) / (lmax * (lmax + 1)))
    return float(np.degrees(sigma * np.sqrt(8 * np.log(2))) * 60)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Configuration: QU at nside=32, lmax=64
    nside = 32
    lmax = 64
    spins = [2]
    labels = ["E", "B"]
    physical_labels = ["Q", "U"]
    npix = 12 * nside**2
    n_fields = len(physical_labels)
    nsims = 10000
    sigma = 1.5

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"MPI Scaling Benchmark: QU nside={nside} lmax={lmax}")
        print(f"MPI ranks: {size}, nsims: {nsims}")
        print(f"{'=' * 60}")

    # Create temp directory (rank 0 only, broadcast path)
    tmpdir = None
    if rank == 0:
        tmpdir = tempfile.mkdtemp(dir="sims", prefix=f"bench_mpi_{size}_")
    tmpdir = comm.bcast(tmpdir, root=0)

    if rank == 0:
        # Generate inputs
        cov = np.zeros((n_fields * npix, n_fields * npix))
        for i in range(n_fields * npix):
            cov[i, i] = sigma**2

        gal_cut = np.radians(10)
        galactic_pixels = hp.query_strip(nside, np.pi / 2 - gal_cut, np.pi / 2 + gal_cut)
        mask = np.ones((n_fields, npix))
        mask[:, galactic_pixels] = 0.0

        # Generate sims
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

        # Write files
        cov.tofile(os.path.join(tmpdir, "ncvm.bin"))
        hp.write_map(os.path.join(tmpdir, "mask.fits"), mask, overwrite=True)
        np.save(os.path.join(tmpdir, "sims.npy"), sim_maps)

        config = {
            "nside": nside,
            "spins": spins,
            "labels": labels,
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
            "calibration": 1.0,
            "smoothing_type": "gaussian",
            "fwhmarcmin": fwhm_arcmin,
            "apply_pixwin": False,
            "beam_file": "",
            "outnoisecovmat1": os.path.join(tmpdir, "reduced_ncvm1.bin"),
            "outnoisecovmat2": os.path.join(tmpdir, "reduced_ncvm2.bin"),
            "feedback": 4,
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
    else:
        config_file = None

    config_file = comm.bcast(config_file, root=0)
    comm.Barrier()

    # Time Fisher
    compression_config = {"method": "harmonic"}
    fisher = Fisher(config_file, compression=compression_config, cache_derivatives=False)

    comm.Barrier()
    t0 = time.perf_counter()
    fisher.run()
    comm.Barrier()
    t_fisher = time.perf_counter() - t0

    # Time Spectra (QML)
    spectra = Spectra(config_file, fisher=fisher, compression=compression_config)

    comm.Barrier()
    t0 = time.perf_counter()
    spectra.run()
    comm.Barrier()
    t_spectra = time.perf_counter() - t0

    if rank == 0:
        print(f"\n--- Results for {size} MPI ranks ---")
        print(f"  Fisher:  {t_fisher:.2f}s")
        print(f"  Spectra: {t_spectra:.2f}s ({nsims} sims)")
        print(f"  QML/sim: {t_spectra / nsims:.4f}s")

        # Save Fisher matrix and spectra for consistency check
        fisher_file = f"benchmark_mpi_fisher_{size}.npy"
        spectra_file = f"benchmark_mpi_spectra_{size}.npy"
        np.save(fisher_file, fisher.fisher)
        power_spectra = spectra.get_power_spectra()
        np.save(spectra_file, power_spectra)

        # Compare against 1-rank reference if available
        ref_fisher_file = "benchmark_mpi_fisher_1.npy"
        ref_spectra_file = "benchmark_mpi_spectra_1.npy"
        if size > 1 and os.path.exists(ref_fisher_file):
            ref_fisher = np.load(ref_fisher_file)
            ref_spectra = np.load(ref_spectra_file)
            fisher_diff = np.max(np.abs(fisher.fisher - ref_fisher))
            fisher_rel = fisher_diff / np.max(np.abs(ref_fisher))
            spectra_diff = np.max(np.abs(power_spectra - ref_spectra))
            spectra_rel = spectra_diff / np.max(np.abs(ref_spectra))
            print(
                f"  Fisher vs 1-rank: max abs diff = {fisher_diff:.2e}, "
                f"max rel diff = {fisher_rel:.2e}"
            )
            print(
                f"  Spectra vs 1-rank: max abs diff = {spectra_diff:.2e}, "
                f"max rel diff = {spectra_rel:.2e}"
            )
            if fisher_rel > 1e-10:
                print("  WARNING: Fisher differs beyond machine precision!")
            if spectra_rel > 1e-10:
                print("  WARNING: Spectra differs beyond machine precision!")

        # Append to results file
        results_file = "benchmark_mpi_results.json"
        if os.path.exists(results_file):
            with open(results_file) as f:
                results = json.load(f)
        else:
            results = {}

        results[str(size)] = {
            "n_ranks": size,
            "fisher_time": t_fisher,
            "spectra_time": t_spectra,
            "qml_per_sim": t_spectra / nsims,
            "nsims": nsims,
            "nside": nside,
            "lmax": lmax,
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results appended to {results_file}")

        # Cleanup
        import shutil

        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
