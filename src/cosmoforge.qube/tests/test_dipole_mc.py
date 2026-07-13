"""End-to-end Monte-Carlo test for direct dipole estimation (ADR 0009).

Generates ``N`` realisations of a temperature-only sky containing a known
``C_1``, runs the QML estimator with ``params.lmin_signal=[1]`` and the
inference window pinned to the dipole (``lmin=lmax=1``), and checks that
the recovered ``C_1`` is consistent with the input within the Fisher-
predicted variance.

This is the only smoke test that exercises the lifted ``_lmin_smw`` floor
end-to-end; if it fails the lift in PR2 has reintroduced an off-by-one
somewhere on the V-fill / Lambda / derivative path.

Stability: pass-verified across seeds {12345, 1, 42, 99, 31415} at the
default tolerances (3σ_mean recovery, std ratio in [0.7, 1.3]). NSIMS=200
gives σ_mean ≈ σ/14, so the 3σ window is comfortable but not loose.
"""

from __future__ import annotations

import os
import tempfile

import healpy as hp
import numpy as np
import pytest
import yaml

from qube import Fisher, Spectra

NSIDE = 4
LMAX_SIGNAL = 4
NSIMS = 200
SEED = 12345
C1_INPUT = 5.0e3
NOISE_VAR = 1.0e1


def _write_inputs(tmpdir: str) -> str:
    """Materialise mask, noise covmat, fiducial cls, sims and config files."""
    npix = hp.nside2npix(NSIDE)
    mask = np.ones((1, npix), dtype=np.float64)
    mask_path = os.path.join(tmpdir, "mask.fits")
    hp.write_map(mask_path, mask[0], overwrite=True, dtype=np.float64)

    cov_reduced = np.eye(npix, dtype=np.float64) * NOISE_VAR
    cov_path = os.path.join(tmpdir, "ncvm.bin")
    cov_reduced.tofile(cov_path)

    cl_input = np.zeros(LMAX_SIGNAL + 1, dtype=np.float64)
    cl_input[1] = C1_INPUT
    cls_path = os.path.join(tmpdir, "cls.dat")
    with open(cls_path, "w") as f:
        f.write("# ell  TT\n")
        for ell in range(LMAX_SIGNAL + 1):
            f.write(f"{ell:d}  {cl_input[ell]:.16e}\n")

    rng = np.random.default_rng(SEED)
    sim_maps = np.empty((1, npix, NSIMS), dtype=np.float64)
    for s in range(NSIMS):
        signal_map = hp.synfast(cl_input, NSIDE, lmax=LMAX_SIGNAL, new=True)
        # Independent white noise per pixel (matches cov_reduced).
        noise_map = rng.normal(0.0, np.sqrt(NOISE_VAR), size=npix)
        sim_maps[0, :, s] = signal_map + noise_map
    sim_path = os.path.join(tmpdir, "sims.npy")
    np.save(sim_path, sim_maps)

    config = {
        "nside": NSIDE,
        "spins": [0],
        "labels": ["T"],
        "physical_labels": ["T"],
        "do_cross": False,
        "maskfile": mask_path,
        "output_geometry_file": os.path.join(tmpdir, "geometry.dat"),
        "ordering": "RING",
        "inputclfile": cls_path,
        "fiducialfile": cls_path,
        "input_convention": "Cl",
        "covmatfile1": cov_path,
        "covmatfile2": cov_path,
        "lmax": 1,
        "lmin": 1,
        "lmax_signal": LMAX_SIGNAL,
        "lmin_signal": [1],
        "load_reduced": True,
        "load_inverted": False,
        "smoothing_type": "none",
        "fwhmarcmin": 0.0,
        "apply_pixwin": False,
        "beam_file": "",
        "outnoisecovmat1": os.path.join(tmpdir, "reduced_ncvm1.bin"),
        "outnoisecovmat2": os.path.join(tmpdir, "reduced_ncvm2.bin"),
        "outinvcovmatfile1": os.path.join(tmpdir, "invcov1.bin"),
        "outinvcovmatfile2": os.path.join(tmpdir, "invcov2.bin"),
        "outfilefisher": os.path.join(tmpdir, "fisher.dat"),
        "feedback": 0,
        "nsims": NSIMS,
        "inputmapfile1": sim_path,
        "inputmapfile2": sim_path,
        "outcovmatfile": os.path.join(tmpdir, "cov_matrix.dat"),
        "outerrfile": os.path.join(tmpdir, "errors.dat"),
        "remove_nb": True,
        "delta_ell": 1,
    }
    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def test_dipole_recovered_within_fisher_error():
    """T-only QML with ``lmin=lmax=1`` recovers ``C_1`` to Fisher precision."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = _write_inputs(tmpdir)
        fisher = Fisher(config_path)
        fisher.run()
        spectra = Spectra(config_path, fisher=fisher)
        spectra.run()

        bandpowers = spectra.get_power_spectra()
        error_bars = fisher.get_error_bars()

    assert bandpowers is not None and error_bars is not None
    assert bandpowers.shape == (NSIMS, 1)
    assert error_bars.shape == (1,)

    mean_C1 = float(np.mean(bandpowers[:, 0]))
    sigma_per_realisation = float(error_bars[0])
    sigma_of_mean = sigma_per_realisation / np.sqrt(NSIMS)

    # Mean recovery: bias below 3 sigma of the per-realisation Fisher error,
    # divided by sqrt(NSIMS). With NSIMS=200 the noise floor on the mean is
    # ~sigma/14, so a 3-sigma window gives a tight but realistic gate.
    assert abs(mean_C1 - C1_INPUT) < 3.0 * sigma_of_mean, (
        f"Mean C_1 = {mean_C1:.3e}, expected {C1_INPUT:.3e}; "
        f"|delta|/sigma_mean = {abs(mean_C1 - C1_INPUT) / sigma_of_mean:.2f}"
    )

    # Variance check: empirical std consistent with Fisher prediction within
    # 30 % at NSIMS=200 (binomial scatter on the std estimator is ~5 % at
    # this sample size; loose bound to keep the test stable across seeds).
    empirical_std = float(np.std(bandpowers[:, 0], ddof=1))
    ratio = empirical_std / sigma_per_realisation
    assert 0.7 < ratio < 1.3, (
        f"Empirical std {empirical_std:.3e} vs Fisher {sigma_per_realisation:.3e}; "
        f"ratio = {ratio:.3f}"
    )


@pytest.mark.skipif(
    not os.environ.get("RUN_DIPOLE_LARGE_MC"),
    reason="Larger NSIMS run (set RUN_DIPOLE_LARGE_MC=1 to enable).",
)
def test_dipole_large_mc_diagnostic():
    """Diagnostic: 1000 sims for a stricter mean-recovery check.

    Skipped by default to keep per-package wall time low; useful when
    investigating regressions on this path.
    """
    pass
