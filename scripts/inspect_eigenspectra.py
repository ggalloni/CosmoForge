"""
Inspect per-field eigenspectra for the nside=4 TQU test configuration.

Uses the same data as the qube TQU Fisher test (nside=4, lmax=8,
spins=[0,2]) to produce:
  1. Per-field eigenvalue spectra (with E/B overlay for the spin-2 field)
  2. Per-field basis comparison (harmonic vs noise_weighted)

Usage:
    uv run python scripts/inspect_eigenspectra.py
"""

import os
import tempfile

import yaml

from cosmocore.compression import PixelProjectedCompression


def _resolve_config(config_path, qube_root):
    """Resolve relative paths in a config file (same logic as qube conftest)."""
    with open(os.path.join(qube_root, config_path)) as f:
        config = yaml.safe_load(f)

    package_prefix = "src/cosmoforge.qube/"
    for key, value in config.items():
        if isinstance(value, str):
            clean = value[3:] if value.startswith("../") else value
            if clean.startswith(("tests/", "inputs/", "scripts/")):
                config[key] = package_prefix + clean

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
    )
    yaml.dump(config, tmp, default_flow_style=False)
    tmp.close()
    return tmp.name


def build_tqu_from_test():
    """
    Build a PixelProjectedCompression from the nside=4 TQU test data.

    Runs the same pipeline as qube's Fisher test: reads the real mask,
    noise covariance, beams, and Cls from the test data directory.
    """
    from qube.fisher import Fisher

    qube_root = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "cosmoforge.qube",
    )
    config_file = _resolve_config(
        "tests/data/nside4/TQU/config.yaml",
        qube_root,
    )

    # Run the Fisher pipeline to get geometry + covariance
    fisher = Fisher(config_file)
    fisher.setup_fields()
    fisher.setup_geometry()
    fisher.setup_covariance_matrices()
    fisher.setup_cls()
    fisher.setup_beams()
    os.unlink(config_file)

    # Build a PixelProjectedCompression from the pipeline products
    from cosmocore.basics import matrix_inverse_symm

    ppc = PixelProjectedCompression(
        N=fisher.NCov1,
        N_inv=matrix_inverse_symm(fisher.NCov1),
        theta=fisher.theta,
        phi=fisher.phi,
        lmax=fisher.params.lmax,
        spins=[f.spin for f in fisher.collection.fields],
    )
    ppc.setup()
    return ppc


def main():
    import matplotlib.pyplot as plt
    import numpy as np

    print("Building TQU setup (nside=4, lmax=8, from qube test data)...")
    ppc = build_tqu_from_test()

    print(f"  n_pix = {ppc.n_pix}")
    print(f"  n_modes = {ppc.n_modes}  (base per component)")
    print(f"  n_components = {ppc.n_components}")
    print(f"  lmax = {ppc.lmax}")
    print()

    # --- Per-field eigenspectra (numeric summary) ---
    per_field = ppc.compute_eigenspectrum_per_field(
        basis="noise_weighted",
    )
    for entry in per_field:
        ev = entry["normalized_eigenvalues"]
        n_sig = int(np.sum(ev > 1e-6))
        print(f"{entry['label']}:")
        print(f"  Total modes: {len(ev)},  significant (>1e-6): {n_sig}")
        if entry["spin"] == 2:
            n_E = int(np.sum(entry["E_normalized"] > 1e-6))
            n_B = int(np.sum(entry["B_normalized"] > 1e-6))
            print(f"  E modes >1e-6: {n_E},  B modes >1e-6: {n_B}")
    print()

    # --- Plot 1: eigenvalue spectra with E/B overlay ---
    print("Plotting per-field eigenvalue spectra...")
    fig1, _ = ppc.plot_eigenvalue_spectrum(
        basis="noise_weighted",
        show_eb_split=True,
        threshold_values=[1e-2, 1e-4, 1e-6],
    )
    fig1.suptitle(
        "Per-field eigenvalue spectra (noise_weighted)",
        fontsize=14,
        y=1.02,
    )

    # --- Plot 2: basis comparison ---
    print("Plotting basis comparison...")
    fig2, _ = ppc.plot_eigenvalue_comparison(
        bases=["harmonic", "noise_weighted"],
    )
    fig2.suptitle("Basis comparison per field", fontsize=14, y=1.02)

    plt.show()


if __name__ == "__main__":
    main()
