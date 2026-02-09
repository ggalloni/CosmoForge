"""
Inspect per-field eigenspectra for a mixed spin-0 + spin-2 (TQU) setup.

Creates a PixelProjectedCompression with one temperature (spin-0) field
and one polarization (spin-2) field, then plots:
  1. Per-field eigenvalue spectra (with E/B overlay for the spin-2 field)
  2. Per-field basis comparison (harmonic vs noise_weighted)

Usage:
    uv run python scripts/inspect_eigenspectra.py
"""

import numpy as np

from cosmocore.compression import PixelProjectedCompression


def build_tqu_setup(n_pix_t=40, n_pix_p=30, lmax=8, seed=42):
    """Build a mixed spin-0 + spin-2 test case."""
    rng = np.random.default_rng(seed)

    # Pixel positions (golden spiral for uniform coverage)
    golden_ratio = (1 + np.sqrt(5)) / 2

    def golden_positions(n, offset=0):
        indices = np.arange(n)
        theta = np.arccos(1 - 2 * (indices + 0.5) / n)
        phi = (2 * np.pi * (indices + offset) / golden_ratio) % (2 * np.pi)
        return theta, phi

    theta_t, phi_t = golden_positions(n_pix_t, offset=0)
    theta_p, phi_p = golden_positions(n_pix_p, offset=n_pix_t)

    # Block-diagonal noise: T has lower noise than QU
    total_pix = n_pix_t + 2 * n_pix_p
    noise_var = np.empty(total_pix)
    noise_var[:n_pix_t] = 0.01  # T noise
    noise_var[n_pix_t:] = 0.05  # Q/U noise (higher)

    # Add some pixel-dependent scatter
    noise_var *= 1 + 0.3 * rng.standard_normal(total_pix) ** 2

    N = np.diag(noise_var)
    N_inv = np.diag(1.0 / noise_var)

    return {
        "N": N,
        "N_inv": N_inv,
        "theta": (theta_t, theta_p),
        "phi": (phi_t, phi_p),
        "lmax": lmax,
        "n_pix_t": n_pix_t,
        "n_pix_p": n_pix_p,
    }


def main():
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("TkAgg")

    print("Building TQU setup (spin-0 T + spin-2 QU)...")
    setup = build_tqu_setup()

    ppc = PixelProjectedCompression(
        N=setup["N"],
        N_inv=setup["N_inv"],
        theta=setup["theta"],
        phi=setup["phi"],
        lmax=setup["lmax"],
        spins=[0, 2],
    )
    ppc.setup()

    print(f"  n_pix = {ppc.n_pix}  (T: {setup['n_pix_t']}, QU: 2x{setup['n_pix_p']})")
    print(f"  n_modes = {ppc.n_modes}  (base per component)")
    print(f"  n_components = {ppc.n_components}")
    print(f"  lmax = {ppc.lmax}")
    print()

    # --- Per-field eigenspectra (numeric summary) ---
    per_field = ppc.compute_eigenspectrum_per_field(basis="noise_weighted")
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
    fig1, axes1 = ppc.plot_eigenvalue_spectrum(
        basis="noise_weighted",
        show_eb_split=True,
        threshold_values=[1e-2, 1e-4, 1e-6],
    )
    fig1.suptitle("Per-field eigenvalue spectra (noise_weighted)", fontsize=14, y=1.02)

    # --- Plot 2: basis comparison ---
    print("Plotting basis comparison...")
    fig2, axes2 = ppc.plot_eigenvalue_comparison(
        bases=["harmonic", "noise_weighted"],
    )
    fig2.suptitle("Basis comparison per field", fontsize=14, y=1.02)

    plt.show()


if __name__ == "__main__":
    main()
