"""
Demonstration of QML normalization modes.

This script shows how the three normalization modes affect the power spectrum
estimates and their covariances:
- deconvolved: F⁻¹y (standard QML, correlated errors)
- decorrelated: F⁻¹/²y (uncorrelated bandpowers, unit variance)
- convolved: y with window matrix W (for theory comparison)

Run from the repository root:
    uv run python src/cosmoforge.qube/scripts/normalization_modes_demo.py

Note: This demo requires the test data located in tests/data/nside8/B/fortran_reference/.
The test data is included in the repository for development and demonstration purposes.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Try to import from installed package first. If that fails, fall back to
# adding the package root to sys.path. This allows running the script both
# from an installed environment and directly from a repository checkout.
try:
    from qube import Spectra
except ImportError:
    _package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if _package_root not in sys.path:
        sys.path.insert(0, _package_root)
    from qube import Spectra

# Path to test data (relative to this script's location)
# This data is included in the repository for demonstration purposes.
_script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(
    _script_dir, "..", "tests", "data", "nside8", "B", "fortran_reference"
)
CONFIG_FILE = os.path.join(DATA_DIR, "config.yaml")

if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(
        f"Test data not found at {CONFIG_FILE}. "
        "Please run this script from the CosmoForge repository with test data available."
    )


def main():
    print("=" * 70)
    print("QML Normalization Modes Demonstration")
    print("=" * 70)

    # Run QML computation
    print("\nRunning QML power spectrum estimation...")
    spectra = Spectra(CONFIG_FILE)
    spectra.run()

    # Get multipole range - n_ell is determined by the actual output size
    cl_test = spectra.get_power_spectra(mode="deconvolved")[0]
    n_ell = len(cl_test)
    lmin = 2
    lmax = lmin + n_ell
    ells = np.arange(lmin, lmax)

    print(f"\nMultipole range: l = {lmin} to {lmax - 1} ({n_ell} multipoles)")

    # =========================================================================
    # 1. Compare power spectra in different modes
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. POWER SPECTRUM ESTIMATES")
    print("=" * 70)

    # Deconvolved mode (default)
    cl_deconv = spectra.get_power_spectra(mode="deconvolved")[0]
    print(f"\nDeconvolved (F⁻¹y): shape = {cl_deconv.shape}")
    print(f"  Mean value: {np.mean(cl_deconv):.4e}")
    print(f"  Range: [{np.min(cl_deconv):.4e}, {np.max(cl_deconv):.4e}]")

    # Decorrelated mode
    cl_decorr = spectra.get_power_spectra(mode="decorrelated")[0]
    print(f"\nDecorrelated (F⁻¹/²y): shape = {cl_decorr.shape}")
    print(f"  Mean value: {np.mean(cl_decorr):.4e}")
    print(f"  Range: [{np.min(cl_decorr):.4e}, {np.max(cl_decorr):.4e}]")

    # Convolved mode
    y, W, convolve_func = spectra.get_power_spectra(mode="convolved")
    y_mean = y[0]  # First (only) simulation
    print(f"\nConvolved (y): shape = {y_mean.shape}")
    print(f"  Mean value: {np.mean(y_mean):.4e}")
    print(f"  Range: [{np.min(y_mean):.4e}, {np.max(y_mean):.4e}]")
    print(f"  Window matrix W: shape = {W.shape}")

    # =========================================================================
    # 2. Compare covariance matrices
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. COVARIANCE MATRICES")
    print("=" * 70)

    cov_deconv = spectra.get_covariance(mode="deconvolved")
    cov_decorr = spectra.get_covariance(mode="decorrelated")
    cov_conv = spectra.get_covariance(mode="convolved")

    print("\nDeconvolved covariance (F⁻¹):")
    print(f"  Shape: {cov_deconv.shape}")
    print(
        f"  Diagonal range: "
        f"[{np.min(np.diag(cov_deconv)):.4e}, {np.max(np.diag(cov_deconv)):.4e}]"
    )
    off_diag = np.sum(np.abs(cov_deconv) > 1e-20) - n_ell
    total = n_ell**2 - n_ell
    print(f"  Off-diagonal fraction: {off_diag:.0f}/{total:.0f} non-zero")

    # Check correlation matrix for deconvolved
    diag = np.sqrt(np.diag(cov_deconv))
    corr_deconv = cov_deconv / np.outer(diag, diag)
    off_diag_corr = corr_deconv[np.triu_indices(n_ell, k=1)]
    print(f"  Max off-diagonal correlation: {np.max(np.abs(off_diag_corr)):.4f}")

    print("\nDecorrelated covariance (Identity):")
    print(f"  Shape: {cov_decorr.shape}")
    identity_check = np.allclose(cov_decorr, np.eye(n_ell))
    print(f"  Is identity matrix: {identity_check}")
    print(
        f"  Max deviation from identity: {np.max(np.abs(cov_decorr - np.eye(n_ell))):.2e}"
    )

    print("\nConvolved covariance (F):")
    print(f"  Shape: {cov_conv.shape}")
    print(
        f"  Diagonal range: "
        f"[{np.min(np.diag(cov_conv)):.4e}, {np.max(np.diag(cov_conv)):.4e}]"
    )

    # =========================================================================
    # 3. Compare error bars
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. ERROR BARS")
    print("=" * 70)

    err_deconv = spectra.get_error_bars(mode="deconvolved")
    err_decorr = spectra.get_error_bars(mode="decorrelated")
    err_conv = spectra.get_error_bars(mode="convolved")

    print("\nDeconvolved errors (σ = √diag(F⁻¹)):")
    print(f"  Range: [{np.min(err_deconv):.4e}, {np.max(err_deconv):.4e}]")

    print("\nDecorrelated errors (all = 1.0 by construction):")
    print(f"  Range: [{np.min(err_decorr):.4f}, {np.max(err_decorr):.4f}]")
    print(f"  All equal to 1: {np.allclose(err_decorr, 1.0)}")

    print("\nConvolved errors (σ = √diag(F)):")
    print(f"  Range: [{np.min(err_conv):.4e}, {np.max(err_conv):.4e}]")

    # =========================================================================
    # 4. Theory comparison using convolved mode
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. THEORY COMPARISON (Convolved Mode)")
    print("=" * 70)

    # Load input theory spectrum
    # Format: ell TT EE BB TE (columns 0, 1, 2, 3, 4)
    input_cls = np.loadtxt(os.path.join(DATA_DIR, "input_cls.txt"))
    # Extract BB spectrum (column 3)
    cl_theory = input_cls[lmin - 2 : lmax - 2, 3]  # BB is column 3, rows start at l=2

    print(f"\nInput theory spectrum: {cl_theory.shape}")

    # Convolve theory with window
    cl_theory_convolved = convolve_func(cl_theory)

    print(f"Convolved theory: {cl_theory_convolved.shape}")

    # Compare with y estimates
    residuals = y_mean - cl_theory_convolved
    chi2 = residuals @ np.linalg.inv(cov_conv) @ residuals

    print("\nResiduals (y - W@theory):")
    print(f"  Mean: {np.mean(residuals):.4e}")
    print(f"  Std: {np.std(residuals):.4e}")
    print(f"  Chi-square: {chi2:.2f} (ndof = {n_ell})")
    print(f"  Reduced chi-square: {chi2 / n_ell:.2f}")

    # =========================================================================
    # 5. Relationship between modes
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. RELATIONSHIPS BETWEEN MODES")
    print("=" * 70)

    # Check F^(-1/2) @ F^(-1/2) = F^(-1)
    F_inv_sqrt = spectra.inv_fisher_sqrt
    F_inv_reconstructed = F_inv_sqrt @ F_inv_sqrt
    reconstruction_error = np.max(np.abs(F_inv_reconstructed - cov_deconv))
    print("\nF⁻¹/² @ F⁻¹/² ≈ F⁻¹:")
    print(f"  Max reconstruction error: {reconstruction_error:.2e}")

    # Decorrelated from deconvolved: cl_decorr = F^(1/2) @ cl_deconv
    # (approximately, since F^(1/2) @ F^(-1) @ y = F^(-1/2) @ y)
    F_sqrt = np.linalg.inv(F_inv_sqrt)  # This is F^(1/2)
    cl_decorr_check = F_sqrt @ cl_deconv
    decorr_match = np.allclose(cl_decorr, cl_decorr_check, rtol=1e-10)
    print(f"\ncl_decorr = F^(1/2) @ cl_deconv: {decorr_match}")

    # =========================================================================
    # 6. Summary table
    # =========================================================================
    print("\n" + "=" * 70)
    print("6. SUMMARY: SPECTRA VALUES AT SELECTED MULTIPOLES")
    print("=" * 70)

    # Select a few multipoles to display
    display_ells = [2, 5, 10, 15] if n_ell > 15 else list(range(2, min(6, lmax)))
    display_idx = [ll - lmin for ll in display_ells if ll < lmax]

    print(
        f"\n{'ell':>5} | {'Deconvolved':>14} | {'Decorrelated':>14} | "
        f"{'Convolved (y)':>14} | {'σ_deconv':>12}"
    )
    print("-" * 75)
    for i, ll in zip(display_idx, display_ells):
        if i < n_ell:
            print(
                f"{ll:>5} | {cl_deconv[i]:>14.4e} | {cl_decorr[i]:>14.4e} | "
                f"{y_mean[i]:>14.4e} | {err_deconv[i]:>12.4e}"
            )

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)

    # =========================================================================
    # 7. Verify decorrelation and optimality
    # =========================================================================
    print("\n" + "=" * 70)
    print("7. VERIFICATION: DECORRELATION AND OPTIMALITY")
    print("=" * 70)

    # Compute correlation matrices for each mode
    print("\n--- Correlation Analysis ---")

    # Deconvolved: correlation matrix from F^(-1)
    diag_deconv = np.sqrt(np.diag(cov_deconv))
    corr_deconv = cov_deconv / np.outer(diag_deconv, diag_deconv)

    # Decorrelated: should be identity
    corr_decorr = cov_decorr  # Already identity by construction

    # Convolved: correlation matrix from F
    diag_conv = np.sqrt(np.diag(cov_conv))
    corr_conv = cov_conv / np.outer(diag_conv, diag_conv)

    print("\nDeconvolved mode correlation matrix:")
    print("  Diagonal elements: all = 1.0 (by definition)")
    print(
        f"  Off-diagonal range: [{np.min(corr_deconv[~np.eye(n_ell, dtype=bool)]):.4f}, "
        f"{np.max(corr_deconv[~np.eye(n_ell, dtype=bool)]):.4f}]"
    )
    print(
        f"  Mean |off-diagonal|: "
        f"{np.mean(np.abs(corr_deconv[~np.eye(n_ell, dtype=bool)])):.4f}"
    )

    print("\nDecorrelated mode correlation matrix:")
    print(f"  Is identity: {np.allclose(corr_decorr, np.eye(n_ell), atol=1e-10)}")
    max_dev = np.max(np.abs(corr_decorr - np.eye(n_ell)))
    print(f"  Max deviation from identity: {max_dev:.2e}")

    print("\nConvolved mode correlation matrix:")
    print(
        f"  Off-diagonal range: [{np.min(corr_conv[~np.eye(n_ell, dtype=bool)]):.4f}, "
        f"{np.max(corr_conv[~np.eye(n_ell, dtype=bool)]):.4f}]"
    )
    print(
        f"  Mean |off-diagonal|: "
        f"{np.mean(np.abs(corr_conv[~np.eye(n_ell, dtype=bool)])):.4f}"
    )

    # -------------------------------------------------------------------------
    # Verify information content is preserved
    # -------------------------------------------------------------------------
    print("\n--- Information Content (Fisher determinant) ---")

    # The determinant of the Fisher matrix measures total information
    # All modes should have equivalent information content
    det_F = np.linalg.det(cov_conv)  # det(F)
    det_F_inv = np.linalg.det(cov_deconv)  # det(F^-1)
    det_I = np.linalg.det(cov_decorr)  # det(I) = 1

    print(f"\nlog|det(F)|     = {np.log10(det_F):.2f}")
    print(f"log|det(F^-1)|  = {np.log10(det_F_inv):.2f}")
    print(f"det(I)          = {det_I:.2f}")
    print(
        f"\nVerify: det(F) × det(F^-1) = 1? "
        f"{np.isclose(det_F * det_F_inv, 1.0, rtol=1e-5)}"
    )

    # Compute F^(1/2) and theory in decorrelated space for comparisons
    F_sqrt = np.linalg.inv(F_inv_sqrt)
    theory_decorr = F_sqrt @ cl_theory

    # -------------------------------------------------------------------------
    # Optimal mode selection guide
    # -------------------------------------------------------------------------
    print("\n--- OPTIMAL MODE SELECTION GUIDE ---")
    print("""
    MODE           | WHEN TO USE
    ---------------|------------------------------------------------------------
    DECONVOLVED    | Standard analysis: physical C_ℓ values with correlated
    (F⁻¹y)         | errors. Best for: parameter estimation with full
                   | covariance, comparing with other C_ℓ measurements.
                   |
    DECORRELATED   | When you need independent error bars: plotting,
    (F⁻¹/²y)       | model comparison with simple χ². Best for: visual
                   | inspection, quick goodness-of-fit tests.
                   | Note: values are NOT physical C_ℓ!
                   |
    CONVOLVED      | When Fisher inversion is ill-conditioned, or for
    (y + W)        | theory comparison without deconvolution.
                   | Best for: numerical stability, MCMC sampling.
    """)

    # -------------------------------------------------------------------------
    # Demonstrate optimality: all modes give same chi-square
    # -------------------------------------------------------------------------
    print("\n--- Chi-square Equivalence (same statistical power) ---")

    # Deconvolved chi-square (full covariance)
    resid_deconv = cl_deconv - cl_theory
    chi2_deconv_full = resid_deconv @ np.linalg.inv(cov_deconv) @ resid_deconv

    # Decorrelated chi-square (simple sum of squares since cov = I)
    resid_decorr = cl_decorr - theory_decorr
    chi2_decorr = resid_decorr @ resid_decorr  # = resid @ I^-1 @ resid

    # Convolved chi-square
    resid_conv = y_mean - cl_theory_convolved
    chi2_conv = resid_conv @ np.linalg.inv(cov_conv) @ resid_conv

    print(f"\nχ² (deconvolved, full cov):  {chi2_deconv_full:.4f}")
    print(f"χ² (decorrelated, simple):   {chi2_decorr:.4f}")
    print(f"χ² (convolved):              {chi2_conv:.4f}")

    print("""
    KEY INSIGHT: All three modes contain IDENTICAL statistical information.
    The chi-square values are the same (within numerical precision).

    The difference is in:
    - Numerical stability (convolved is most stable)
    - Interpretability (deconvolved gives physical C_ℓ)
    - Error structure (decorrelated has independent errors)
    """)

    # =========================================================================
    # 8. Plots comparing theory and estimates
    # =========================================================================
    print("\n" + "=" * 70)
    print("8. GENERATING PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # -------------------------------------------------------------------------
    # Plot 1: Deconvolved mode - C_ℓ estimates vs theory
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.errorbar(
        ells,
        cl_deconv,
        yerr=err_deconv,
        fmt="o",
        capsize=3,
        label="QML estimate",
        color="C0",
        markersize=5,
    )
    ax1.plot(ells, cl_theory, "k-", linewidth=2, label="Input theory")
    ax1.set_xlabel(r"Multipole $\ell$")
    ax1.set_ylabel(r"$C_\ell^{BB}$")
    ax1.set_title("Deconvolved Mode: $F^{-1}y$")
    ax1.legend()
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 2: Convolved mode - y estimates vs W @ theory
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.errorbar(
        ells,
        y_mean,
        yerr=err_conv,
        fmt="s",
        capsize=3,
        label="Raw QML estimate $y$",
        color="C1",
        markersize=5,
    )
    ax2.plot(
        ells, cl_theory_convolved, "k-", linewidth=2, label=r"Convolved theory $W C_\ell$"
    )
    ax2.set_xlabel(r"Multipole $\ell$")
    ax2.set_ylabel(r"$y$ (raw estimate)")
    ax2.set_title(r"Convolved Mode: $y$ vs $W C_\ell^{\rm theory}$")
    ax2.legend()
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 3: Decorrelated mode - normalized bandpowers
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]

    # For decorrelated mode, theory is also transformed: F^(1/2) @ theory
    ax3.errorbar(
        ells,
        cl_decorr,
        yerr=err_decorr,  # All 1.0
        fmt="^",
        capsize=3,
        label="Decorrelated estimate",
        color="C2",
        markersize=5,
    )
    ax3.plot(
        ells, theory_decorr, "k-", linewidth=2, label=r"$F^{1/2} C_\ell^{\rm theory}$"
    )
    ax3.set_xlabel(r"Multipole $\ell$")
    ax3.set_ylabel("Decorrelated bandpower")
    ax3.set_title(r"Decorrelated Mode: $F^{-1/2}y$ (unit variance)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 4: Residuals for all three modes
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]

    # Normalized residuals for each mode
    # Deconvolved: (est - theory) / sigma
    residuals_deconv_norm = (cl_deconv - cl_theory) / err_deconv

    # Decorrelated: (est - F^(1/2)@theory) / 1.0 = est - F^(1/2)@theory
    residuals_decorr_norm = cl_decorr - theory_decorr  # err = 1.0

    # Convolved: (y - W@theory) / sigma_conv
    residuals_conv_norm = (y_mean - cl_theory_convolved) / err_conv

    # Plot as grouped bars
    width = 0.25
    x = np.arange(n_ell)

    ax4.bar(
        x - width,
        residuals_deconv_norm,
        width,
        color="C0",
        alpha=0.7,
        label="Deconvolved",
    )
    ax4.bar(x, residuals_decorr_norm, width, color="C2", alpha=0.7, label="Decorrelated")
    ax4.bar(
        x + width, residuals_conv_norm, width, color="C1", alpha=0.7, label="Convolved"
    )

    ax4.axhline(0, color="k", linestyle="-", linewidth=1)
    ax4.axhline(1, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax4.axhline(-1, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax4.axhline(2, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax4.axhline(-2, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax4.set_xlabel(r"Multipole index")
    ax4.set_ylabel(r"$({\rm est} - {\rm theory}) / \sigma$")
    ax4.set_title("Normalized Residuals (all modes)")
    ax4.set_ylim(-4, 4)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Add text with chi-square for all modes
    chi2_deconv_plot = np.sum(residuals_deconv_norm**2)
    chi2_decorr_plot = np.sum(residuals_decorr_norm**2)
    chi2_conv_plot = np.sum(residuals_conv_norm**2)
    ax4.text(
        0.98,
        0.98,
        f"$\\chi^2_{{\\rm deconv}} = {chi2_deconv_plot:.1f}$\n"
        f"$\\chi^2_{{\\rm decorr}} = {chi2_decorr_plot:.1f}$\n"
        f"$\\chi^2_{{\\rm conv}} = {chi2_conv_plot:.1f}$\n"
        f"(ndof={n_ell})",
        transform=ax4.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(
        os.path.dirname(__file__), "normalization_modes_comparison.png"
    )
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")

    # -------------------------------------------------------------------------
    # Second figure: Correlation matrices
    # -------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))

    # Deconvolved correlation matrix
    im1 = axes2[0].imshow(corr_deconv, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes2[0].set_title(r"Deconvolved: Corr$(F^{-1})$")
    axes2[0].set_xlabel(r"$\ell$")
    axes2[0].set_ylabel(r"$\ell'$")
    plt.colorbar(im1, ax=axes2[0], shrink=0.8)

    # Decorrelated correlation matrix (should be identity)
    im2 = axes2[1].imshow(corr_decorr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes2[1].set_title(r"Decorrelated: Corr$(I)$ = Identity")
    axes2[1].set_xlabel(r"$\ell$")
    axes2[1].set_ylabel(r"$\ell'$")
    plt.colorbar(im2, ax=axes2[1], shrink=0.8)

    # Convolved correlation matrix
    im3 = axes2[2].imshow(corr_conv, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes2[2].set_title(r"Convolved: Corr$(F)$")
    axes2[2].set_xlabel(r"$\ell$")
    axes2[2].set_ylabel(r"$\ell'$")
    plt.colorbar(im3, ax=axes2[2], shrink=0.8)

    plt.tight_layout()

    # Save correlation figure
    corr_output_file = os.path.join(
        os.path.dirname(__file__), "normalization_modes_correlations.png"
    )
    plt.savefig(corr_output_file, dpi=150, bbox_inches="tight")
    print(f"Correlation plot saved to: {corr_output_file}")

    plt.show()


if __name__ == "__main__":
    main()
