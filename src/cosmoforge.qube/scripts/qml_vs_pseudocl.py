"""Compare QML vs pseudo-Cl scatter at nside=32, fsky=10%, delta_ell=10.

Uses the buffer approach: estimate to 2.5*nside, plot science range up to 2*nside.
Theory is shown windowed by each estimator's own window function:
- QML: Fisher-weighted theory via Spectra.convolve_theory_for_inference()
- PCL: bandpower-windowed theory via NaMaster wsp.decouple_cell(wsp.couple_cell(...))

Run: ``uv run --extra pcl python src/cosmoforge.qube/scripts/qml_vs_pseudocl.py``
The output PNG is written to the current working directory.
"""

import os
import tempfile
import time
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt
import yaml

from cosmocore import Bins
from qube import Fisher, Spectra

THEORY_CL_FILE = Path(__file__).resolve().parent / "dls.txt"


def gaussian_fwhm_for_lmax(lmax, beam_at_lmax=0.5):
    sigma = np.sqrt(-2 * np.log(beam_at_lmax) / (lmax * (lmax + 1)))
    return float(np.degrees(sigma * np.sqrt(8 * np.log(2))) * 60)


nside = 32
lmax_science = 2 * nside
lmax_buffer = int(2.5 * nside)
nsims = 1000
sigma = 1.5
mask_gal_cut_deg = 64
delta_ell = 10
lmax_sim = 4 * nside
fwhm_arcmin = gaussian_fwhm_for_lmax(lmax_science)
fwhm_rad = np.radians(fwhm_arcmin / 60.0)

print(
    f"nside={nside}, lmax_science={lmax_science}, "
    f"lmax_buffer={lmax_buffer}, nsims={nsims}"
)
print(f"FWHM={fwhm_arcmin:.1f} arcmin, sigma={sigma}, delta_ell={delta_ell}")

npix = 12 * nside**2
cov = np.zeros((npix, npix))
np.fill_diagonal(cov, sigma**2)

gal_cut = np.radians(mask_gal_cut_deg)
galactic_pixels = hp.query_strip(nside, np.pi / 2 - gal_cut, np.pi / 2 + gal_cut)
mask = np.ones(npix)
mask[galactic_pixels] = 0.0
fsky = np.mean(mask)
print(f"fsky={fsky:.3f}")

raw_cls = np.loadtxt(THEORY_CL_FILE)
ells_file = raw_cls[:, 0].astype(int)
dl2cl = np.ones(len(ells_file))
dl2cl[ells_file > 0] = (
    2 * np.pi / (ells_file[ells_file > 0] * (ells_file[ells_file > 0] + 1))
)

cl_tt_full = np.zeros(lmax_sim + 1)
cl_ee_full = np.zeros(lmax_sim + 1)
cl_bb_full = np.zeros(lmax_sim + 1)
cl_te_full = np.zeros(lmax_sim + 1)
for i, ell_val in enumerate(ells_file):
    if ell_val <= lmax_sim:
        cl_tt_full[ell_val] = raw_cls[i, 1] * dl2cl[i]
        cl_ee_full[ell_val] = raw_cls[i, 2] * dl2cl[i]
        cl_bb_full[ell_val] = raw_cls[i, 3] * dl2cl[i]
        cl_te_full[ell_val] = raw_cls[i, 4] * dl2cl[i]

ells = np.arange(2, lmax_science + 1)
cl_tt_theory = cl_tt_full[2 : lmax_science + 1]

beam = hp.gauss_beam(fwhm_rad, lmax=lmax_sim)

print(f"Generating {nsims} simulations...")
sim_maps_full = np.empty((npix, nsims), dtype=np.float64)
for i in range(nsims):
    np.random.seed(42 + i)
    alms = hp.synalm(
        [cl_tt_full, cl_ee_full, cl_bb_full, cl_te_full], lmax=lmax_sim, new=True
    )
    hp.almxfl(alms[0], beam, inplace=True)
    sim_maps_full[:, i] = hp.alm2map(alms, nside=nside, lmax=lmax_sim)[0]

# =========================================================================
# Pseudo-Cl with NaMaster
# =========================================================================
print("\n--- Pseudo-Cl (NaMaster) ---")
t0 = time.perf_counter()

nmt_lmax = 3 * nside - 1
nmt_lmins = np.arange(2, nmt_lmax + 1, delta_ell)
nmt_lmaxs = np.minimum(nmt_lmins + delta_ell - 1, nmt_lmax)
b = nmt.NmtBin.from_edges(nmt_lmins, nmt_lmaxs + 1)
ells_nmt = b.get_effective_ells()

beam_nmt = hp.gauss_beam(fwhm_rad, lmax=nmt_lmax)
f0 = nmt.NmtField(mask, [sim_maps_full[:, 0]], beam=beam_nmt)
wsp = nmt.NmtWorkspace()
wsp.compute_coupling_matrix(f0, f0, b)

cl_tt_nmt = np.zeros(nmt_lmax + 1)
cl_tt_nmt[: min(lmax_sim + 1, nmt_lmax + 1)] = cl_tt_full[
    : min(lmax_sim + 1, nmt_lmax + 1)
]
pcl_theory_windowed = wsp.decouple_cell(wsp.couple_cell([cl_tt_nmt]))[0]

pcl_spectra = np.zeros((nsims, len(ells_nmt)))
for i in range(nsims):
    f = nmt.NmtField(mask, [sim_maps_full[:, i]], beam=beam_nmt)
    cl_coupled = nmt.compute_coupled_cell(f, f)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    pcl_spectra[i, :] = cl_decoupled[0]

t_pcl = time.perf_counter() - t0
print(f"  Time: {t_pcl:.1f}s ({t_pcl / nsims * 1000:.1f}ms/sim)")

# =========================================================================
# QML with CosmoForge (buffer approach)
# =========================================================================
print(f"\n--- QML (CosmoForge), lmax_buffer={lmax_buffer} ---")

with tempfile.TemporaryDirectory(prefix="qml_vs_pcl_") as tmpdir:
    cov.tofile(os.path.join(tmpdir, "ncvm.bin"))
    hp.write_map(os.path.join(tmpdir, "mask.fits"), mask, overwrite=True)
    sim_maps_3d = sim_maps_full.reshape(1, npix, nsims)
    np.save(os.path.join(tmpdir, "sims.npy"), sim_maps_3d)

    cl_table = np.zeros((raw_cls.shape[0], 7))
    cl_table[:, :5] = raw_cls[:, :5]
    cosmoforge_cl_file = os.path.join(tmpdir, "dls.txt")
    np.savetxt(cosmoforge_cl_file, cl_table, header="ell TT EE BB TE TB EB", fmt="%.16e")

    config = {
        "nside": nside,
        "spins": [0],
        "labels": ["T"],
        "physical_labels": ["T"],
        "do_cross": False,
        "maskfile": os.path.join(tmpdir, "mask.fits"),
        "output_geometry_file": os.path.join(tmpdir, "geometry.dat"),
        "ordering": "RING",
        "inputclfile": cosmoforge_cl_file,
        "input_convention": "Dl",
        "covmatfile1": os.path.join(tmpdir, "ncvm.bin"),
        "covmatfile2": os.path.join(tmpdir, "ncvm.bin"),
        "lmax": lmax_buffer,
        "calibration": 1.0,
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
        "remove_nb": True,
        "nspectra": 1,
    }

    config_file = os.path.join(tmpdir, "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    bins = Bins.fromdeltal(2, lmax_buffer, delta_ell)

    t0 = time.perf_counter()
    fisher = Fisher(config_file, compression={"method": "harmonic"})
    fisher.set_binning(bins)
    fisher.run()
    t_fisher = time.perf_counter() - t0

    t0 = time.perf_counter()
    spectra = Spectra(config_file, fisher=fisher, compression={"method": "harmonic"})
    spectra.set_binning(bins)
    spectra.run()
    t_qml = time.perf_counter() - t0

    qml_spectra = spectra.get_power_spectra(mode="deconvolved")
    qml_ells_bin = spectra.get_effective_ells()

    qml_theory_windowed = spectra.convolve_theory_for_inference(cl_tt_full)

    print(f"  Fisher (binned): {t_fisher:.1f}s")
    print(f"  QML:             {t_qml:.1f}s")

# =========================================================================
# Restrict to science range and align bins
# =========================================================================
qml_keep = qml_ells_bin <= lmax_science
qml_ells = qml_ells_bin[qml_keep]
qml_mean = np.mean(qml_spectra[:, qml_keep], axis=0)
qml_std = np.std(qml_spectra[:, qml_keep], axis=0)
qml_theory_kept = qml_theory_windowed[qml_keep]

pcl_keep = (ells_nmt >= 2) & (ells_nmt <= lmax_science)
pcl_ells = ells_nmt[pcl_keep]
pcl_mean = np.mean(pcl_spectra[:, pcl_keep], axis=0)
pcl_std = np.std(pcl_spectra[:, pcl_keep], axis=0)
pcl_theory_kept = pcl_theory_windowed[pcl_keep]

n_match = min(len(qml_ells), len(pcl_ells))

# =========================================================================
# Knox formula — optimal-estimator variance bound on partial sky.
# Per-ell:   σ²(C_ℓ) = 2 (C_ℓ + N_ℓ/b²_ℓ)² / [(2ℓ+1) f_sky]
# Per-bin:   σ²(C_b) = 1 / Σ_{ℓ∈bin} 1/σ²(C_ℓ)         (inverse-variance sum)
# QML approaches this bound; PCL exceeds it (suboptimal).
# =========================================================================
omega_pix = 4.0 * np.pi / npix
n_ell_white = sigma**2 * omega_pix
ell_grid_full = np.arange(lmax_sim + 1)
b_ell = beam.copy()
b_ell[b_ell < 1e-12] = 1e-12
n_ell_eff = n_ell_white / b_ell**2
sigma2_perell = np.where(
    ell_grid_full >= 2,
    2.0 * (cl_tt_full + n_ell_eff) ** 2 / np.maximum(2 * ell_grid_full + 1, 1) / fsky,
    np.inf,
)

knox_std = np.zeros(len(qml_ells))
for bi, ell_eff in enumerate(qml_ells):
    bin_idx = np.argmin(np.abs(bins.lbin - ell_eff))
    lo, hi = bins.lmins[bin_idx], bins.lmaxs[bin_idx]
    inv_var_sum = np.sum(1.0 / sigma2_perell[lo : hi + 1])
    knox_std[bi] = np.sqrt(1.0 / inv_var_sum) if inv_var_sum > 0 else np.inf

print("\n" + "=" * 110)
print("QML vs PCL vs Knox — windowed theory means and variance ratios")
print("=" * 110)
print(
    f"{'ell':>6} {'QML mean':>12} {'QML F-th':>10} | "
    f"{'PCL mean':>12} {'PCL B-th':>10} | "
    f"{'σ_Knox':>10} {'σQML/Kx':>8} {'σPCL/Kx':>8} {'σPCL/QML':>9}"
)
print("-" * 110)
for i in range(n_match):
    rq = qml_mean[i] / qml_theory_kept[i] if qml_theory_kept[i] != 0 else 0
    rp = pcl_mean[i] / pcl_theory_kept[i] if pcl_theory_kept[i] != 0 else 0
    qk = qml_std[i] / knox_std[i] if knox_std[i] > 0 else float("inf")
    pk = pcl_std[i] / knox_std[i] if knox_std[i] > 0 else float("inf")
    sr = pcl_std[i] / qml_std[i] if qml_std[i] > 0 else float("inf")
    print(
        f"{qml_ells[i]:>6.1f} {qml_mean[i]:>12.4e} {qml_theory_kept[i]:>10.3e} | "
        f"{pcl_mean[i]:>12.4e} {pcl_theory_kept[i]:>10.3e} | "
        f"{knox_std[i]:>10.3e} {qk:>7.2f}x {pk:>7.2f}x {sr:>8.2f}x"
    )

ratio = pcl_std[:n_match] / qml_std[:n_match]
ratio_valid = ratio[np.isfinite(ratio) & (ratio > 0)]
qml_over_knox = qml_std[:n_match] / knox_std[:n_match]
pcl_over_knox = pcl_std[:n_match] / knox_std[:n_match]
print(
    f"\nσ_PCL / σ_QML — mean: {np.mean(ratio_valid):.2f}x, "
    f"median: {np.median(ratio_valid):.2f}x, "
    f"range: {np.min(ratio_valid):.2f}x — {np.max(ratio_valid):.2f}x"
)
print(
    f"σ_QML / σ_Knox — mean: {np.mean(qml_over_knox):.2f}x, "
    f"median: {np.median(qml_over_knox):.2f}x"
)
print(
    f"σ_PCL / σ_Knox — mean: {np.mean(pcl_over_knox):.2f}x, "
    f"median: {np.median(pcl_over_knox):.2f}x"
)

# =========================================================================
# Plot
# =========================================================================
import matplotlib.pyplot as plt

dl_factor_full = ells * (ells + 1) / (2 * np.pi)
dl_factor_qml = qml_ells * (qml_ells + 1) / (2 * np.pi)
dl_factor_pcl = pcl_ells * (pcl_ells + 1) / (2 * np.pi)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

dl_theory_full = cl_tt_theory * dl_factor_full
dl_qml_mean = qml_mean * dl_factor_qml
dl_qml_std = qml_std * dl_factor_qml
dl_qml_th = qml_theory_kept * dl_factor_qml
dl_pcl_mean = pcl_mean * dl_factor_pcl
dl_pcl_std = pcl_std * dl_factor_pcl
dl_pcl_th = pcl_theory_kept * dl_factor_pcl

ax1.plot(ells, dl_theory_full, "k-", lw=1.2, label=r"Input theory $C_\ell$", zorder=4)
ax1.plot(
    qml_ells,
    dl_qml_th,
    "C0--",
    lw=1.0,
    marker="x",
    ms=8,
    mew=1.5,
    label="QML theory (Fisher-windowed)",
    zorder=3,
)
ax1.plot(
    pcl_ells,
    dl_pcl_th,
    "C1--",
    lw=1.0,
    marker="x",
    ms=8,
    mew=1.5,
    label="PCL theory (bandpower-windowed)",
    zorder=3,
)
ax1.errorbar(
    qml_ells - 0.5,
    dl_qml_mean,
    yerr=dl_qml_std,
    fmt="o",
    ms=6,
    capsize=3,
    color="C0",
    label="QML estimates",
    zorder=2,
)
ax1.errorbar(
    pcl_ells + 0.5,
    dl_pcl_mean,
    yerr=dl_pcl_std,
    fmt="s",
    ms=6,
    capsize=3,
    color="C1",
    label=r"Pseudo-$C_\ell$ (NaMaster)",
    zorder=1,
)
ax1.set_xlabel(r"$\ell$")
ax1.set_ylabel(r"$D_\ell = \ell(\ell+1)C_\ell / 2\pi$")
ax1.set_title(
    f"TT Power Spectrum (nside={nside}, "
    f"$f_{{\\rm sky}}$={fsky:.2f}, $\\Delta\\ell$={delta_ell})"
)
ax1.legend(fontsize=8, loc="upper right")
ax1.set_xlim(0, lmax_science + 2)

ells_plot = qml_ells[:n_match]
ax2.axhline(1, color="k", ls="--", lw=0.8, label="Knox bound")
ax2.plot(
    ells_plot,
    qml_over_knox,
    "o-",
    ms=6,
    color="C0",
    label=r"$\sigma_{\rm QML}/\sigma_{\rm Knox}$",
)
ax2.plot(
    ells_plot,
    pcl_over_knox,
    "s-",
    ms=6,
    color="C1",
    label=r"$\sigma_{\rm PCL}/\sigma_{\rm Knox}$",
)
ax2.fill_between(
    ells_plot, 1, pcl_over_knox, where=(pcl_over_knox > 1), alpha=0.15, color="C1"
)
ax2.fill_between(
    ells_plot, 1, qml_over_knox, where=(qml_over_knox > 1), alpha=0.25, color="C0"
)
ax2.set_xlabel(r"$\ell$")
ax2.set_ylabel(r"$\sigma / \sigma_{\rm Knox}$")
ax2.set_title(
    f"Variance vs Knox (QML median = {np.median(qml_over_knox):.2f}x, "
    f"PCL median = {np.median(pcl_over_knox):.2f}x)"
)
ax2.legend(fontsize=9, loc="upper right")
ax2.set_xlim(0, lmax_science + 2)
ymax = max(2.0, np.max(pcl_over_knox) * 1.1)
ax2.set_ylim(0.9, ymax)

plt.tight_layout()
outname = f"qml_vs_pcl_fsky{fsky:.2f}_dl{delta_ell}.png"
plt.savefig(outname, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {outname}")
