"""Compare QML vs pseudo-Cl scatter at two sky fractions (TT, galactic strip).

Methodology demonstration:
  - QML approaches the optimal Knox bound for the available information
  - PCL is suboptimal at low fsky due to mode coupling
  - QML decorrelated mode produces empirically identity covariance
  - Empirical covariances match analytic predictions (F^-1, F)

Two configurations (galactic-strip mask, TT only, nside=32):
  - low fsky  ~ 0.10  (gal cut 64 deg)
  - high fsky ~ 0.60  (gal cut 24 deg)

For each configuration:
  - PCL via NaMaster (single decoupled mode)
  - QML via CosmoForge in three normalization modes
    (deconvolved, decorrelated, convolved)
  - Sims include CMB signal + diagonal white noise so that the Knox
    bound is a meaningful absolute reference

Diagnostics per case:
  - Per-bin mean and standard deviation
  - Full empirical bandpower covariance vs analytic prediction
  - chi^2 of the sample mean against the windowed theory (tests bias)
  - Mean per-realisation chi^2 against analytic covariance (tests cov)
  - QML decorrelated: off-diagonal RMS of empirical correlation matrix
  - Variance ratio against Knox at this fsky and at full sky
  - Wall-clock timings

Outputs:
  - qml_vs_pcl_results.json   — all numbers (full covariances included)
  - qml_vs_pcl_dl_variance.png — Dl bandpowers + variance ratios per case
  - qml_vs_pcl_correlations.png — bandpower correlation heatmaps

Run: ``uv run --extra pcl python src/cosmoforge.qube/scripts/qml_vs_pseudocl.py``
"""

import json
import os
import tempfile
import time
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt
import yaml

from cosmocore import Bins
from qube import Fisher, Spectra

THEORY_CL_FILE = Path(__file__).resolve().parent / "dls.txt"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NSIDE = 32
LMAX_SCIENCE = 2 * NSIDE
LMAX_BUFFER = int(2.5 * NSIDE)
LMAX_SIM = 4 * NSIDE
NSIMS = 1000
SIGMA_NOISE = 1.5
ADD_NOISE_TO_SIMS = True

CASES = [
    {
        "name": "low_fsky",
        "gal_cut_deg": 64,
        "label": r"$f_{\rm sky}\sim 0.10$",
        "basis": {"method": "auto"},  # n_pix < n_modes -> pixel-direct
        "delta_ell": 10,
    },  # bin: modes are mode-coupled
    {
        "name": "high_fsky",
        "gal_cut_deg": 24,
        "label": r"$f_{\rm sky}\sim 0.60$",
        "basis": {"method": "harmonic"},
        "delta_ell": 1,
    },  # unbinned: modes nearly independent
]


def gaussian_fwhm_for_lmax(lmax, beam_at_lmax=0.5):
    sigma = np.sqrt(-2 * np.log(beam_at_lmax) / (lmax * (lmax + 1)))
    return float(np.degrees(sigma * np.sqrt(8 * np.log(2))) * 60)


FWHM_ARCMIN = gaussian_fwhm_for_lmax(LMAX_SCIENCE)
FWHM_RAD = np.radians(FWHM_ARCMIN / 60.0)
NPIX = 12 * NSIDE**2


# ---------------------------------------------------------------------------
# Plot styling (matches src/cosmoforge.cosmocore/outputs/QU_spectra_test_*.ipynb)
# ---------------------------------------------------------------------------
def configure_plt():
    plt.rc("axes", labelsize=16, linewidth=1.5)
    plt.rc("xtick", direction="in", labelsize=12, top=True)
    plt.rc("ytick", direction="in", labelsize=12, right=True)
    plt.rc("xtick.major", width=1.1, size=5)
    plt.rc("ytick.major", width=1.1, size=5)
    plt.rc("xtick.minor", width=1.1, size=3)
    plt.rc("ytick.minor", width=1.1, size=3)
    plt.rc("lines", linewidth=2)
    plt.rc("legend", frameon=False, fontsize=11)
    plt.rc("figure", dpi=100, autolayout=True)
    plt.rc("savefig", dpi=200, bbox="tight")


C_QML = "dodgerblue"
C_QML_DARK = "darkblue"
C_PCL = "orange"
C_PCL_DARK = "chocolate"
C_KNOX = "black"


# ---------------------------------------------------------------------------
# Theory loading
# ---------------------------------------------------------------------------
def load_theory(lmax_sim):
    raw_cls = np.loadtxt(THEORY_CL_FILE)
    ells_file = raw_cls[:, 0].astype(int)
    dl2cl = np.ones(len(ells_file))
    nz = ells_file > 0
    dl2cl[nz] = 2 * np.pi / (ells_file[nz] * (ells_file[nz] + 1))

    cl_full = {k: np.zeros(lmax_sim + 1) for k in ("TT", "EE", "BB", "TE")}
    cols = {"TT": 1, "EE": 2, "BB": 3, "TE": 4}
    for i, ell in enumerate(ells_file):
        if ell <= lmax_sim:
            for k, c in cols.items():
                cl_full[k][ell] = raw_cls[i, c] * dl2cl[i]
    return raw_cls, cl_full


# ---------------------------------------------------------------------------
# Mask + sims
# ---------------------------------------------------------------------------
def make_galactic_strip(nside, gal_cut_deg):
    npix = 12 * nside**2
    gal_cut = np.radians(gal_cut_deg)
    galactic_pixels = hp.query_strip(nside, np.pi / 2 - gal_cut, np.pi / 2 + gal_cut)
    mask = np.ones(npix)
    mask[galactic_pixels] = 0.0
    return mask, float(np.mean(mask))


def generate_sims(cl_full, beam, nside, npix, nsims, lmax_sim, sigma_noise, add_noise):
    label = "signal+noise" if add_noise else "signal-only"
    print(f"Generating {nsims} {label} full-sky sims...")
    sim_maps = np.empty((npix, nsims), dtype=np.float64)
    cls = [cl_full["TT"], cl_full["EE"], cl_full["BB"], cl_full["TE"]]
    for i in range(nsims):
        np.random.seed(42 + i)
        alms = hp.synalm(cls, lmax=lmax_sim, new=True)
        hp.almxfl(alms[0], beam, inplace=True)
        signal = hp.alm2map(alms, nside=nside, lmax=lmax_sim)[0]
        if add_noise:
            np.random.seed(10000 + i)
            noise = np.random.normal(0.0, sigma_noise, npix)
            sim_maps[:, i] = signal + noise
        else:
            sim_maps[:, i] = signal
    return sim_maps


# ---------------------------------------------------------------------------
# PCL pipeline
# ---------------------------------------------------------------------------
def run_pcl(sim_maps, mask, beam_arr, nside, delta_ell, lmax_sim, cl_tt_full):
    nsims = sim_maps.shape[1]
    nmt_lmax = 3 * nside - 1
    nmt_lmins = np.arange(2, nmt_lmax + 1, delta_ell)
    nmt_lmaxs = np.minimum(nmt_lmins + delta_ell - 1, nmt_lmax)
    b = nmt.NmtBin.from_edges(nmt_lmins, nmt_lmaxs + 1)
    ells_nmt = b.get_effective_ells()

    beam_nmt = beam_arr[: nmt_lmax + 1]
    f0 = nmt.NmtField(mask, [sim_maps[:, 0]], beam=beam_nmt)
    wsp = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(f0, f0, b)

    cl_tt_nmt = np.zeros(nmt_lmax + 1)
    n_use = min(lmax_sim + 1, nmt_lmax + 1)
    cl_tt_nmt[:n_use] = cl_tt_full[:n_use]
    pcl_theory = wsp.decouple_cell(wsp.couple_cell([cl_tt_nmt]))[0]

    t0 = time.perf_counter()
    pcl_spectra = np.zeros((nsims, len(ells_nmt)))
    for i in range(nsims):
        f = nmt.NmtField(mask, [sim_maps[:, i]], beam=beam_nmt)
        cl_coupled = nmt.compute_coupled_cell(f, f)
        pcl_spectra[i, :] = wsp.decouple_cell(cl_coupled)[0]
    t_pcl = time.perf_counter() - t0

    return {
        "ells_all": ells_nmt,
        "spectra": pcl_spectra,
        "windowed_theory_all": pcl_theory,
        "time_s": t_pcl,
    }


# ---------------------------------------------------------------------------
# QML pipeline (returns all three modes)
# ---------------------------------------------------------------------------
def run_qml(
    sim_maps,
    mask,
    sigma_noise,
    raw_cls,
    nside,
    lmax_buffer,
    fwhm_arcmin,
    delta_ell,
    nsims,
    cl_tt_full,
    basis_kwargs,
):
    npix = 12 * nside**2
    cov = np.zeros((npix, npix))
    np.fill_diagonal(cov, sigma_noise**2)

    with tempfile.TemporaryDirectory(prefix="qml_vs_pcl_") as tmpdir:
        cov.tofile(os.path.join(tmpdir, "ncvm.bin"))
        hp.write_map(os.path.join(tmpdir, "mask.fits"), mask, overwrite=True)
        sims_3d = sim_maps.reshape(1, npix, nsims)
        np.save(os.path.join(tmpdir, "sims.npy"), sims_3d)

        cl_table = np.zeros((raw_cls.shape[0], 7))
        cl_table[:, :5] = raw_cls[:, :5]
        cl_file = os.path.join(tmpdir, "dls.txt")
        np.savetxt(cl_file, cl_table, header="ell TT EE BB TE TB EB", fmt="%.16e")

        config = {
            "nside": nside,
            "spins": [0],
            "labels": ["T"],
            "physical_labels": ["T"],
            "do_cross": False,
            "maskfile": os.path.join(tmpdir, "mask.fits"),
            "output_geometry_file": os.path.join(tmpdir, "geometry.dat"),
            "ordering": "RING",
            "inputclfile": cl_file,
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
        fisher = Fisher(config_file, compression=basis_kwargs)
        fisher.set_binning(bins)
        fisher.run()
        t_fisher = time.perf_counter() - t0

        t0 = time.perf_counter()
        spectra = Spectra(config_file, fisher=fisher, compression=basis_kwargs)
        spectra.set_binning(bins)
        spectra.run()
        t_spec = time.perf_counter() - t0

        ells_bin = spectra.get_effective_ells()
        deconv_y = spectra.get_power_spectra(mode="deconvolved")
        decorr_y = spectra.get_power_spectra(mode="decorrelated")
        conv_y, W_conv, conv_theory_func = spectra.get_power_spectra(mode="convolved")

        cov_pred_deconv = spectra.get_covariance(mode="deconvolved")
        cov_pred_decorr = spectra.get_covariance(mode="decorrelated")
        cov_pred_conv = spectra.get_covariance(mode="convolved")

        deconv_th = spectra.convolve_theory_for_inference(cl_tt_full)
        # Convolved-mode theory: <y> = F @ <deconvolved>. The
        # convolve_theory_func returned by get_power_spectra("convolved")
        # operates between bandpower spaces and does not reproduce <y>
        # from per-ell C_ell input; the F-product is the unambiguous
        # closure since cov(y) = F.
        conv_th = cov_pred_conv @ deconv_th

    return {
        "ells_all": ells_bin,
        "deconvolved": {
            "spectra": deconv_y,
            "windowed_theory_all": deconv_th,
            "cov_pred_all": cov_pred_deconv,
        },
        "decorrelated": {"spectra": decorr_y, "cov_pred_all": cov_pred_decorr},
        "convolved": {
            "spectra": conv_y,
            "windowed_theory_all": conv_th,
            "cov_pred_all": cov_pred_conv,
            "window_matrix_all": W_conv,
        },
        "time_fisher_s": t_fisher,
        "time_spectra_s": t_spec,
    }


# ---------------------------------------------------------------------------
# Knox bounds
# ---------------------------------------------------------------------------
def knox_per_ell(cl_tt_full, beam, sigma_noise, npix, lmax_sim, fsky):
    omega_pix = 4.0 * np.pi / npix
    n_white = sigma_noise**2 * omega_pix
    b = beam.copy()
    b[b < 1e-12] = 1e-12
    n_eff = n_white / b**2
    ell = np.arange(lmax_sim + 1)
    var = np.where(
        ell >= 2,
        2.0 * (cl_tt_full + n_eff) ** 2 / np.maximum(2 * ell + 1, 1) / fsky,
        np.inf,
    )
    return var


def knox_per_bin(var_per_ell, bins, ells_eff):
    out = np.zeros(len(ells_eff))
    for bi, ell_eff in enumerate(ells_eff):
        bin_idx = int(np.argmin(np.abs(bins.lbin - ell_eff)))
        lo, hi = int(bins.lmins[bin_idx]), int(bins.lmaxs[bin_idx])
        inv = np.sum(1.0 / var_per_ell[lo : hi + 1])
        out[bi] = np.sqrt(1.0 / inv) if inv > 0 else np.inf
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def empirical_covariance(spectra):
    return np.cov(spectra.T, ddof=1)


def correlation_matrix(cov):
    d = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    return cov / np.outer(d, d)


def chi2_of_mean(mean, theory, cov_emp, nsims):
    """Chi^2 testing whether <estimator> == theory (covariance of the mean)."""
    residual = mean - theory
    cov_mean = cov_emp / nsims
    try:
        inv = np.linalg.inv(cov_mean)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov_mean)
    chi2 = float(residual @ inv @ residual)
    dof = len(residual)
    return chi2, dof, chi2 / dof


def mean_per_sim_chi2(spectra, theory, cov_pred):
    """Mean of {chi^2_i / dof}_i against analytic covariance."""
    nbins = spectra.shape[1]
    try:
        inv = np.linalg.inv(cov_pred)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov_pred)
    residuals = spectra - theory[None, :]
    chi2_i = np.einsum("ij,jk,ik->i", residuals, inv, residuals)
    return float(np.mean(chi2_i / nbins)), float(np.std(chi2_i / nbins))


# ---------------------------------------------------------------------------
# Per-case orchestration
# ---------------------------------------------------------------------------
def analyze_case(
    case,
    sim_maps,
    raw_cls,
    cl_full,
    beam,
    lmax_science,
    lmax_buffer,
    nside,
    npix,
    sigma_noise,
    fwhm_arcmin,
    nsims,
    lmax_sim,
):
    name = case["name"]
    delta_ell = case["delta_ell"]
    print(
        f"\n{'=' * 70}\n  Case: {name} (gal cut {case['gal_cut_deg']} deg, "
        f"delta_ell={delta_ell})\n{'=' * 70}"
    )
    mask, fsky = make_galactic_strip(nside, case["gal_cut_deg"])
    print(f"  fsky = {fsky:.3f}")

    print("\n--- PCL ---")
    pcl = run_pcl(sim_maps, mask, beam, nside, delta_ell, lmax_sim, cl_full["TT"])
    print(f"  PCL time: {pcl['time_s']:.1f}s")

    print(f"\n--- QML (3 modes, basis={case['basis']}) ---")
    qml = run_qml(
        sim_maps,
        mask,
        sigma_noise,
        raw_cls,
        nside,
        lmax_buffer,
        fwhm_arcmin,
        delta_ell,
        nsims,
        cl_full["TT"],
        case["basis"],
    )
    print(
        f"  QML Fisher: {qml['time_fisher_s']:.1f}s, "
        f"Spectra: {qml['time_spectra_s']:.1f}s"
    )

    qml_keep = qml["ells_all"] <= lmax_science
    pcl_keep = (pcl["ells_all"] >= 2) & (pcl["ells_all"] <= lmax_science)
    qml_ells = qml["ells_all"][qml_keep]
    pcl_ells = pcl["ells_all"][pcl_keep]
    nbins_qml = int(qml_keep.sum())
    nbins_pcl = int(pcl_keep.sum())

    pcl_spec = pcl["spectra"][:, pcl_keep]
    pcl_mean = pcl_spec.mean(0)
    pcl_std = pcl_spec.std(0, ddof=1)
    pcl_cov_emp = empirical_covariance(pcl_spec)
    pcl_th = pcl["windowed_theory_all"][pcl_keep]
    pcl_chi2, pcl_dof, pcl_chi2red = chi2_of_mean(pcl_mean, pcl_th, pcl_cov_emp, nsims)

    qml_diag = {}
    for mode in ("deconvolved", "decorrelated", "convolved"):
        m = qml[mode]
        spec = m["spectra"][:, qml_keep]
        mean = spec.mean(0)
        std = spec.std(0, ddof=1)
        cov_emp = empirical_covariance(spec)
        cov_pred = m["cov_pred_all"][np.ix_(qml_keep, qml_keep)]
        d = {
            "mean": mean,
            "std": std,
            "cov_emp": cov_emp,
            "cov_pred": cov_pred,
            "corr_emp": correlation_matrix(cov_emp),
            "corr_pred": correlation_matrix(cov_pred),
        }
        if mode in ("deconvolved", "convolved"):
            th = m["windowed_theory_all"][qml_keep]
            chi2, dof, chi2red = chi2_of_mean(mean, th, cov_emp, nsims)
            ps_mean, ps_std = mean_per_sim_chi2(spec, th, cov_pred)
            d.update(
                {
                    "windowed_theory": th,
                    "chi2_mean": chi2,
                    "dof": dof,
                    "chi2red_mean": chi2red,
                    "per_sim_chi2red_mean": ps_mean,
                    "per_sim_chi2red_std": ps_std,
                }
            )
        else:
            corr = d["corr_emp"]
            offdiag = corr[~np.eye(corr.shape[0], dtype=bool)]
            d.update(
                {
                    "diag_mean": float(np.mean(np.diag(cov_emp))),
                    "diag_std": float(np.std(np.diag(cov_emp))),
                    "offdiag_rms": float(np.sqrt(np.mean(offdiag**2))),
                    "offdiag_max_abs": float(np.max(np.abs(offdiag))),
                }
            )
        qml_diag[mode] = d

    knox_var_part = knox_per_ell(cl_full["TT"], beam, sigma_noise, npix, lmax_sim, fsky)
    knox_var_full = knox_per_ell(cl_full["TT"], beam, sigma_noise, npix, lmax_sim, 1.0)
    bins = Bins.fromdeltal(2, lmax_buffer, delta_ell)
    knox_std_part = knox_per_bin(knox_var_part, bins, qml_ells)
    knox_std_full = knox_per_bin(knox_var_full, bins, qml_ells)

    n_match = min(nbins_qml, nbins_pcl)
    deconv_std = qml_diag["deconvolved"]["std"]
    qml_over_knox_part = deconv_std / knox_std_part
    qml_over_knox_full = deconv_std / knox_std_full
    pcl_over_knox_part = pcl_std[:n_match] / knox_std_part[:n_match]
    pcl_over_knox_full = pcl_std[:n_match] / knox_std_full[:n_match]
    pcl_over_qml = pcl_std[:n_match] / deconv_std[:n_match]

    print("\n  Recovery (chi^2/dof of the sample mean):")
    for mode in ("deconvolved", "convolved"):
        d = qml_diag[mode]
        print(
            f"    QML {mode:11s}: {d['chi2red_mean']:6.2f}  (dof={d['dof']})  "
            f"per-sim mean: {d['per_sim_chi2red_mean']:.2f} "
            f"+/- {d['per_sim_chi2red_std']:.2f}"
        )
    print(f"    PCL              : {pcl_chi2red:6.2f}  (dof={pcl_dof})")
    print("\n  QML decorrelated identity check (target: diag=1, offdiag=0):")
    dec = qml_diag["decorrelated"]
    print(f"    diag mean    = {dec['diag_mean']:.3f}  std = {dec['diag_std']:.3f}")
    print(
        f"    offdiag RMS  = {dec['offdiag_rms']:.3f}  "
        f"max|offdiag| = {dec['offdiag_max_abs']:.3f}"
    )
    print("\n  Variance vs Knox (median over bins):")
    print(f"    sigma_QML / sigma_Knox(fsky)    = {np.median(qml_over_knox_part):.2f}")
    print(f"    sigma_QML / sigma_Knox(fullsky) = {np.median(qml_over_knox_full):.2f}")
    print(f"    sigma_PCL / sigma_Knox(fsky)    = {np.median(pcl_over_knox_part):.2f}")
    print(f"    sigma_PCL / sigma_Knox(fullsky) = {np.median(pcl_over_knox_full):.2f}")
    print(f"    sigma_PCL / sigma_QML           = {np.median(pcl_over_qml):.2f}")

    return {
        "fsky": fsky,
        "gal_cut_deg": case["gal_cut_deg"],
        "label": case["label"],
        "basis": case["basis"],
        "delta_ell": delta_ell,
        "qml_ells": qml_ells,
        "pcl_ells": pcl_ells,
        "qml": qml_diag,
        "pcl": {
            "mean": pcl_mean,
            "std": pcl_std,
            "cov_emp": pcl_cov_emp,
            "corr_emp": correlation_matrix(pcl_cov_emp),
            "windowed_theory": pcl_th,
            "chi2_mean": pcl_chi2,
            "dof": pcl_dof,
            "chi2red_mean": pcl_chi2red,
        },
        "knox": {"partial_sky": knox_std_part, "full_sky": knox_std_full},
        "ratios": {
            "qml_over_knox_partial": qml_over_knox_part,
            "qml_over_knox_full": qml_over_knox_full,
            "pcl_over_knox_partial": pcl_over_knox_part,
            "pcl_over_knox_full": pcl_over_knox_full,
            "pcl_over_qml": pcl_over_qml,
        },
        "timings_s": {
            "pcl": pcl["time_s"],
            "qml_fisher": qml["time_fisher_s"],
            "qml_spectra": qml["time_spectra_s"],
        },
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_dl_variance_figure(results, lmax_science, fname):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.6 * n))
    if n == 1:
        axes = axes[None, :]

    for row, (key, r) in enumerate(results.items()):
        qml_ells = r["qml_ells"]
        pcl_ells = r["pcl_ells"]
        deconv = r["qml"]["deconvolved"]
        dl_q = qml_ells * (qml_ells + 1) / (2 * np.pi)
        dl_p = pcl_ells * (pcl_ells + 1) / (2 * np.pi)

        ax_dl, ax_var = axes[row]

        ax_dl.plot(
            qml_ells,
            deconv["windowed_theory"] * dl_q,
            color=C_QML_DARK,
            ls="--",
            marker="x",
            ms=8,
            mew=1.5,
            label="QML windowed theory",
        )
        ax_dl.plot(
            pcl_ells,
            r["pcl"]["windowed_theory"] * dl_p,
            color=C_PCL_DARK,
            ls="--",
            marker="x",
            ms=8,
            mew=1.5,
            label="PCL windowed theory",
        )
        ax_dl.errorbar(
            qml_ells - 0.5,
            deconv["mean"] * dl_q,
            yerr=deconv["std"] * dl_q,
            fmt="o",
            ms=6,
            capsize=3,
            color=C_QML,
            label="QML estimates",
        )
        ax_dl.errorbar(
            pcl_ells + 0.5,
            r["pcl"]["mean"] * dl_p,
            yerr=r["pcl"]["std"] * dl_p,
            fmt="s",
            ms=6,
            capsize=3,
            color=C_PCL,
            label="PCL estimates",
        )
        ax_dl.set_xlabel(r"Multipole $\ell$")
        ax_dl.set_ylabel(r"$D_\ell^{TT}\;[\mu K^2]$")
        ax_dl.set_title(f"{r['label']}: TT bandpowers")
        ax_dl.legend(loc="upper right")
        ax_dl.set_xlim(0, lmax_science + 2)

        nm = len(r["ratios"]["pcl_over_knox_full"])
        ax_var.axhline(1, color=C_KNOX, ls=":", lw=1.0, label="Knox bound")
        ax_var.plot(
            qml_ells,
            r["ratios"]["qml_over_knox_full"],
            "o-",
            color=C_QML,
            ms=6,
            label=r"$\sigma_{\rm QML}/\sigma_{\rm Knox}^{\rm full}$",
        )
        ax_var.plot(
            pcl_ells[:nm],
            r["ratios"]["pcl_over_knox_full"],
            "s-",
            color=C_PCL,
            ms=6,
            label=r"$\sigma_{\rm PCL}/\sigma_{\rm Knox}^{\rm full}$",
        )
        ax_var.plot(
            qml_ells,
            r["ratios"]["qml_over_knox_partial"],
            "o--",
            color=C_QML_DARK,
            ms=4,
            alpha=0.6,
            label=r"$\sigma_{\rm QML}/\sigma_{\rm Knox}(f_{\rm sky})$",
        )
        ax_var.plot(
            pcl_ells[:nm],
            r["ratios"]["pcl_over_knox_partial"],
            "s--",
            color=C_PCL_DARK,
            ms=4,
            alpha=0.6,
            label=r"$\sigma_{\rm PCL}/\sigma_{\rm Knox}(f_{\rm sky})$",
        )
        ax_var.set_xlabel(r"Multipole $\ell$")
        ax_var.set_ylabel(r"$\sigma / \sigma_{\rm Knox}$")
        med_qf = np.median(r["ratios"]["qml_over_knox_full"])
        med_pf = np.median(r["ratios"]["pcl_over_knox_full"])
        ax_var.set_title(
            f"{r['label']}: median QML/Knox$^{{\\rm full}}$={med_qf:.2f}, "
            f"PCL/Knox$^{{\\rm full}}$={med_pf:.2f}"
        )
        ax_var.legend(loc="upper right", ncol=2)
        ax_var.set_xlim(0, lmax_science + 2)
        ax_var.set_yscale("log")

    fig.savefig(fname)
    print(f"  Wrote {fname}")
    plt.close(fig)


def make_correlation_figure(results, fname):
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.8 * n))
    if n == 1:
        axes = axes[None, :]
    titles = ["QML deconvolved", "QML decorrelated", "PCL", r"Predicted $F^{-1}$"]

    for row, (key, r) in enumerate(results.items()):
        mats = [
            r["qml"]["deconvolved"]["corr_emp"],
            r["qml"]["decorrelated"]["corr_emp"],
            r["pcl"]["corr_emp"],
            correlation_matrix(r["qml"]["deconvolved"]["cov_pred"]),
        ]
        for col, (m, t) in enumerate(zip(mats, titles)):
            ax = axes[row, col]
            im = ax.imshow(m, vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
            ax.set_title(f"{r['label']}\n{t}")
            ax.set_xlabel("bin")
            ax.set_ylabel("bin")
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.savefig(fname)
    print(f"  Wrote {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------
def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_results_json(results, path, config):
    out = {"config": _to_jsonable(config), "cases": _to_jsonable(results)}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    configure_plt()
    print(
        f"Config: nside={NSIDE}, lmax_science={LMAX_SCIENCE}, "
        f"lmax_buffer={LMAX_BUFFER}, nsims={NSIMS}, sigma={SIGMA_NOISE}, "
        f"FWHM={FWHM_ARCMIN:.1f} arcmin, "
        f"add_noise={ADD_NOISE_TO_SIMS}"
    )

    raw_cls, cl_full = load_theory(LMAX_SIM)
    beam = hp.gauss_beam(FWHM_RAD, lmax=LMAX_SIM)
    sim_maps = generate_sims(
        cl_full, beam, NSIDE, NPIX, NSIMS, LMAX_SIM, SIGMA_NOISE, ADD_NOISE_TO_SIMS
    )

    results = {}
    for case in CASES:
        results[case["name"]] = analyze_case(
            case,
            sim_maps,
            raw_cls,
            cl_full,
            beam,
            LMAX_SCIENCE,
            LMAX_BUFFER,
            NSIDE,
            NPIX,
            SIGMA_NOISE,
            FWHM_ARCMIN,
            NSIMS,
            LMAX_SIM,
        )

    print(f"\n{'=' * 70}\n  Outputs\n{'=' * 70}")
    config = {
        "nside": NSIDE,
        "lmax_science": LMAX_SCIENCE,
        "lmax_buffer": LMAX_BUFFER,
        "lmax_sim": LMAX_SIM,
        "nsims": NSIMS,
        "sigma_noise": SIGMA_NOISE,
        "fwhm_arcmin": FWHM_ARCMIN,
        "add_noise_to_sims": ADD_NOISE_TO_SIMS,
    }
    save_results_json(results, "qml_vs_pcl_results.json", config)
    make_dl_variance_figure(results, LMAX_SCIENCE, "qml_vs_pcl_dl_variance.png")
    make_correlation_figure(results, "qml_vs_pcl_correlations.png")


if __name__ == "__main__":
    main()
