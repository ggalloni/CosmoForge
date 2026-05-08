"""Compare QML vs pseudo-Cl scatter at two sky fractions (TT, galactic strip).

Paper I §Performance figure: demonstrates the optimality gap between QML and
PCL across moderate-fsky and low-fsky configurations.

Two galactic-strip configurations (TT only, nside=32, lmax=2*nside=64):
  - low fsky  ~ 0.10  (gal cut 64 deg, delta_ell=5, invvar weights)
  - high fsky ~ 0.60  (gal cut 24 deg, delta_ell=1, per-ell)

For each configuration:
  - PCL via NaMaster (deconvolved/decorrelated bandpowers via M^-1)
  - QML via CosmoForge (deconvolved, decorrelated, convolved modes)
  - Sims = CMB signal + diagonal white noise; sigma_pix rescaled from
    a 2 µK·arcmin polarisation sensitivity to the TT-equivalent S/N at
    ell = 50 (i.e. C_TT/C_BB rescaling) so the TT analysis sits in the
    same regime as a polarisation experiment's BB measurement.

Outputs:
  - qml_vs_pcl_results.json   — all numbers (full covariances included)
  - qml_vs_pcl_dl_variance.png — D_ell bandpowers + sigma_PCL/sigma_QML
  - qml_vs_pcl_correlations.png — split-triangle bandpower correlations

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
# 2*nside is the healpix-exact regime: alm2map/map2alm round-trip is
# accurate to numerical precision below this, which keeps PCL bandpower
# means unbiased throughout the science range.
LMAX_SCIENCE = 2 * NSIDE  # = 64
# Band-limit sims to the science range so QUBE (which we run with
# config["lmax"] = lmax_science to avoid Schur-complement variance
# inflation from buffer modes) is not biased by unmodeled signal above
# lmax_science. Both methods then operate on the same multipole range.
LMAX_SIM = LMAX_SCIENCE
NSIMS = 1000
ADD_NOISE_TO_SIMS = True
DELTA_ELL = 5

# Noise: rescale a 2 µK·arcmin polarisation sensitivity (the SO/LiteBIRD-class
# benchmark for BB) to TT, by matching N_ell/C_ell at NOISE_REF_ELL. C_TT >> C_BB
# at large scales, so this gives a much higher TT noise than the polarisation
# experiment's nominal sensitivity — but it places the TT analysis in the same
# S/N regime that BB faces with that experiment, which is the point of the
# Paper I optimality demonstration.
NOISE_SENS_UKARCMIN_POL = 2.0
NOISE_REF_ELL = 50
# SIGMA_NOISE is computed in main() once theory is loaded.

CASES = [
    {
        "name": "low_fsky",
        "gal_cut_deg": 64,
        "label": r"$f_{\rm sky}\sim 0.10$",
        "basis": {"method": "auto"},
        "delta_ell": DELTA_ELL,
        # NaMaster bin weights matched to QUBE's inverse-variance binning,
        # so PCL and QML produce the same bandpower observable.
        "nmt_use_invvar_weights": True,
    },
    {
        "name": "high_fsky",
        "gal_cut_deg": 24,
        "label": r"$f_{\rm sky}\sim 0.60$",
        "basis": {"method": "auto"},
        # Per-ell (no binning): unambiguous apples-to-apples comparison.
        "delta_ell": 1,
        "nmt_use_invvar_weights": False,
    },
]


def gaussian_fwhm_for_lmax(lmax, beam_at_lmax=0.5):
    sigma = np.sqrt(-2 * np.log(beam_at_lmax) / (lmax * (lmax + 1)))
    return float(np.degrees(sigma * np.sqrt(8 * np.log(2))) * 60)


def sigma_noise_tt_matching_bb(cl_full, sens_ukarcmin_pol, ref_ell, n_pix):
    """White-noise sigma_pix [µK] in TT giving the same N_ell/C_ell as a
    polarisation experiment with `sens_ukarcmin_pol` would have on BB at
    multipole `ref_ell`. Rescales by the C_TT(ref_ell)/C_BB(ref_ell) ratio.
    """
    n_pol = (sens_ukarcmin_pol * np.pi / 10800.0) ** 2  # in muK^2 (per sr)
    ratio = float(cl_full["TT"][ref_ell]) / float(cl_full["BB"][ref_ell])
    n_tt = n_pol * ratio
    return float(np.sqrt(n_tt * n_pix / (4.0 * np.pi)))


# Beam tuned so b^2(lmax_science) ~= 0.25 — signal at the top science
# multipole is half-suppressed. No buffer: sims are band-limited to
# lmax_science and QUBE config["lmax"] = lmax_science.
FWHM_ARCMIN = gaussian_fwhm_for_lmax(LMAX_SCIENCE, beam_at_lmax=0.5)
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
def run_pcl(
    sim_maps,
    mask,
    beam_arr,
    nside,
    delta_ell,
    lmax_sim,
    cl_tt_full,
    sigma_noise=0.0,
    npix=None,
    use_invvar_bin=False,
    var_per_ell_for_weights=None,
):
    """PCL pipeline. The expected bandpower includes a noise-bias term
    `N_ell / B**2` because NaMaster does not auto-subtract noise; we add
    it to the theory comparison so that <PCL> can be compared meaningfully.

    If `use_invvar_bin` is True, NaMaster bandpowers are formed with per-
    ell weights proportional to `1/var_per_ell_for_weights` (normalised
    per bin). This matches QUBE's native binning, where the bandpower is
    the inverse-variance-weighted average of per-ell C_ell, so PCL and
    QML produce the *same* bandpower observable rather than two
    different bandpower definitions.
    """
    nsims = sim_maps.shape[1]
    # NaMaster's NmtField requires the beam to span the healpix natural
    # lmax (3*nside-1). The *binning* is independently capped at lmax_sim
    # so we don't try to bin where the sims have no signal.
    nmt_lmax_field = 3 * nside - 1
    nmt_lmax_bin = min(nmt_lmax_field, lmax_sim)
    # Match QUBE's Bins.fromdeltal: emit only full-width bins from lmin=2,
    # dropping any trailing partial bin so the two methods produce the
    # same number of bandpowers on the same ell grid.
    n_full_bins = (nmt_lmax_bin + 1 - 2) // delta_ell
    nmt_lmins = 2 + np.arange(n_full_bins) * delta_ell
    nmt_lmaxs = nmt_lmins + delta_ell - 1
    # Unified NmtBin construction: bpws/weights span the full healpix
    # range so NmtBin.lmax matches the workspace lmax (NaMaster enforces
    # this); ells in (nmt_lmax_bin, nmt_lmax_field] get bpws=-1 and are
    # excluded from any bandpower.
    ells_full = np.arange(nmt_lmax_field + 1)
    bpws = -np.ones(nmt_lmax_field + 1, dtype=int)
    weights = np.zeros(nmt_lmax_field + 1)
    use_iv = use_invvar_bin and delta_ell > 1 and var_per_ell_for_weights is not None
    for bi, (lo, hi) in enumerate(zip(nmt_lmins, nmt_lmaxs)):
        in_bin = (ells_full >= lo) & (ells_full <= hi)
        if use_iv:
            w = 1.0 / var_per_ell_for_weights[lo : hi + 1]
            weights[in_bin] = w / w.sum()
        else:
            n_in_bin = int(hi - lo + 1)
            weights[in_bin] = 1.0 / n_in_bin
        bpws[in_bin] = bi
    b = nmt.NmtBin(
        bpws=bpws[2:],
        ells=ells_full[2:],
        weights=weights[2:],
        lmax=nmt_lmax_field,
    )
    ells_nmt = b.get_effective_ells()

    # Sims use healpy `alm2map` with default `pixwin=False`: signal is
    # point-sampled at pixel centers (no pixel-window suppression), so
    # NaMaster's beam should be the gaussian beam alone. Adding pixwin
    # here would over-deconvolve, biasing PCL bandpowers at high ell.
    beam_nmt = beam_arr[: nmt_lmax_field + 1]
    f0 = nmt.NmtField(mask, [sim_maps[:, 0]], beam=beam_nmt)
    wsp = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(f0, f0, b)

    # Theory cl input padded to the workspace lmax. Signal is zero above
    # lmax_sim (band-limited sims).
    cl_tt_nmt = np.zeros(nmt_lmax_field + 1)
    n_use = min(lmax_sim + 1, nmt_lmax_field + 1)
    cl_tt_nmt[:n_use] = cl_tt_full[:n_use]
    pcl_theory = wsp.decouple_cell(wsp.couple_cell([cl_tt_nmt]))[0]

    # Analytic noise-bias bandpower: white noise added unbeamed to the
    # map gives <decoupled noise>(ell) = N_white / B(ell)^2 at every ell
    # (noise has alm content at all ells, including above lmax_sim,
    # because it lives in pixel space). Use the same couple_cell+decouple
    # round-trip as the theory comparison so the subtraction matches
    # NaMaster's exact bandpower convention.
    noise_bandpower = None
    if npix is not None and sigma_noise > 0.0:
        n_ell_white = sigma_noise**2 * (4.0 * np.pi / npix)
        beam_safe = np.where(beam_nmt > 1e-6, beam_nmt, 1e-6)
        n_ell_decoupled = n_ell_white / beam_safe**2
        noise_bandpower = wsp.decouple_cell(wsp.couple_cell([n_ell_decoupled]))[0]

    t0 = time.perf_counter()
    pcl_spectra = np.zeros((nsims, len(ells_nmt)))
    for i in range(nsims):
        f = nmt.NmtField(mask, [sim_maps[:, i]], beam=beam_nmt)
        cl_coupled = nmt.compute_coupled_cell(f, f)
        pcl_spectra[i, :] = wsp.decouple_cell(cl_coupled)[0]
    if noise_bandpower is not None:
        pcl_spectra -= noise_bandpower
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
    lmax_science,
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
            "lmax": lmax_science,
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

        bins = Bins.fromdeltal(2, lmax_science, delta_ell)

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
    # Beam may be longer than lmax_sim+1 (we compute it at the healpix
    # natural lmax for NaMaster). Slice to match the theory array length.
    b = beam[: lmax_sim + 1].copy()
    b[b < 1e-12] = 1e-12
    n_eff = n_white / b**2
    ell = np.arange(lmax_sim + 1)
    var = np.where(
        ell >= 2,
        2.0 * (cl_tt_full + n_eff) ** 2 / np.maximum(2 * ell + 1, 1) / fsky,
        np.inf,
    )
    return var


def knox_per_bin_invvar(var_per_ell, bins, ells_eff):
    """Inverse-variance combine — the optimal binned Knox, matches QML's
    binned Fisher at full sky (QUBE uses a flat-Cell P_b = sum_ell, giving
    inverse-variance-weighted bandpowers).
    """
    out = np.zeros(len(ells_eff))
    for bi, ell_eff in enumerate(ells_eff):
        bin_idx = int(np.argmin(np.abs(bins.lbin - ell_eff)))
        lo, hi = int(bins.lmins[bin_idx]), int(bins.lmaxs[bin_idx])
        inv = np.sum(1.0 / var_per_ell[lo : hi + 1])
        out[bi] = np.sqrt(1.0 / inv) if inv > 0 else np.inf
    return out


def knox_per_bin_uniform(var_per_ell, bins, ells_eff):
    """Uniform-weight binning — matches NaMaster's default `from_edges`
    bandpower convention, which gives Var(C_b) = (1/Delta_ell^2) sum
    Var(C_ell). Used as the PCL reference; deviates from invvar at low
    ell where Var(ell) varies fast within a bin.
    """
    out = np.zeros(len(ells_eff))
    for bi, ell_eff in enumerate(ells_eff):
        bin_idx = int(np.argmin(np.abs(bins.lbin - ell_eff)))
        lo, hi = int(bins.lmins[bin_idx]), int(bins.lmaxs[bin_idx])
        delta = hi - lo + 1
        var_bin = float(np.sum(var_per_ell[lo : hi + 1])) / (delta * delta)
        out[bi] = np.sqrt(var_bin)
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

    use_invvar_bin = case.get("nmt_use_invvar_weights", False)
    var_per_ell_full = knox_per_ell(cl_full["TT"], beam, sigma_noise, npix, lmax_sim, 1.0)
    print("\n--- PCL ---")
    pcl = run_pcl(
        sim_maps,
        mask,
        beam,
        nside,
        delta_ell,
        lmax_sim,
        cl_full["TT"],
        sigma_noise=sigma_noise,
        npix=npix,
        use_invvar_bin=use_invvar_bin,
        var_per_ell_for_weights=var_per_ell_full if use_invvar_bin else None,
    )
    bin_label = (
        "invvar (matches QUBE)" if use_invvar_bin else "uniform (NaMaster default)"
    )
    print(
        f"  PCL time: {pcl['time_s']:.1f}s, binning: {bin_label}, delta_ell={delta_ell}"
    )

    print(f"\n--- QML (3 modes, basis={case['basis']}) ---")
    qml = run_qml(
        sim_maps,
        mask,
        sigma_noise,
        raw_cls,
        nside,
        lmax_science,
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

    # Both methods now produce inverse-variance bandpowers (QML natively;
    # PCL via custom NaMaster weights when use_invvar_bin=True; or per-ell
    # when delta_ell=1). Use a single invvar Knox reference, with the
    # same science-range binning as QML's Fisher.
    knox_var_part = knox_per_ell(cl_full["TT"], beam, sigma_noise, npix, lmax_sim, fsky)
    bins = Bins.fromdeltal(2, lmax_science, delta_ell)
    knox_std_part = knox_per_bin_invvar(knox_var_part, bins, qml_ells)
    knox_std_full = knox_per_bin_invvar(var_per_ell_full, bins, qml_ells)

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
    """Two-panel layout (one row): bandpowers + sigma_PCL/sigma_QML ratio.

    Both fsky cases overlaid in each panel: color encodes fsky (low -> blue
    family, high -> orange family); marker shape distinguishes QML (circle)
    vs PCL (square) where applicable.
    """
    fig, (ax_dl, ax_ratio) = plt.subplots(1, 2, figsize=(13, 4.6))

    case_colors = {
        "low_fsky": ("#1f5fae", "#5fa1d8"),  # QML, PCL shades for low fsky
        "high_fsky": ("#cc6a05", "#f0a55a"),  # QML, PCL shades for high fsky
    }
    # Vertical offsets in D_ell to separate the two fsky cases visually.
    # We shift the binned (low-fsky) lane *up* rather than the unbinned
    # (high-fsky) lane down, so all displayed values stay positive.
    # Offsets are disclosed in the legend.
    case_offsets = {"low_fsky": +2500.0, "high_fsky": 0.0}

    for key, r in results.items():
        qml_ells = np.asarray(r["qml_ells"])
        pcl_ells = np.asarray(r["pcl_ells"])
        deconv = r["qml"]["deconvolved"]
        # Use NaMaster's invvar-weighted ell for both methods. QUBE reports
        # the bin midpoint, but its bandpower observable is also an invvar
        # average, so the invvar-weighted ell is the physically correct
        # "effective ell" for both. (QUBE's get_effective_ells() reporting
        # is being addressed in a separate issue.)
        n_q = min(len(qml_ells), len(pcl_ells))
        ell_eff = pcl_ells[:n_q]
        dl = ell_eff * (ell_eff + 1) / (2 * np.pi)
        c_qml, c_pcl = case_colors.get(key, (C_QML, C_PCL))
        offset = case_offsets.get(key, 0.0)
        offset_str = "" if offset == 0.0 else f" $({offset:+g}$ µK$^2)$"

        # Theory line (we use QML's windowed theory; the two methods'
        # windowed theories differ slightly at low ell due to mode-coupling
        # treatment but the invvar-weighted means align well).
        ax_dl.plot(
            ell_eff,
            deconv["windowed_theory"][:n_q] * dl + offset,
            color=c_qml,
            ls="--",
            lw=1.5,
            alpha=0.7,
            label=f"{r['label']}: theory{offset_str}",
        )
        ax_dl.errorbar(
            ell_eff - 0.25,
            deconv["mean"][:n_q] * dl + offset,
            yerr=deconv["std"][:n_q] * dl,
            fmt="o",
            ms=5,
            capsize=2.5,
            color=c_qml,
            label=f"{r['label']}: QML",
        )
        ax_dl.errorbar(
            ell_eff + 0.25,
            r["pcl"]["mean"][:n_q] * dl + offset,
            yerr=r["pcl"]["std"][:n_q] * dl,
            fmt="s",
            ms=5,
            capsize=2.5,
            color=c_pcl,
            label=f"{r['label']}: PCL",
        )

        # Right panel: sigma_PCL / sigma_QML ratio (no offset — physical ratio)
        ratio = r["pcl"]["std"][:n_q] / deconv["std"][:n_q]
        ax_ratio.plot(
            ell_eff,
            ratio,
            "o-",
            color=c_qml,
            ms=5,
            lw=1.5,
            label=r["label"],
        )

    ax_dl.set_xlabel(r"Multipole $\ell$")
    ax_dl.set_ylabel(r"$D_\ell^{TT}\;[\mu K^2]$ (with offsets)")
    ax_dl.legend(loc="upper left", fontsize=11, ncol=2, framealpha=0.95)
    ax_dl.set_xlim(0, lmax_science + 2)
    # Add headroom at the top so the legend doesn't overlap data.
    ymin, ymax = ax_dl.get_ylim()
    ax_dl.set_ylim(ymin, ymax + 0.45 * (ymax - ymin))

    ax_ratio.axhline(1.0, color="black", ls=":", lw=1.0, label="QML = PCL")
    ax_ratio.set_xlabel(r"Multipole $\ell$")
    ax_ratio.set_ylabel(r"$\sigma_{\rm PCL} / \sigma_{\rm QML}$")
    ax_ratio.legend(loc="upper right", fontsize=12)
    ax_ratio.set_xlim(0, lmax_science + 2)

    fig.savefig(fname)
    print(f"  Wrote {fname}")
    plt.close(fig)


def make_correlation_figure(results, fname):
    """Split-triangle bandpower correlation per fsky case.

    Upper triangle = QML deconvolved (F^-1 q), lower triangle = PCL — both
    estimators of the same bandpower observable, so the comparison is
    apples-to-apples. A narrow band around the diagonal is left as NaN to
    render a clear divider between the two triangle halves; its half-width is
    controlled by ``gap = max(1, nbin // 40)``.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 5.6), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (key, r) in zip(axes, results.items()):
        qml = np.asarray(r["qml"]["deconvolved"]["corr_emp"])
        pcl = np.asarray(r["pcl"]["corr_emp"])
        nbin = min(qml.shape[0], pcl.shape[0])
        qml = qml[:nbin, :nbin]
        pcl = pcl[:nbin, :nbin]

        gap = max(1, nbin // 40)
        composite = np.full_like(qml, np.nan, dtype=float)
        iu = np.triu_indices(nbin, k=gap)
        il = np.tril_indices(nbin, k=-gap)
        # origin="lower": tril displays top-left (where the QML label sits),
        # triu displays bottom-right (where the PCL label sits).
        composite[il] = qml[il]
        composite[iu] = pcl[iu]

        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad(color="white")
        im = ax.imshow(composite, vmin=-1, vmax=1, cmap=cmap, origin="lower")
        ax.plot(
            [-0.5, nbin - 0.5],
            [-0.5, nbin - 0.5],
            color="black",
            lw=1.4,
            solid_capstyle="butt",
        )
        ax.set_xlim(-0.5, nbin - 0.5)
        ax.set_ylim(-0.5, nbin - 0.5)
        ax.set_xlabel("Bandpower")
        ax.set_ylabel("Bandpower")
        ax.set_title(r["label"], pad=10)

        ax.text(
            0.97,
            0.04,
            "PCL",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="black",
            bbox=dict(
                facecolor="white",
                alpha=0.9,
                edgecolor="black",
                boxstyle="round,pad=0.3",
            ),
        )
        ax.text(
            0.04,
            0.96,
            "QML decoupled",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="black",
            bbox=dict(
                facecolor="white",
                alpha=0.9,
                edgecolor="black",
                boxstyle="round,pad=0.3",
            ),
        )

    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Correlation")
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
    raw_cls, cl_full = load_theory(LMAX_SIM)
    sigma_noise = sigma_noise_tt_matching_bb(
        cl_full, NOISE_SENS_UKARCMIN_POL, NOISE_REF_ELL, NPIX
    )
    print(
        f"Config: nside={NSIDE}, lmax_science={LMAX_SCIENCE}, "
        f"lmax_sim={LMAX_SIM}, nsims={NSIMS}, sigma={sigma_noise:.4f} muK "
        f"(equiv {NOISE_SENS_UKARCMIN_POL} muK*arcmin pol on BB at ell={NOISE_REF_ELL}), "
        f"FWHM={FWHM_ARCMIN:.1f} arcmin, "
        f"add_noise={ADD_NOISE_TO_SIMS}"
    )

    # NaMaster's NmtField requires the beam to extend to at least 3*nside-1
    # (the healpix natural lmax). Compute it long enough; downstream consumers
    # slice down to whatever range they need.
    beam = hp.gauss_beam(FWHM_RAD, lmax=max(LMAX_SIM, 3 * NSIDE - 1))
    sim_maps = generate_sims(
        cl_full, beam, NSIDE, NPIX, NSIMS, LMAX_SIM, sigma_noise, ADD_NOISE_TO_SIMS
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
            NSIDE,
            NPIX,
            sigma_noise,
            FWHM_ARCMIN,
            NSIMS,
            LMAX_SIM,
        )

    print(f"\n{'=' * 70}\n  Outputs\n{'=' * 70}")
    config = {
        "nside": NSIDE,
        "lmax_science": LMAX_SCIENCE,
        "lmax_sim": LMAX_SIM,
        "nsims": NSIMS,
        "sigma_noise": sigma_noise,
        "noise_sens_ukarcmin_pol_reference": NOISE_SENS_UKARCMIN_POL,
        "noise_ref_ell": NOISE_REF_ELL,
        "fwhm_arcmin": FWHM_ARCMIN,
        "delta_ell": DELTA_ELL,
        "add_noise_to_sims": ADD_NOISE_TO_SIMS,
    }
    save_results_json(results, "qml_vs_pcl_results.json", config)
    make_dl_variance_figure(results, LMAX_SCIENCE, "qml_vs_pcl_dl_variance.png")
    make_correlation_figure(results, "qml_vs_pcl_correlations.png")


if __name__ == "__main__":
    main()
