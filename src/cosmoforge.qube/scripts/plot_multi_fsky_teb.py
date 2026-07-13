"""TEB QML validation: MC scatter vs Fisher prediction at one sky fraction.

Self-contained script. Builds T (scalar) + Q,U (spin-2) maps at fixed
nside, runs the full QML pipeline with all six TEB spectra
(TT, EE, BB, TE, TB, EB) enabled, and compares per-ell MC scatter against
the Fisher prediction over 1000 simulations.

Setup follows the notebook recipe (cosmocore/outputs/QU_spectra_test_*):
``fisher.lmax_signal = LMAX_BUFFER = LMAX_SIM`` so that signal model,
estimator support, and simulation power all share the same multipole
range — this makes the recovery unbiased without relying on a buffer.

Tensor BB is set to r=0.01 by linear rescaling of a reference r template
from the picslike Fortran-reference dataset. Noise is white at
NOISE_P_UKARCMIN µK·arcmin in polarization (T at P/sqrt(2), the
standard convention).

Diagnostics per spectrum:
  - Per-ell MC mean and standard deviation (NSIMS realisations)
  - Fisher-predicted error bar sqrt(diag(F^-1)) per ell
  - Ratio MC_std / Fisher_std (target 1.0; expected sampling error
    1/sqrt(2N))

Outputs:
  - plot_multi_fsky_teb_results.json
  - plot_multi_fsky_teb.png — 3x2 panel: auto on left (TT/EE/BB, log y),
    cross on right (TE/TB/EB, symlog y)

Run: ``uv run --extra pcl python plot_multi_fsky_teb.py``
"""

import json
import os
import tempfile
import time
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import yaml

from cosmocore import Bins
from qube import Fisher, Spectra

THEORY_CL_FILE = Path(__file__).resolve().parent / "dls.txt"
# Reference r-spectrum file used to extract the tensor BB component
# (linear in r), then rescaled to TENSOR_R below.
PICSLIKE_TENSOR_FILE = (
    Path(__file__).resolve().parents[2]
    / "cosmoforge.picslike/tests/data/nside8/B/fortran_reference/theory_spectra"
    / "fine_dls_r_likelihood_r0.00473684.txt"
)
TENSOR_R = 0.01
TENSOR_R_REF = 0.00473684


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NSIDE = 16
LMAX_SCIENCE = 2 * NSIDE
LMAX_BUFFER = LMAX_SCIENCE  # following the notebook: signal lmax == buffer
LMAX_SIM = LMAX_BUFFER  # truncate sims at the same lmax; clean validation
NSIMS = 1000
NOISE_P_UKARCMIN = 2.5  # polarization sensitivity (LiteBIRD-class)
NOISE_T_UKARCMIN = NOISE_P_UKARCMIN / np.sqrt(2.0)  # standard T = P/sqrt(2)
DELTA_ELL = 1  # per-ell validation; no binning


def ukarcmin_to_pixel_sigma(noise_ukarcmin, nside):
    """Convert white-noise sensitivity in uK*arcmin to per-pixel rms (uK)."""
    npix = 12 * nside**2
    omega_pix_sr = 4.0 * np.pi / npix
    omega_pix_arcmin2 = omega_pix_sr * (10800.0 / np.pi) ** 2
    return noise_ukarcmin / np.sqrt(omega_pix_arcmin2)


SIGMA_T = ukarcmin_to_pixel_sigma(NOISE_T_UKARCMIN, NSIDE)
SIGMA_P = ukarcmin_to_pixel_sigma(NOISE_P_UKARCMIN, NSIDE)

CASES = [
    {"name": "fsky_050", "gal_cut_deg": 30, "label": r"$f_{\rm sky}\sim 0.50$"},
]


def gaussian_fwhm_for_lmax(lmax, beam_at_lmax=0.5):
    sigma = np.sqrt(-2 * np.log(beam_at_lmax) / (lmax * (lmax + 1)))
    return float(np.degrees(sigma * np.sqrt(8 * np.log(2))) * 60)


FWHM_ARCMIN = gaussian_fwhm_for_lmax(LMAX_SCIENCE)
FWHM_RAD = np.radians(FWHM_ARCMIN / 60.0)
NPIX = 12 * NSIDE**2


# ---------------------------------------------------------------------------
# Plot styling (matches QU_spectra_test_*.ipynb)
# ---------------------------------------------------------------------------
def configure_plt():
    plt.rc("axes", labelsize=14, linewidth=1.5)
    plt.rc("xtick", direction="in", labelsize=11, top=True)
    plt.rc("ytick", direction="in", labelsize=11, right=True)
    plt.rc("xtick.major", width=1.1, size=5)
    plt.rc("ytick.major", width=1.1, size=5)
    plt.rc("xtick.minor", width=1.1, size=3)
    plt.rc("ytick.minor", width=1.1, size=3)
    plt.rc("lines", linewidth=2)
    plt.rc("legend", frameon=False, fontsize=10)
    plt.rc("figure", dpi=100, autolayout=True)
    plt.rc("savefig", dpi=200, bbox="tight")


C_MC = "dodgerblue"
C_FISHER = "darkorange"
C_KNOX_FSKY = "gray"
C_KNOX_FULL = "black"
SPECTRA = ["TT", "EE", "BB", "TE", "TB", "EB"]


# ---------------------------------------------------------------------------
# Theory loading
# ---------------------------------------------------------------------------
def load_theory(lmax_sim):
    """Load r=0 spectra; rescale BB to TENSOR_R using the ref-r tensor template.

    BB is linear in r:
        BB(r) = BB_lensing + r * BB_tensor_per_unit_r
    Subtract lensing (r=0) to extract per-unit-r tensor, then rescale to
    TENSOR_R and add lensing back.
    """
    raw_cls = np.loadtxt(THEORY_CL_FILE)  # r=0 (lensing only)
    raw_ref = np.loadtxt(PICSLIKE_TENSOR_FILE)  # r=TENSOR_R_REF

    # Both files share the same ell grid; align on the first column
    ells_r0 = raw_cls[:, 0].astype(int)
    ells_ref = raw_ref[:, 0].astype(int)
    common = np.intersect1d(ells_r0, ells_ref)
    idx_r0 = np.searchsorted(ells_r0, common)
    idx_ref = np.searchsorted(ells_ref, common)

    bb_r0 = raw_cls[idx_r0, 3]
    bb_ref = raw_ref[idx_ref, 3]
    bb_target = bb_r0 + (TENSOR_R / TENSOR_R_REF) * (bb_ref - bb_r0)

    raw_cls = raw_cls.copy()
    raw_cls[idx_r0, 3] = bb_target

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
# Mask
# ---------------------------------------------------------------------------
def make_galactic_strip(nside, gal_cut_deg, n_fields):
    npix = 12 * nside**2
    gal_cut = np.radians(gal_cut_deg)
    galactic_pixels = hp.query_strip(nside, np.pi / 2 - gal_cut, np.pi / 2 + gal_cut)
    mask_one = np.ones(npix)
    mask_one[galactic_pixels] = 0.0
    mask = np.tile(mask_one, (n_fields, 1))
    return mask, float(np.mean(mask_one))


# ---------------------------------------------------------------------------
# Sims (TQU full-sky)
# ---------------------------------------------------------------------------
def generate_tqu_sims(cl_full, beam, nside, npix, nsims, lmax_sim, sigma_t, sigma_p):
    print(f"Generating {nsims} TQU signal+noise full-sky sims...")
    # Three maps per sim: T, Q, U.  Layout: shape (3, npix, nsims).
    sims = np.empty((3, npix, nsims), dtype=np.float64)
    cls = [cl_full["TT"], cl_full["EE"], cl_full["BB"], cl_full["TE"]]
    beam_for_sims = beam[: lmax_sim + 1]
    for i in range(nsims):
        np.random.seed(42 + i)
        alms = hp.synalm(cls, lmax=lmax_sim, new=True)
        for a in alms:
            hp.almxfl(a, beam_for_sims, inplace=True)
        tqu = hp.alm2map(alms, nside=nside, lmax=lmax_sim)  # T, Q, U
        np.random.seed(10000 + i)
        sims[0, :, i] = tqu[0] + np.random.normal(0.0, sigma_t, npix)
        np.random.seed(20000 + i)
        sims[1, :, i] = tqu[1] + np.random.normal(0.0, sigma_p, npix)
        np.random.seed(30000 + i)
        sims[2, :, i] = tqu[2] + np.random.normal(0.0, sigma_p, npix)
    return sims


# ---------------------------------------------------------------------------
# Noise covariance assembly: block-diagonal across {T, Q, U}
# ---------------------------------------------------------------------------
def build_noise_cov(npix, sigma_t, sigma_p):
    """Block-diagonal cov for [T | Q | U] with diagonal pixel noise."""
    n_total = 3 * npix
    cov = np.zeros((n_total, n_total))
    np.fill_diagonal(cov[:npix, :npix], sigma_t**2)
    np.fill_diagonal(cov[npix : 2 * npix, npix : 2 * npix], sigma_p**2)
    np.fill_diagonal(cov[2 * npix :, 2 * npix :], sigma_p**2)
    return cov


# ---------------------------------------------------------------------------
# QML pipeline (TQU, no cross-spectra: 3 spectra TT/EE/BB)
# ---------------------------------------------------------------------------
def run_qml(
    sims_tqu,
    mask,
    raw_cls,
    nside,
    lmax_buffer,
    fwhm_arcmin,
    delta_ell,
    nsims,
    sigma_t,
    sigma_p,
):
    npix = 12 * nside**2
    cov = build_noise_cov(npix, sigma_t, sigma_p)

    with tempfile.TemporaryDirectory(prefix="teb_qml_") as tmpdir:
        cov.tofile(os.path.join(tmpdir, "ncvm.bin"))
        # Mask is multi-component (3 maps); write as TQU mask cube.
        mask_path = os.path.join(tmpdir, "mask.fits")
        hp.write_map(mask_path, list(mask), overwrite=True)
        # Sims: shape (3, npix, nsims)
        np.save(os.path.join(tmpdir, "sims.npy"), sims_tqu)

        cl_table = np.zeros((raw_cls.shape[0], 7))
        cl_table[:, :5] = raw_cls[:, :5]
        cl_file = os.path.join(tmpdir, "dls.txt")
        np.savetxt(cl_file, cl_table, header="ell TT EE BB TE TB EB", fmt="%.16e")

        config = {
            "nside": nside,
            "spins": [0, 2],
            "labels": ["T", "E", "B"],
            "physical_labels": ["T", "Q", "U"],
            "do_cross": False,
            "maskfile": mask_path,
            "output_geometry_file": os.path.join(tmpdir, "geometry.dat"),
            "ordering": "RING",
            "inputclfile": cl_file,
            "input_convention": "Dl",
            "covmatfile1": os.path.join(tmpdir, "ncvm.bin"),
            "covmatfile2": os.path.join(tmpdir, "ncvm.bin"),
            "lmax": lmax_buffer,
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
            "nspectra": 6,
        }
        config_file = os.path.join(tmpdir, "config.yaml")
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        bins = Bins.fromdeltal(2, lmax_buffer, delta_ell)

        t0 = time.perf_counter()
        fisher = Fisher(config_file, compression={"method": "auto", "lmax": lmax_buffer})
        fisher.lmax_signal = lmax_buffer  # notebook recipe: signal == buffer
        fisher.set_binning(bins)
        fisher.run()
        t_fisher = time.perf_counter() - t0

        t0 = time.perf_counter()
        spectra = Spectra(
            config_file,
            fisher=fisher,
            compression={"method": "auto", "lmax": lmax_buffer},
        )
        spectra.set_binning(bins)
        spectra.run()
        t_spec = time.perf_counter() - t0

        ells_bin = spectra.get_effective_ells()
        deconv = spectra.get_power_spectra(mode="deconvolved")
        cov_pred_full = spectra.get_covariance(mode="deconvolved")
        spec_labels = list(fisher.collection.spectra_manager.labels)
        # All TEB cross-spectra: TT/EE/BB are in raw_cls cols; TB/EB are zero.
        cl_per_spectrum = {
            "TT": _load_cl_array(raw_cls, "TT"),
            "EE": _load_cl_array(raw_cls, "EE"),
            "BB": _load_cl_array(raw_cls, "BB"),
            "TE": _load_cl_array(raw_cls, "TE"),
            "TB": np.zeros(int(raw_cls[:, 0].max()) + 1),
            "EB": np.zeros(int(raw_cls[:, 0].max()) + 1),
        }

    return {
        "ells": ells_bin,
        "deconv": deconv,  # shape (nsims, n_params=nspectra*nbins)
        "cov_pred": cov_pred_full,  # shape (n_params, n_params)
        "spec_labels": spec_labels,
        "cl_theory": cl_per_spectrum,
        "time_fisher_s": t_fisher,
        "time_spectra_s": t_spec,
    }


def _load_cl_array(raw_cls, label):
    """Convert dls.txt rows to a per-ell C_ell array starting at ell=0."""
    cols = {"TT": 1, "EE": 2, "BB": 3, "TE": 4}
    ells = raw_cls[:, 0].astype(int)
    dl_to_cl = np.where(ells > 0, 2 * np.pi / (ells * (ells + 1) + 1e-30), 0.0)
    out = np.zeros(int(ells.max()) + 1)
    out[ells] = raw_cls[:, cols[label]] * dl_to_cl
    return out


# ---------------------------------------------------------------------------
# Knox bounds (per spectrum)
# ---------------------------------------------------------------------------
def knox_per_ell(cl_signal, beam, sigma_noise, npix, lmax_sim, fsky):
    omega_pix = 4.0 * np.pi / npix
    n_white = sigma_noise**2 * omega_pix
    n_use = lmax_sim + 1
    b = beam[:n_use].copy()
    b[b < 1e-12] = 1e-12
    n_eff = n_white / b**2
    cl = np.zeros(n_use)
    cl[: min(n_use, len(cl_signal))] = cl_signal[: min(n_use, len(cl_signal))]
    ell = np.arange(n_use)
    var = np.where(
        ell >= 2,
        2.0 * (cl + n_eff) ** 2 / np.maximum(2 * ell + 1, 1) / fsky,
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
# Per-case orchestration
# ---------------------------------------------------------------------------
def analyze_case(
    case,
    sims_tqu,
    raw_cls,
    cl_full,
    beam,
    lmax_science,
    lmax_buffer,
    nside,
    npix,
    fwhm_arcmin,
    delta_ell,
    nsims,
    lmax_sim,
    sigma_t,
    sigma_p,
):
    name = case["name"]
    print(
        f"\n{'=' * 72}\n  Case: {name}  (gal_cut={case['gal_cut_deg']} deg)\n{'=' * 72}"
    )
    mask, fsky = make_galactic_strip(nside, case["gal_cut_deg"], n_fields=3)
    print(f"  fsky = {fsky:.3f}")

    qml = run_qml(
        sims_tqu,
        mask,
        raw_cls,
        nside,
        lmax_buffer,
        fwhm_arcmin,
        delta_ell,
        nsims,
        sigma_t,
        sigma_p,
    )
    print(f"  Fisher: {qml['time_fisher_s']:.1f}s  Spectra: {qml['time_spectra_s']:.1f}s")

    ells_all = qml["ells"]
    keep = ells_all <= lmax_science
    ells = ells_all[keep]
    nbins_full = len(ells_all)

    spec_block = qml["deconv"]  # (nsims, nspectra*nbins_full)
    cov_pred_full = qml["cov_pred"]  # (nspectra*nbins_full,  ditto)
    spec_labels = qml["spec_labels"]
    label_to_idx = {lbl: i for i, lbl in enumerate(spec_labels)}

    out = {
        "fsky": fsky,
        "label": case["label"],
        "ells": ells,
        "spec_labels_internal": spec_labels,
        "spectra": {},
    }

    print(f"  spectra returned in order: {spec_labels}")
    for sname in SPECTRA:
        if sname not in label_to_idx:
            print(f"  {sname}: not in returned labels, skipping")
            continue
        s_idx = label_to_idx[sname]
        s = slice(s_idx * nbins_full, (s_idx + 1) * nbins_full)
        spec_data = spec_block[:, s][:, keep]  # (nsims, nbins_keep)
        cov_pred = cov_pred_full[s, s][np.ix_(keep, keep)]  # (nbins_keep,)^2

        mc_mean = spec_data.mean(axis=0)
        mc_std = spec_data.std(axis=0, ddof=1)
        fisher_std = np.sqrt(np.maximum(np.diag(cov_pred), 0.0))
        cl_th = qml["cl_theory"][sname][2 : int(ells[-1]) + 1]

        out["spectra"][sname] = {
            "mc_mean": mc_mean,
            "mc_std": mc_std,
            "fisher_std": fisher_std,
            "cl_theory_perell": cl_th,
            "ratio_mc_over_fisher": mc_std / np.where(fisher_std > 0, fisher_std, np.inf),
        }
        med_ratio = float(np.median(out["spectra"][sname]["ratio_mc_over_fisher"]))
        print(f"  {sname}: median MC/Fisher = {med_ratio:.3f}")

    out["timings_s"] = {"fisher": qml["time_fisher_s"], "spectra": qml["time_spectra_s"]}
    return out


# ---------------------------------------------------------------------------
# Plotting (3 fskies × 3 spectra)
# ---------------------------------------------------------------------------
def make_figure(results, lmax_science, fname, nsims):
    """Single fsky, six-spectrum (TT/EE/BB/TE/TB/EB) validation plot.

    Layout: 3 rows x 2 cols of D_ell panels, each panel showing per-ell
    theory C_ell line, MC mean +/- MC std markers, and a Fisher
    +/- sqrt((F^-1)_ll) band centered on the MC mean. The annotation
    reports median sigma_MC / sigma_Fisher: the primary validation
    metric (should be 1.0). The 1-sigma sampling uncertainty on this
    ratio with N realisations is 1/sqrt(2N).
    """
    # We expect a single fsky case
    if len(results) != 1:
        raise ValueError(f"This figure layout assumes 1 case, got {len(results)}")
    r = next(iter(results.values()))
    ells = r["ells"]
    dl = ells * (ells + 1) / (2 * np.pi)
    sigma_ratio_uncertainty = 1.0 / np.sqrt(2 * nsims)

    layout = [
        ["TT", "TE"],
        ["EE", "TB"],
        ["BB", "EB"],
    ]
    # Auto spectra positive -> log; cross spectra cross zero -> symlog
    yscale = {
        "TT": "log",
        "EE": "log",
        "BB": "log",
        "TE": "symlog",
        "TB": "symlog",
        "EB": "symlog",
    }
    n_rows = len(layout)
    n_cols = len(layout[0])
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.0 * n_cols, 3.0 * n_rows), sharex=True
    )

    for i in range(n_rows):
        for j in range(n_cols):
            sname = layout[i][j]
            ax = axes[i, j]
            if sname not in r["spectra"]:
                ax.text(
                    0.5,
                    0.5,
                    f"{sname} not available",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            d = r["spectra"][sname]
            mc_mean_dl = d["mc_mean"] * dl
            mc_std_dl = d["mc_std"] * dl
            fisher_std_dl = d["fisher_std"] * dl
            theory_dl = d["cl_theory_perell"][: len(ells)] * dl

            ax.plot(
                ells,
                theory_dl,
                color="black",
                lw=1.2,
                ls="-",
                zorder=2,
                label="Theory" if (i, j) == (0, 0) else None,
            )
            ax.fill_between(
                ells,
                mc_mean_dl - fisher_std_dl,
                mc_mean_dl + fisher_std_dl,
                color=C_FISHER,
                alpha=0.30,
                zorder=1,
                label=r"MC mean $\pm\sqrt{(F^{-1})_{\ell\ell}}$"
                if (i, j) == (0, 0)
                else None,
            )
            ax.errorbar(
                ells,
                mc_mean_dl,
                yerr=mc_std_dl,
                fmt="o",
                ms=3.5,
                capsize=2,
                color=C_MC,
                zorder=3,
                label=r"MC mean $\pm\sigma_{\rm MC}$" if (i, j) == (0, 0) else None,
            )
            ax.set_xlim(0, lmax_science + 2)
            ax.set_ylabel(rf"$D_\ell^{{{sname}}}\;[\mu K^2]$")
            if i == n_rows - 1:
                ax.set_xlabel(r"Multipole $\ell$")

            scale = yscale[sname]
            if scale == "log":
                ax.set_yscale("log")
                pos_theory = theory_dl[theory_dl > 0]
                if len(pos_theory) > 0:
                    ax.set_ylim(0.3 * pos_theory.min(), 3.0 * pos_theory.max())
            else:
                lin = max(float(np.median(np.abs(fisher_std_dl))) * 0.1, 1e-4)
                ax.set_yscale("symlog", linthresh=lin)
                ax.axhline(0, color="gray", lw=0.5, ls=":", alpha=0.7)

            med_ratio = float(np.median(d["ratio_mc_over_fisher"]))
            ax.text(
                0.04,
                0.95,
                rf"$\sigma_{{\rm MC}}/\sigma_{{\rm Fisher}}={med_ratio:.3f}"
                rf"\pm{sigma_ratio_uncertainty:.3f}$",
                transform=ax.transAxes,
                fontsize=10,
                va="top",
            )

    fig.suptitle(
        rf"TEB QML validation, {r['label']}, "
        f"{nsims} sims, per-$\\ell$",
        y=1.02,
        fontsize=13,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {fname}")


# ---------------------------------------------------------------------------
# JSON serialisation
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
        f"lmax_buffer=lmax_signal={LMAX_BUFFER}, lmax_sim={LMAX_SIM}, "
        f"nsims={NSIMS}, sigma_T={SIGMA_T}, sigma_P={SIGMA_P}, "
        f"delta_ell={DELTA_ELL}, FWHM={FWHM_ARCMIN:.1f} arcmin"
    )

    raw_cls, cl_full = load_theory(LMAX_SIM)
    # Beam needed up to 2*lmax_buffer for Fisher internal computations
    beam = hp.gauss_beam(FWHM_RAD, lmax=2 * LMAX_BUFFER + 1)
    sims_tqu = generate_tqu_sims(
        cl_full, beam, NSIDE, NPIX, NSIMS, LMAX_SIM, SIGMA_T, SIGMA_P
    )

    results = {}
    for case in CASES:
        results[case["name"]] = analyze_case(
            case,
            sims_tqu,
            raw_cls,
            cl_full,
            beam,
            LMAX_SCIENCE,
            LMAX_BUFFER,
            NSIDE,
            NPIX,
            FWHM_ARCMIN,
            DELTA_ELL,
            NSIMS,
            LMAX_SIM,
            SIGMA_T,
            SIGMA_P,
        )

    print(f"\n{'=' * 72}\n  Outputs\n{'=' * 72}")
    config = {
        "nside": NSIDE,
        "lmax_science": LMAX_SCIENCE,
        "lmax_buffer": LMAX_BUFFER,
        "lmax_sim": LMAX_SIM,
        "nsims": NSIMS,
        "sigma_T": SIGMA_T,
        "sigma_P": SIGMA_P,
        "delta_ell": DELTA_ELL,
        "fwhm_arcmin": FWHM_ARCMIN,
    }
    save_results_json(results, "plot_multi_fsky_teb_results.json", config)
    make_figure(results, LMAX_SCIENCE, "plot_multi_fsky_teb.png", NSIMS)


if __name__ == "__main__":
    main()
