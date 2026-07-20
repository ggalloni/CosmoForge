"""
End-to-end tests for the Core setup pipeline.

Exercises the realistic call sequence
(setup_fields → setup_geometry → setup_covariance_matrices →
setup_cls → setup_beams → setup_computation_basis) and the
basis-aware wrappers (get_binned_derivative_matrix,
get_total_covariance, get_full_logdet, quadratic_form).
Targets the ``setup_computation_basis`` paths that previous unit
tests bypassed (beam extraction, ``method="auto"`` resolution,
SMW lswitch + ``S_fixed``).
"""

import os
import tempfile
from unittest.mock import patch

import healpy as hp
import numpy as np
import pytest

from cosmocore.bins import Bins
from cosmocore.core import Core
from cosmocore.settings import InputParams
from cosmocore.spectrum_key import SpectrumKey, SpectrumKind


class _PipelineCore(Core):
    """Concrete Core stub required for instantiation in tests."""

    def compute(self):  # pragma: no cover - abstract stub
        return None

    def run(self):  # pragma: no cover - abstract stub
        return None


def _write_cls_file(path, lmax_max=128):
    """Write a small valid Cls text file (Cl convention) covering 0..lmax_max."""
    ell = np.arange(0, lmax_max + 1)
    # Mildly red spectra; magnitudes irrelevant for the pipeline shape tests.
    tt = 1000.0 * np.exp(-((ell / 50.0) ** 2)) + 1.0
    ee = 50.0 * np.exp(-((ell / 50.0) ** 2)) + 0.01
    bb = 1.0 * np.exp(-((ell / 50.0) ** 2)) + 1e-4
    te = 30.0 * np.exp(-((ell / 50.0) ** 2))
    tb = np.zeros_like(ell, dtype=np.float64)
    eb = np.zeros_like(ell, dtype=np.float64)
    data = np.column_stack([ell, tt, ee, bb, te, tb, eb])
    with open(path, "w") as f:
        f.write("# ell TT EE BB TE TB EB\n")
        np.savetxt(f, data, fmt="%.6e")


def _make_params(tmpdir, *, nside, lmax, do_cross=False, params_lmax=None):
    """Build an InputParams with mask + cls file written under tmpdir."""
    params = InputParams()
    params.nside = nside
    # params.lmax controls binning + covariance shape; basis_lmax can be larger.
    params.lmax = params_lmax if params_lmax is not None else lmax
    params.nfields = 1
    params.spins = [0]
    params.labels = ["T"]
    params.ordering = "RING"
    params.do_cross = do_cross
    params.smoothing_type = "gaussian"
    params.fwhmarcmin = 60.0
    params.beam_file = ""
    params.apply_pixwin = False
    params.input_convention = "Cl"

    # Mask: keep ~half the sky.
    mask_file = os.path.join(tmpdir, "mask.fits")
    npix = 12 * nside**2
    mask = np.ones(npix, dtype=np.float64)
    mask[: npix // 2] = 0.0
    hp.write_map(mask_file, mask, overwrite=True)
    params.maskfile = mask_file

    cls_file = os.path.join(tmpdir, "cls.txt")
    _write_cls_file(cls_file, lmax_max=4 * nside)
    params.inputclfile = cls_file

    params.covmatfile1 = os.path.join(tmpdir, "ncvm1.bin")  # mocked in tests
    return params


def _setup_through_basis(params, *, basis_lmax, method, use_smw_optimization):
    """Run setup_fields → ... → setup_computation_basis on a fresh Core."""
    core = _PipelineCore(params)
    core.setup_fields()
    core.setup_geometry()

    n_active = int(core.collection.total_active_pixels)
    mock_cov = np.eye(n_active) * 0.1
    with patch("cosmocore.core.read_covmat", return_value=mock_cov):
        core.setup_covariance_matrices()
    core.setup_cls(lmax=basis_lmax)
    core.setup_beams(lmax=basis_lmax)
    cm = core.setup_computation_basis(
        method=method,
        lmax_signal=basis_lmax,
        use_smw_optimization=use_smw_optimization,
    )
    return core, cm


# =========================================================================
# 1. Compressed pipeline via Core (covers wrapper dispatch + beam extraction)
# =========================================================================


@pytest.mark.parametrize(
    "method,nside,basis_lmax",
    [
        ("harmonic", 8, 16),
        ("auto", 8, 16),  # auto picks harmonic for this geometry
        ("auto", 4, 16),  # auto picks pixel-direct (n_pix < n_modes)
    ],
)
def test_compressed_pipeline_via_core(method, nside, basis_lmax):
    """Build a Core end-to-end with a basis manager, then exercise the wrappers.

    Covers (across the three parametrisations):
      - beam dict extraction + length-truncation in setup_computation_basis
      - method="auto" resolution with the n_pix-vs-n_modes heuristic, both
        outcomes (harmonic and pixel-direct)
      - the pixel-direct lswitch zeroing branch in setup_computation_basis
      - get_binned_derivative_matrix dispatch (both basis-aware paths:
        the pixel-direct fast path and the per-ℓ summed fallback)
      - get_total_covariance / get_covariance_logdet / quadratic_form
        through the basis manager
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        params = _make_params(tmpdir, nside=nside, lmax=basis_lmax)
        core, cm = _setup_through_basis(
            params,
            basis_lmax=basis_lmax,
            method=method,
            use_smw_optimization=False,
        )

        assert cm is not None
        assert core.basis_manager is cm

        # Use a coarse binning so each bin spans several ℓ.
        core.set_binning(Bins.fromdeltal(2, basis_lmax, 4))

        # Binned derivative through the basis-aware dispatch.
        ss_key = SpectrumKey(0, 0, SpectrumKind.SS, spins=(0,))
        beam = np.ones(basis_lmax + 1, dtype=np.float64)
        dC_b = core.get_binned_derivative_matrix(
            bin_idx=1,
            key=ss_key,
            beam_smoothing=beam,
        )
        assert dC_b is not None
        assert np.all(np.isfinite(dC_b))

        # Build a Cls vector (ell=2..basis_lmax) and exercise the C-aware
        # wrappers. For T-only single-spectrum, compressed and uncompressed
        # paths both accept a 1-D array.
        cls = np.ones(basis_lmax + 1, dtype=np.float64) * 1e-3

        cov = core.get_total_covariance(cls)
        assert cov is not None
        assert np.all(np.isfinite(cov))

        logdet = core.get_covariance_logdet(cls)
        assert np.isfinite(logdet)

        # noise_cov1 is dropped after setup_computation_basis (basis owns it).
        # Use the basis's pixel count instead.
        n = core.basis_manager.n_pix
        rng = np.random.default_rng(0)
        d = rng.standard_normal(n)
        q = core.quadratic_form(d, cls)
        assert np.isfinite(q)
        assert q > 0  # positive-definite covariance ⇒ d^T C^{-1} d > 0


# =========================================================================
# 2. SMW inference-window + S_fixed setup branch
# =========================================================================


def test_setup_computation_basis_smw_lswitch():
    """Force the SMW inference-window branch: params.lmax < basis_lmax + fiducial.

    Covers the auto-QML path in setup_computation_basis where ``params.lmax``
    narrows the inference window below ``basis_lmax``; the S_fixed build reads
    the inputclfile, zeroes the inference range, applies the beam, builds
    S_fixed, and restores the original spectra.
    """
    nside = 8
    basis_lmax = 16
    params_lmax = 10  # < basis_lmax → triggers automatic inference-window narrowing

    with tempfile.TemporaryDirectory() as tmpdir:
        params = _make_params(
            tmpdir,
            nside=nside,
            lmax=basis_lmax,
            params_lmax=params_lmax,
        )
        core, cm = _setup_through_basis(
            params,
            basis_lmax=basis_lmax,
            method="harmonic",
            use_smw_optimization=True,
        )
        assert cm is not None
        # Spectra must have been restored after S_fixed build (they were
        # temporarily replaced with the high-ℓ-only fiducial during setup).
        cls_after = core.collection.spectra_manager.get_cls(0, 0, 0)
        assert cls_after is not None
        assert np.all(np.isfinite(cls_after))
        # And the basis must still produce a finite log-determinant (S_fixed
        # was correctly absorbed).
        cls = np.ones(basis_lmax + 1, dtype=np.float64) * 1e-3
        logdet = core.get_covariance_logdet(cls)
        assert np.isfinite(logdet)


# =========================================================================
# 3. output_convention="Dl" folds the bandpower shape weight into the
#    binned derivative (ADR-0019).
# =========================================================================


@pytest.mark.parametrize(
    "method,nside,basis_lmax",
    [
        ("harmonic", 8, 16),  # exercises the Core per-ℓ summed fallback
        ("auto", 4, 16),  # exercises the pixel-direct batched kernel
    ],
)
def test_binned_derivative_dl_shape_weight(method, nside, basis_lmax):
    """Dl convention weights dC^ℓ by w_ℓ = 2π/(ℓ(ℓ+1)) inside the bin.

    Independent of the derivative kernel: builds the per-ℓ Cl derivatives via
    delta_ell=1 bins, sums them with the closed-form shape weight, and checks
    that a wide Dl bin reproduces Σ_{ℓ∈bin} w_ℓ · dC^ℓ. Cl (default) must be
    the plain unit-weight sum.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        params = _make_params(tmpdir, nside=nside, lmax=basis_lmax)
        core, cm = _setup_through_basis(
            params,
            basis_lmax=basis_lmax,
            method=method,
            use_smw_optimization=False,
        )
        ss_key = SpectrumKey(0, 0, SpectrumKind.SS, spins=(0,))
        beam = np.ones(basis_lmax + 1, dtype=np.float64)

        # Per-ℓ Cl derivatives dC^ℓ (each delta_ell=1 bin is a single ℓ).
        core.params.output_convention = "Cl"
        core.set_binning(Bins.fromdeltal(2, basis_lmax, 1))
        dC_ell = {
            int(core.bins.lmins[i]): core.get_binned_derivative_matrix(
                bin_idx=i, key=ss_key, beam_smoothing=beam
            )
            for i in range(core.bins.nbins)
        }

        wide = Bins.fromdeltal(2, basis_lmax, 4)
        w = wide.shape_weights("Dl")
        bin_idx = 1
        lo, hi = int(wide.lmins[bin_idx]), int(wide.lmaxs[bin_idx])
        expected = sum(w[ell] * dC_ell[ell] for ell in range(lo, hi + 1))

        core.params.output_convention = "Dl"
        core.set_binning(wide)
        actual = core.get_binned_derivative_matrix(
            bin_idx=bin_idx, key=ss_key, beam_smoothing=beam
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

        # Cl mode stays the plain unit-weight sum.
        core.params.output_convention = "Cl"
        expected_cl = sum(dC_ell[ell] for ell in range(lo, hi + 1))
        actual_cl = core.get_binned_derivative_matrix(
            bin_idx=bin_idx, key=ss_key, beam_smoothing=beam
        )
        np.testing.assert_allclose(actual_cl, expected_cl, rtol=1e-10, atol=1e-12)


def test_binned_derivative_dl_monopole_bin_raises():
    """A Dl bin including ℓ=0 surfaces the shape-weight guard through the
    derivative wiring (ADR-0019): D_ℓ diverges at the monopole. ℓ=0 is only
    reachable via lmin_floor=0 (foreground/template work)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = _make_params(tmpdir, nside=8, lmax=16)
        core, _ = _setup_through_basis(
            params, basis_lmax=16, method="harmonic", use_smw_optimization=False
        )
        core.params.output_convention = "Dl"
        core.params.lmin = 0  # binning floor must match the inference floor
        core.set_binning(Bins([0], [3], lmin_floor=0))
        ss_key = SpectrumKey(0, 0, SpectrumKind.SS, spins=(0,))
        with pytest.raises(ValueError, match="monopole"):
            core.get_binned_derivative_matrix(bin_idx=0, key=ss_key)
