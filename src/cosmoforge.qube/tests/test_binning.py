"""
Tests for QML binning support.

Test 1: Identity — delta_ell=1 reproduces unbinned (covered by existing tests)
Test 2: Linearity — native binned Fisher == P @ F_unbinned @ P^T (pixel-space)
Test 3: Linearity — same check with compression
Test 4: Shape tests
Test 5: Normalization modes with binning
"""

import os

import numpy as np
import pytest

from cosmocore import Bins
from qube import Fisher, Spectra

# =============================================================================
# Helpers
# =============================================================================

_cache: dict[str, dict] = {}


def _resolve_config_with_overrides(fields, config_resolver, overrides=None):
    """Resolve config file and apply YAML-level overrides."""
    import tempfile

    import yaml

    config_file = config_resolver(f"tests/data/nside4/{fields}/config.yaml")
    if overrides:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        config.update(overrides)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config, tmp, default_flow_style=False)
        tmp.close()
        os.unlink(config_file)
        return tmp.name
    return config_file


def _run_fisher_with_bins(
    fields: str,
    config_resolver,
    bins: Bins | None = None,
    basis=False,
    config_overrides: dict | None = None,
) -> Fisher:
    """Run Fisher pipeline with optional binning and computation basis.

    ``basis`` defaults to ``False`` (the traditional path) so these binning
    linearity checks keep their pre-auto-default behaviour.
    """
    config_file = _resolve_config_with_overrides(
        fields, config_resolver, config_overrides
    )
    fisher = Fisher(config_file, basis=basis)
    if bins is not None:
        fisher.set_binning(bins)
    fisher.run()
    os.unlink(config_file)
    return fisher


def _run_spectra_with_bins(
    fields: str,
    config_resolver,
    bins: Bins | None = None,
    basis=False,
    fisher: Fisher | None = None,
) -> Spectra:
    """Run Spectra pipeline with optional binning and computation basis."""
    config_file = config_resolver(f"tests/data/nside4/{fields}/config.yaml")
    qml = Spectra(config_file, fisher=fisher, basis=basis)
    if bins is not None:
        qml.set_binning(bins)
    qml.run()
    os.unlink(config_file)
    return qml


def _get_cached(key, runner):
    """Cache runner results."""
    if key not in _cache:
        _cache[key] = runner()
    return _cache[key]


# =============================================================================
# Test 2: Linearity — native binning equals post-hoc binning (pixel-space)
# =============================================================================


class TestLinearityPixelSpace:
    """Native binned Fisher/q should equal post-hoc binned unbinned results."""

    @pytest.fixture
    def lmax(self):
        return 8  # nside=4 config

    @pytest.fixture
    def delta_ell(self):
        return 2

    @pytest.fixture
    def bins(self, lmax, delta_ell):
        # Bins cover ells 2..7 (3 bins of width 2). Ell 8 is excluded
        # since (8-2+1)//2 = 3 bins covering [2,3],[4,5],[6,7].
        return Bins.fromdeltal(2, lmax, delta_ell)

    @pytest.fixture
    def unbinned_fisher(self, config_resolver):
        """Unbinned (delta_ell=1) Fisher — default."""

        def runner():
            f = _run_fisher_with_bins("T", config_resolver)
            return {"fisher": f.get_fisher_matrix(), "instance": f}

        return _get_cached("unbinned_fisher_T", runner)

    @pytest.fixture
    def binned_fisher(self, config_resolver, bins):
        """Natively binned Fisher."""

        def runner():
            f = _run_fisher_with_bins("T", config_resolver, bins=bins)
            return {"fisher": f.get_fisher_matrix(), "instance": f}

        return _get_cached("binned_fisher_T", runner)

    def _make_Q_ell(self, bins, n_ell):
        """Build (nbins, n_ell) sum matrix mapping Fisher ell indices to bins.

        Unlike the bin-averaging operator P (weight 1/dl), Q uses weight 1
        so that QML derivatives are summed, not averaged. This ensures
        F^{-1} q returns the bin-averaged spectrum without a spurious dl factor.
        """
        Q = np.zeros((bins.nbins, n_ell))
        for b, (a, z) in enumerate(zip(bins.lmins, bins.lmaxs)):
            Q[b, a - 2 : z - 2 + 1] = 1.0
        return Q

    def test_fisher_linearity(self, unbinned_fisher, binned_fisher, bins):
        """F_binned == Q @ F_unbinned @ Q^T (exact by linearity of trace)."""
        F_unbinned = unbinned_fisher["fisher"]
        F_binned = binned_fisher["fisher"]

        n_ell = F_unbinned.shape[0]
        Q_ell = self._make_Q_ell(bins, n_ell)

        F_posthoc = Q_ell @ F_unbinned @ Q_ell.T

        np.testing.assert_allclose(
            F_binned,
            F_posthoc,
            rtol=1e-6,
            err_msg="Native binned Fisher does not match post-hoc binned Fisher",
        )

    def test_fisher_shape(self, binned_fisher, bins):
        """Binned Fisher has shape (nbins, nbins)."""
        F = binned_fisher["fisher"]
        assert F.shape == (bins.nbins, bins.nbins)

    def test_fisher_symmetric(self, binned_fisher):
        """Binned Fisher is symmetric."""
        F = binned_fisher["fisher"]
        np.testing.assert_allclose(F, F.T, atol=1e-14)

    def test_fisher_positive_eigenvalues(self, binned_fisher):
        """Binned Fisher has positive eigenvalues."""
        F = binned_fisher["fisher"]
        eigvals = np.linalg.eigvalsh(F)
        assert np.all(eigvals > 0), f"Negative eigenvalue: {eigvals.min()}"

    def test_qml_linearity(self, config_resolver, bins):
        """q_binned == Q @ q_unbinned (exact by linearity)."""
        # Unbinned spectra
        f_unb = _run_fisher_with_bins("T", config_resolver)
        s_unb = _run_spectra_with_bins("T", config_resolver, fisher=f_unb)
        q_unbinned = s_unb.qml_results  # (nsims, n_ell)

        # Binned spectra
        f_bin = _run_fisher_with_bins("T", config_resolver, bins=bins)
        s_bin = _run_spectra_with_bins("T", config_resolver, bins=bins, fisher=f_bin)
        q_binned = s_bin.qml_results  # (nsims, nbins)

        n_ell = q_unbinned.shape[1]
        Q_ell = self._make_Q_ell(bins, n_ell)

        q_posthoc = q_unbinned @ Q_ell.T  # (nsims, nbins)

        np.testing.assert_allclose(
            q_binned,
            q_posthoc,
            rtol=1e-10,
            err_msg="Native binned q does not match post-hoc binned q",
        )


# =============================================================================
# Test 3: Linearity with compression
# =============================================================================


class TestLinearityCompressed:
    """Same linearity checks with harmonic compression enabled."""

    @pytest.fixture
    def lmax(self):
        return 8

    @pytest.fixture
    def delta_ell(self):
        return 2

    @pytest.fixture
    def bins(self, lmax, delta_ell):
        return Bins.fromdeltal(2, lmax, delta_ell)

    @pytest.fixture
    def compression(self):
        return {"method": "harmonic"}

    def _make_Q_ell(self, bins, n_ell):
        Q = np.zeros((bins.nbins, n_ell))
        for b, (a, z) in enumerate(zip(bins.lmins, bins.lmaxs)):
            Q[b, a - 2 : z - 2 + 1] = 1.0
        return Q

    def test_fisher_linearity_compressed(self, config_resolver, bins, compression):
        """Compressed: F_binned == Q @ F_unbinned @ Q^T."""
        f_unb = _run_fisher_with_bins("T", config_resolver, basis=compression)
        F_unbinned = f_unb.get_fisher_matrix()

        f_bin = _run_fisher_with_bins("T", config_resolver, bins=bins, basis=compression)
        F_binned = f_bin.get_fisher_matrix()

        n_ell = F_unbinned.shape[0]
        Q_ell = self._make_Q_ell(bins, n_ell)
        F_posthoc = Q_ell @ F_unbinned @ Q_ell.T

        np.testing.assert_allclose(
            F_binned,
            F_posthoc,
            rtol=1e-10,
            err_msg="Compressed binned Fisher does not match post-hoc binned",
        )

    def test_qml_linearity_compressed(self, config_resolver, bins, compression):
        """Compressed: q_binned == Q @ q_unbinned."""
        f_unb = _run_fisher_with_bins("T", config_resolver, basis=compression)
        s_unb = _run_spectra_with_bins(
            "T", config_resolver, basis=compression, fisher=f_unb
        )
        q_unbinned = s_unb.qml_results

        f_bin = _run_fisher_with_bins("T", config_resolver, bins=bins, basis=compression)
        s_bin = _run_spectra_with_bins(
            "T", config_resolver, bins=bins, basis=compression, fisher=f_bin
        )
        q_binned = s_bin.qml_results

        n_ell = q_unbinned.shape[1]
        Q_ell = self._make_Q_ell(bins, n_ell)
        q_posthoc = q_unbinned @ Q_ell.T

        np.testing.assert_allclose(
            q_binned,
            q_posthoc,
            rtol=1e-10,
            err_msg="Compressed binned q does not match post-hoc binned",
        )


# =============================================================================
# Test: Multi-spectrum (TQU) linearity
# =============================================================================


class TestMultiSpectrumBinning:
    """Binning works correctly for multi-spectrum (TQU) case."""

    @pytest.fixture
    def bins(self):
        return Bins.fromdeltal(2, 8, 2)

    def test_tqu_fisher_shape(self, config_resolver, bins):
        """TQU binned Fisher has correct shape."""
        f = _run_fisher_with_bins("TQU", config_resolver, bins=bins)
        F = f.get_fisher_matrix()
        nspectra = f.params.nspectra
        assert F.shape == (nspectra * bins.nbins, nspectra * bins.nbins)

    def test_tqu_fisher_symmetric_positive(self, config_resolver, bins):
        """TQU binned Fisher is symmetric and positive semi-definite."""
        f = _run_fisher_with_bins("TQU", config_resolver, bins=bins)
        F = f.get_fisher_matrix()
        np.testing.assert_allclose(F, F.T, atol=1e-14)
        eigvals = np.linalg.eigvalsh(F)
        assert np.all(eigvals > -1e-10)

    def test_tqu_spectra_shape(self, config_resolver, bins):
        """TQU binned spectra have correct shape."""
        f = _run_fisher_with_bins("TQU", config_resolver, bins=bins)
        s = _run_spectra_with_bins("TQU", config_resolver, bins=bins, fisher=f)
        power = s.get_power_spectra()
        nspectra = f.params.nspectra
        assert power.shape[1] == nspectra * bins.nbins

    def test_tqu_spectra_finite(self, config_resolver, bins):
        """TQU binned spectra are finite."""
        f = _run_fisher_with_bins("TQU", config_resolver, bins=bins)
        s = _run_spectra_with_bins("TQU", config_resolver, bins=bins, fisher=f)
        power = s.get_power_spectra()
        assert np.all(np.isfinite(power))


# =============================================================================
# Test 4: Shape tests
# =============================================================================


class TestBinnedShapes:
    """Verify output dimensions with various bin widths."""

    @pytest.mark.parametrize("delta_ell", [2, 3])
    def test_fisher_shape(self, config_resolver, delta_ell):
        lmax = 8
        bins = Bins.fromdeltal(2, lmax, delta_ell)
        f = _run_fisher_with_bins("T", config_resolver, bins=bins)
        F = f.get_fisher_matrix()
        assert F.shape == (bins.nbins, bins.nbins)

    @pytest.mark.parametrize("delta_ell", [2, 3])
    def test_spectra_shape(self, config_resolver, delta_ell):
        lmax = 8
        bins = Bins.fromdeltal(2, lmax, delta_ell)
        f = _run_fisher_with_bins("T", config_resolver, bins=bins)
        s = _run_spectra_with_bins("T", config_resolver, bins=bins, fisher=f)
        power = s.get_power_spectra()
        assert power.shape[1] == bins.nbins

    @pytest.mark.parametrize("delta_ell", [2, 3])
    def test_covariance_shape(self, config_resolver, delta_ell):
        lmax = 8
        bins = Bins.fromdeltal(2, lmax, delta_ell)
        f = _run_fisher_with_bins("T", config_resolver, bins=bins)
        s = _run_spectra_with_bins("T", config_resolver, bins=bins, fisher=f)
        cov = s.get_covariance()
        assert cov.shape == (bins.nbins, bins.nbins)

    def test_effective_ells(self):
        """Effective ells match bin midpoints."""
        bins = Bins.fromdeltal(2, 7, 2)
        expected = np.array([2.5, 4.5, 6.5])  # midpoints of [2,3], [4,5], [6,7]
        np.testing.assert_array_equal(bins.lbin, expected)


# =============================================================================
# Test 5: Normalization modes with binning
# =============================================================================


class TestNormalizationModesBinned:
    """All three normalization modes work with binned quantities."""

    @pytest.fixture
    def binned_spectra(self, config_resolver):
        bins = Bins.fromdeltal(2, 8, 2)
        f = _run_fisher_with_bins("T", config_resolver, bins=bins)
        s = _run_spectra_with_bins("T", config_resolver, bins=bins, fisher=f)
        return s, bins

    def test_deconvolved_shape(self, binned_spectra):
        s, bins = binned_spectra
        power = s.get_power_spectra(mode="deconvolved")
        assert power.shape[1] == bins.nbins

    def test_decorrelated_shape(self, binned_spectra):
        s, bins = binned_spectra
        power = s.get_power_spectra(mode="decorrelated")
        assert power.shape[1] == bins.nbins

    def test_convolved_shape(self, binned_spectra):
        s, bins = binned_spectra
        y, W, convolve = s.get_power_spectra(mode="convolved")
        assert y.shape[1] == bins.nbins
        assert W.shape == (bins.nbins, bins.nbins)

    def test_decorrelated_covariance_is_identity(self, binned_spectra):
        s, bins = binned_spectra
        cov = s.get_covariance(mode="decorrelated")
        np.testing.assert_allclose(
            cov,
            np.eye(bins.nbins),
            atol=1e-10,
            err_msg="Decorrelated covariance should be identity",
        )

    def test_deconvolved_covariance_shape(self, binned_spectra):
        s, bins = binned_spectra
        cov = s.get_covariance(mode="deconvolved")
        assert cov.shape == (bins.nbins, bins.nbins)

    def test_error_bars_finite(self, binned_spectra):
        """Binned error bars should be finite and positive."""
        s, bins = binned_spectra
        errors = s.get_error_bars()
        assert errors.shape == (bins.nbins,)
        assert np.all(np.isfinite(errors))
        assert np.all(errors > 0)


# =============================================================================
# Bandpower window function (for parameter inference)
# =============================================================================


class TestBandpowerWindow:
    """Bin-to-ell bandpower window for binned QML inference."""

    @pytest.fixture
    def setup(self, config_resolver):
        bins = Bins.fromdeltal(2, 8, 2)
        f = _run_fisher_with_bins("T", config_resolver, bins=bins)
        s = _run_spectra_with_bins("T", config_resolver, bins=bins, fisher=f)
        return f, s, bins

    def test_window_shape(self, setup):
        f, _, bins = setup
        W = f.get_bandpower_window_function()
        n_ell = f.params.lmax - 1
        assert W.shape == (bins.nbins, n_ell)

    def test_window_consistency_with_perell_fisher(self, setup):
        """W = F_b^-1 @ Q @ F_perell."""
        f, _, bins = setup
        W = f.get_bandpower_window_function()
        F_b = f.get_fisher_matrix()
        F_pe = f._compute_per_ell_fisher()
        n_ell = f.params.lmax - 1
        Q = np.zeros((bins.nbins, n_ell))
        for b, (lo, hi) in enumerate(zip(bins.lmins, bins.lmaxs)):
            Q[b, lo - 2 : hi - 2 + 1] = 1.0
        W_expected = np.linalg.solve(F_b, Q @ F_pe)
        np.testing.assert_allclose(W, W_expected, rtol=1e-10)

    def test_window_cached(self, setup):
        """Second call returns the same matrix without recomputation."""
        f, _, _ = setup
        W1 = f.get_bandpower_window_function()
        W2 = f.get_bandpower_window_function()
        assert W1 is W2

    def test_convolve_theory_matches_window_product(self, setup):
        """convolve_theory_for_inference applies the window correctly."""
        f, s, _ = setup
        W = f.get_bandpower_window_function()
        cl_test = 1.0 / (np.arange(2, f.params.lmax + 1) ** 2)
        mu = s.convolve_theory_for_inference(cl_test)
        np.testing.assert_allclose(mu, W @ cl_test, rtol=1e-12)

    def test_convolve_theory_input_formats(self, setup):
        """Both ell=2..lmax and ell=0..lmax input formats give same result."""
        _, s, _ = setup
        cl_short = np.arange(2.0, s.params.lmax + 1) ** -2
        cl_full = np.zeros(s.params.lmax + 1)
        cl_full[2:] = cl_short
        mu_short = s.convolve_theory_for_inference(cl_short)
        mu_full = s.convolve_theory_for_inference(cl_full)
        np.testing.assert_allclose(mu_short, mu_full, rtol=1e-12)

    def test_convolve_theory_invalid_length_raises(self, setup):
        """Wrong length raises ValueError."""
        _, s, _ = setup
        with pytest.raises(ValueError, match="length"):
            s.convolve_theory_for_inference(np.ones(3))

    def test_window_default_delta_ell_one_is_identity(self, config_resolver):
        """Without explicit binning (delta_ell=1 default) W is identity."""
        f = _run_fisher_with_bins("T", config_resolver)
        W = f.get_bandpower_window_function()
        n_ell = f.params.lmax - 1
        assert W.shape == (n_ell, n_ell)
        np.testing.assert_allclose(W, np.eye(n_ell), atol=1e-10)


# =============================================================================
# Config-based binning (delta_ell, custom lmins/lmaxs)
# =============================================================================


class TestConfigBinning:
    """Binning configured via YAML matches Python API."""

    def test_delta_ell_config_matches_api(self, config_resolver):
        """delta_ell in config gives same Fisher as set_binning()."""
        bins = Bins.fromdeltal(2, 8, 2)
        f_api = _run_fisher_with_bins("T", config_resolver, bins=bins)

        f_cfg = _run_fisher_with_bins(
            "T",
            config_resolver,
            config_overrides={"delta_ell": 2},
        )

        np.testing.assert_allclose(
            f_api.get_fisher_matrix(),
            f_cfg.get_fisher_matrix(),
        )

    def test_custom_bins_config_matches_api(self, config_resolver):
        """bin_lmins/bin_lmaxs in config gives same Fisher as Bins()."""
        lmins = [2, 5]
        lmaxs = [4, 7]
        bins = Bins(lmins, lmaxs)
        f_api = _run_fisher_with_bins("T", config_resolver, bins=bins)

        f_cfg = _run_fisher_with_bins(
            "T",
            config_resolver,
            config_overrides={"bin_lmins": lmins, "bin_lmaxs": lmaxs},
        )

        np.testing.assert_allclose(
            f_api.get_fisher_matrix(),
            f_cfg.get_fisher_matrix(),
        )

    def test_custom_bins_config_shape(self, config_resolver):
        """Custom bins from config produce correct Fisher shape."""
        lmins = [2, 4, 7]
        lmaxs = [3, 6, 8]
        f = _run_fisher_with_bins(
            "T",
            config_resolver,
            config_overrides={"bin_lmins": lmins, "bin_lmaxs": lmaxs},
        )
        assert f.get_fisher_matrix().shape == (3, 3)
