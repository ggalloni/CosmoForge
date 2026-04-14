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


def _run_fisher_with_bins(
    fields: str,
    config_resolver,
    bins: Bins | None = None,
    compression: dict | None = None,
) -> Fisher:
    """Run Fisher pipeline with optional binning and compression."""
    config_file = config_resolver(f"tests/data/nside4/{fields}/config.yaml")
    fisher = Fisher(config_file, compression=compression)
    if bins is not None:
        fisher.set_binning(bins)
    fisher.run()
    os.unlink(config_file)
    return fisher


def _run_spectra_with_bins(
    fields: str,
    config_resolver,
    bins: Bins | None = None,
    compression: dict | None = None,
    fisher: Fisher | None = None,
) -> Spectra:
    """Run Spectra pipeline with optional binning and compression."""
    config_file = config_resolver(f"tests/data/nside4/{fields}/config.yaml")
    qml = Spectra(config_file, fisher=fisher, compression=compression)
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

    def _make_P_ell(self, bins, n_ell):
        """Build (nbins, n_ell) binning matrix mapping Fisher ell indices to bins."""
        P_full, _ = bins._bin_operators()
        # P_full is (nbins, bins.lmax+1). Fisher indices: 0..n_ell-1 = ells 2..lmax.
        # Pad P to cover full Fisher range if bins don't span all ells.
        n_ell_P = P_full.shape[1] - 2  # columns for ell>=2
        P_ell = np.zeros((bins.nbins, n_ell))
        P_ell[:, :n_ell_P] = P_full[:, 2:]
        return P_ell

    def test_fisher_linearity(self, unbinned_fisher, binned_fisher, bins):
        """F_binned == P @ F_unbinned @ P^T (exact by linearity of trace)."""
        F_unbinned = unbinned_fisher["fisher"]
        F_binned = binned_fisher["fisher"]

        n_ell = F_unbinned.shape[0]
        P_ell = self._make_P_ell(bins, n_ell)

        F_posthoc = P_ell @ F_unbinned @ P_ell.T

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
        """q_binned == P @ q_unbinned (exact by linearity)."""
        # Unbinned spectra
        f_unb = _run_fisher_with_bins("T", config_resolver)
        s_unb = _run_spectra_with_bins("T", config_resolver, fisher=f_unb)
        q_unbinned = s_unb.qml_results  # (nsims, n_ell)

        # Binned spectra
        f_bin = _run_fisher_with_bins("T", config_resolver, bins=bins)
        s_bin = _run_spectra_with_bins("T", config_resolver, bins=bins, fisher=f_bin)
        q_binned = s_bin.qml_results  # (nsims, nbins)

        n_ell = q_unbinned.shape[1]
        P_ell = self._make_P_ell(bins, n_ell)

        q_posthoc = q_unbinned @ P_ell.T  # (nsims, nbins)

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

    def _make_P_ell(self, bins, n_ell):
        P_full, _ = bins._bin_operators()
        n_ell_P = P_full.shape[1] - 2
        P_ell = np.zeros((bins.nbins, n_ell))
        P_ell[:, :n_ell_P] = P_full[:, 2:]
        return P_ell

    def test_fisher_linearity_compressed(self, config_resolver, bins, compression):
        """Compressed: F_binned == P @ F_unbinned @ P^T."""
        f_unb = _run_fisher_with_bins("T", config_resolver, compression=compression)
        F_unbinned = f_unb.get_fisher_matrix()

        f_bin = _run_fisher_with_bins(
            "T", config_resolver, bins=bins, compression=compression
        )
        F_binned = f_bin.get_fisher_matrix()

        n_ell = F_unbinned.shape[0]
        P_ell = self._make_P_ell(bins, n_ell)
        F_posthoc = P_ell @ F_unbinned @ P_ell.T

        np.testing.assert_allclose(
            F_binned,
            F_posthoc,
            rtol=1e-10,
            err_msg="Compressed binned Fisher does not match post-hoc binned",
        )

    def test_qml_linearity_compressed(self, config_resolver, bins, compression):
        """Compressed: q_binned == P @ q_unbinned."""
        f_unb = _run_fisher_with_bins("T", config_resolver, compression=compression)
        s_unb = _run_spectra_with_bins(
            "T", config_resolver, compression=compression, fisher=f_unb
        )
        q_unbinned = s_unb.qml_results

        f_bin = _run_fisher_with_bins(
            "T", config_resolver, bins=bins, compression=compression
        )
        s_bin = _run_spectra_with_bins(
            "T", config_resolver, bins=bins, compression=compression, fisher=f_bin
        )
        q_binned = s_bin.qml_results

        n_ell = q_unbinned.shape[1]
        P_ell = self._make_P_ell(bins, n_ell)
        q_posthoc = q_unbinned @ P_ell.T

        np.testing.assert_allclose(
            q_binned,
            q_posthoc,
            rtol=1e-10,
            err_msg="Compressed binned q does not match post-hoc binned",
        )


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
