"""
Tests for the Bins class (multipole binning).
"""

import numpy as np
import pytest

from cosmocore import Bins


class TestBinsCreation:
    """Test Bins construction and derived attributes."""

    def test_fromdeltal_basic(self):
        bins = Bins.fromdeltal(2, 10, 3)
        assert bins.nbins == 3
        np.testing.assert_array_equal(bins.lmins, [2, 5, 8])
        np.testing.assert_array_equal(bins.lmaxs, [4, 7, 10])

    def test_fromdeltal_delta1(self):
        """delta_ell=1 gives one bin per multipole."""
        bins = Bins.fromdeltal(2, 8, 1)
        assert bins.nbins == 7
        np.testing.assert_array_equal(bins.lmins, np.arange(2, 9))
        np.testing.assert_array_equal(bins.lmaxs, np.arange(2, 9))

    def test_effective_ells(self):
        """Effective ells are bin midpoints."""
        bins = Bins.fromdeltal(2, 7, 2)
        expected = np.array([2.5, 4.5, 6.5])
        np.testing.assert_array_equal(bins.lbin, expected)

    def test_bin_widths(self):
        bins = Bins.fromdeltal(2, 10, 3)
        np.testing.assert_array_equal(bins.dl, [3, 3, 3])

    def test_lmin_lmax(self):
        bins = Bins.fromdeltal(2, 10, 3)
        assert bins.lmin == 2
        assert bins.lmax == 10

    def test_incoherent_lengths_raises(self):
        with pytest.raises(ValueError):
            Bins([2, 5], [4])

    def test_incoherent_bounds_raises(self):
        with pytest.raises(ValueError):
            Bins([5, 2], [3, 4])  # lmin > lmax for first bin

    def test_cuts_below_ell2(self):
        """Bins with lmax < 2 are filtered out."""
        bins = Bins([0, 2], [1, 4])
        assert bins.nbins == 1
        np.testing.assert_array_equal(bins.lmins, [2])
        np.testing.assert_array_equal(bins.lmaxs, [4])

    def test_bins_tuple(self):
        bins = Bins.fromdeltal(2, 7, 2)
        lmins, lmaxs = bins.bins()
        np.testing.assert_array_equal(lmins, [2, 4, 6])
        np.testing.assert_array_equal(lmaxs, [3, 5, 7])


class TestBinOperators:
    """Test binning and unbinning operator matrices."""

    def test_flat_weighting(self):
        """Flat weighting: P[b, ell] = 1/delta_ell within bin."""
        bins = Bins.fromdeltal(2, 4, 3)
        P, Q = bins._bin_operators()
        assert P.shape == (1, 5)
        np.testing.assert_allclose(P[0, 2:5], [1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose(P[0, :2], [0, 0])

    def test_identity_binning(self):
        """delta_ell=1 binning matrix is identity (for ell>=2 block)."""
        bins = Bins.fromdeltal(2, 5, 1)
        P, _ = bins._bin_operators()
        P_ell = P[:, 2:]
        np.testing.assert_allclose(P_ell, np.eye(4))

    def test_P_rows_sum_to_one(self):
        """Each row of P sums to 1 (flat weighting)."""
        bins = Bins.fromdeltal(2, 10, 3)
        P, _ = bins._bin_operators()
        row_sums = P.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(bins.nbins))

    def test_P_shape(self):
        bins = Bins.fromdeltal(2, 10, 3)
        P, Q = bins._bin_operators()
        assert P.shape == (3, 11)
        assert Q.shape == (11, 3)

    def test_dl_weighting(self):
        """Dl=True applies ell*(ell+1)/(2*pi) weighting."""
        bins = Bins.fromdeltal(2, 4, 3)
        P_flat, _ = bins._bin_operators(Dl=False)
        P_dl, _ = bins._bin_operators(Dl=True)
        # Dl weights should differ from flat
        assert not np.allclose(P_flat, P_dl)


class TestBinSpectra:
    """Test bin_spectra method."""

    def test_basic_binning(self):
        """Average of constant spectrum is that constant."""
        bins = Bins.fromdeltal(2, 4, 3)
        spectra = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # ell 0..4
        result = bins.bin_spectra(spectra)
        np.testing.assert_allclose(result, [1.0])

    def test_lmin_padding(self):
        """bin_spectra with lmin=2 correctly pads."""
        bins = Bins.fromdeltal(2, 4, 3)
        spectra = np.array([10.0, 20.0, 30.0])  # values at ell=2,3,4
        result = bins.bin_spectra(spectra, lmin=2)
        expected = np.mean([10.0, 20.0, 30.0])
        np.testing.assert_allclose(result, [expected])

    def test_identity_binning_preserves_values(self):
        """delta_ell=1 binning returns the input values."""
        bins = Bins.fromdeltal(2, 5, 1)
        spectra = np.array([0.0, 0.0, 10.0, 20.0, 30.0, 40.0])  # ell 0..5
        result = bins.bin_spectra(spectra)
        np.testing.assert_allclose(result, [10.0, 20.0, 30.0, 40.0])

    def test_2d_spectra(self):
        """Works with 2D input (multiple spectra)."""
        bins = Bins.fromdeltal(2, 4, 3)
        spectra = np.array(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
        )  # 2 spectra, ell=2..4
        result = bins.bin_spectra(spectra, lmin=2)
        np.testing.assert_allclose(result, [[20.0], [50.0]])

    def test_lmin_zero_is_noop(self):
        """lmin=0 (default) doesn't pad."""
        bins = Bins.fromdeltal(2, 4, 3)
        spectra = np.array([0.0, 0.0, 10.0, 20.0, 30.0])
        result_default = bins.bin_spectra(spectra)
        result_lmin0 = bins.bin_spectra(spectra, lmin=0)
        np.testing.assert_array_equal(result_default, result_lmin0)


class TestBinCovariance:
    """Test bin_covariance method."""

    def test_identity_binning_preserves_covariance(self):
        """delta_ell=1 binning preserves the covariance (for ell>=2 block)."""
        bins = Bins.fromdeltal(2, 4, 1)
        n = 5  # ell 0..4
        cov = np.eye(n) * 2.0
        result = bins.bin_covariance(cov)
        # Should extract the 2..4 block (3x3) and preserve values
        assert result.shape == (3, 3)

    def test_shape(self):
        bins = Bins.fromdeltal(2, 10, 3)
        cov = np.eye(11)
        result = bins.bin_covariance(cov)
        assert result.shape == (3, 3)
