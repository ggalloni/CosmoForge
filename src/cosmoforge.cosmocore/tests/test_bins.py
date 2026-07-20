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

    def test_lmin_below_two(self):
        """``Bins.fromdeltal(1, 4, 1)`` keeps the dipole bin (ADR 0009)."""
        bins = Bins.fromdeltal(1, 4, 1)
        assert bins.nbins == 4
        np.testing.assert_array_equal(bins.lmins, [1, 2, 3, 4])
        np.testing.assert_array_equal(bins.lmaxs, [1, 2, 3, 4])
        assert bins.lmin == 1
        assert bins.lmin_floor == 1

    def test_lmin_floor_zero_keeps_monopole(self):
        """Explicit ``lmin_floor=0`` is honoured for foreground templates."""
        bins = Bins([0, 1, 2], [0, 1, 2], lmin_floor=0)
        np.testing.assert_array_equal(bins.lmins, [0, 1, 2])
        np.testing.assert_array_equal(bins.lmaxs, [0, 1, 2])

    def test_default_lmin_floor_drops_below_two(self):
        """Default ``lmin_floor=2`` drops bins below 2 (legacy behaviour)."""
        bins = Bins([0, 1, 2, 3], [0, 1, 2, 3])
        np.testing.assert_array_equal(bins.lmins, [2, 3])
        np.testing.assert_array_equal(bins.lmaxs, [2, 3])

    def test_fromdeltal_delta1(self):
        """delta_ell=1 gives one bin per multipole."""
        bins = Bins.fromdeltal(2, 8, 1)
        assert bins.nbins == 7
        np.testing.assert_array_equal(bins.lmins, np.arange(2, 9))
        np.testing.assert_array_equal(bins.lmaxs, np.arange(2, 9))

    def test_midpoints(self):
        """lmid is the bin midpoint."""
        bins = Bins.fromdeltal(2, 7, 2)
        expected = np.array([2.5, 4.5, 6.5])
        np.testing.assert_array_equal(bins.lmid, expected)

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


class TestShapeWeights:
    """Test the per-ℓ bandpower shape weight (ADR-0019)."""

    def test_flat_cl_is_unity(self):
        """The flat-C_ell shape weight is 1 for every ell."""
        bins = Bins.fromdeltal(2, 10, 3)
        w = bins.shape_weights("Cl")
        assert w.shape == (bins.lmax + 1,)
        np.testing.assert_array_equal(w, np.ones(bins.lmax + 1))

    def test_flat_dl_closed_form(self):
        """The flat-D_ell shape weight is 2*pi/(ell*(ell+1))."""
        bins = Bins.fromdeltal(2, 10, 3)
        w = bins.shape_weights("Dl")
        ell = np.arange(2, bins.lmax + 1)
        np.testing.assert_allclose(w[2:], 2 * np.pi / (ell * (ell + 1)))

    def test_convention_case_insensitive(self):
        bins = Bins.fromdeltal(2, 10, 3)
        np.testing.assert_array_equal(bins.shape_weights("dl"), bins.shape_weights("Dl"))

    def test_unknown_convention_raises(self):
        bins = Bins.fromdeltal(2, 10, 3)
        with pytest.raises(ValueError, match="Must be 'Cl' or 'Dl'"):
            bins.shape_weights("Bl")

    def test_flat_dl_at_dipole(self):
        """ell=1 is well defined for the flat-D_ell shape (w = pi)."""
        bins = Bins.fromdeltal(1, 4, 1)  # lmin_floor=1, includes dipole
        w = bins.shape_weights("Dl")
        assert w[1] == pytest.approx(np.pi)

    def test_flat_dl_with_monopole_raises(self):
        """flat-D_ell is undefined at ell=0 and must raise."""
        bins = Bins([0, 1, 2], [0, 1, 2], lmin_floor=0)
        with pytest.raises(ValueError, match="monopole"):
            bins.shape_weights("Dl")


class TestLbinDeprecation:
    """lbin survives one release as a warn-and-forward alias for lmid."""

    def test_lbin_warns_and_forwards(self):
        bins = Bins.fromdeltal(2, 7, 2)
        with pytest.warns(DeprecationWarning, match="lmid"):
            legacy = bins.lbin
        np.testing.assert_array_equal(legacy, bins.lmid)
