"""Tests for spectra normalization conversion (Cl <-> Dl)."""

import numpy as np
import pytest

from cosmocore.in_out import convert_spectra_normalization, readcl
from cosmocore.settings import InputParams


class TestConvertSpectraNormalization:
    """Tests for the convert_spectra_normalization helper."""

    def test_identity_cl_to_cl(self):
        """Same normalization returns unchanged dict."""
        cls = {"TT": np.array([1.0, 2.0, 3.0])}
        result = convert_spectra_normalization(cls, "Cl", "Cl")
        np.testing.assert_array_equal(result["TT"], np.array([1.0, 2.0, 3.0]))

    def test_identity_dl_to_dl(self):
        """Same normalization returns unchanged dict."""
        cls = {"TT": np.array([1.0, 2.0, 3.0])}
        result = convert_spectra_normalization(cls, "Dl", "Dl")
        np.testing.assert_array_equal(result["TT"], np.array([1.0, 2.0, 3.0]))

    def test_cl_to_dl(self):
        """Cl -> Dl multiplies by ell*(ell+1)/(2*pi)."""
        ell = np.arange(2, 5, dtype=np.float64)
        factor = ell * (ell + 1) / (2 * np.pi)
        cl_values = np.array([1.0, 2.0, 3.0])
        cls = {"TT": cl_values.copy()}
        result = convert_spectra_normalization(cls, "Cl", "Dl")
        np.testing.assert_allclose(result["TT"], cl_values * factor)

    def test_dl_to_cl(self):
        """Dl -> Cl divides by ell*(ell+1)/(2*pi)."""
        ell = np.arange(2, 5, dtype=np.float64)
        factor = ell * (ell + 1) / (2 * np.pi)
        dl_values = np.array([1.0, 2.0, 3.0])
        cls = {"TT": dl_values.copy()}
        result = convert_spectra_normalization(cls, "Dl", "Cl")
        np.testing.assert_allclose(result["TT"], dl_values / factor)

    def test_round_trip(self):
        """Cl -> Dl -> Cl recovers original values."""
        original = np.array([1e-10, 2e-11, 5e-12, 3e-13])
        cls = {"TT": original.copy(), "EE": original.copy() * 0.1}
        convert_spectra_normalization(cls, "Cl", "Dl")
        convert_spectra_normalization(cls, "Dl", "Cl")
        np.testing.assert_allclose(cls["TT"], original, rtol=1e-14)
        np.testing.assert_allclose(cls["EE"], original * 0.1, rtol=1e-14)

    def test_multiple_spectra(self):
        """Conversion applies to all entries in the dict."""
        cls = {
            "TT": np.array([1.0, 2.0]),
            "EE": np.array([0.5, 0.3]),
        }
        ell = np.arange(2, 4, dtype=np.float64)
        factor = ell * (ell + 1) / (2 * np.pi)
        convert_spectra_normalization(cls, "Cl", "Dl")
        np.testing.assert_allclose(cls["TT"], np.array([1.0, 2.0]) * factor)
        np.testing.assert_allclose(cls["EE"], np.array([0.5, 0.3]) * factor)

    def test_custom_lstart(self):
        """Custom lstart shifts the ell range."""
        cls = {"TT": np.array([1.0, 2.0])}
        # lstart=0: ell=0 gives factor=0, ell=1 gives factor=1*2/(2pi)
        result = convert_spectra_normalization(cls, "Cl", "Dl", lstart=0)
        np.testing.assert_allclose(result["TT"][0], 0.0)
        np.testing.assert_allclose(result["TT"][1], 2.0 * 1 * 2 / (2 * np.pi))

    def test_invalid_normalization(self):
        """Unknown normalization raises ValueError."""
        cls = {"TT": np.array([1.0])}
        with pytest.raises(ValueError, match="Unknown spectra convention"):
            convert_spectra_normalization(cls, "Cl", "XX")

    def test_in_place_modification(self):
        """Function modifies and returns the same dict object."""
        cls = {"TT": np.array([1.0, 2.0])}
        result = convert_spectra_normalization(cls, "Cl", "Dl")
        assert result is cls

    def test_case_insensitive(self):
        """Convention strings are case-insensitive."""
        ell = np.arange(2, 5, dtype=np.float64)
        factor = ell * (ell + 1) / (2 * np.pi)
        cl_values = np.array([1.0, 2.0, 3.0])
        for from_str, to_str in [("cl", "dl"), ("CL", "DL"), ("cL", "Dl")]:
            cls = {"TT": cl_values.copy()}
            result = convert_spectra_normalization(cls, from_str, to_str)
            np.testing.assert_allclose(result["TT"], cl_values * factor)


class TestReadclNormalization:
    """Tests that readcl applies input_convention correctly."""

    def _write_cl_file(self, tmp_path, labels, data):
        """Write a Cl file with header and ell column + data columns."""
        filepath = tmp_path / "test_spectra.dat"
        n_rows = data.shape[0]
        ell_col = np.arange(2, 2 + n_rows, dtype=np.float64).reshape(-1, 1)
        full_data = np.hstack([ell_col, data])
        all_labels = ["ell"] + labels
        with open(filepath, "w") as f:
            f.write("# " + " ".join(all_labels) + "\n")
            for row in full_data:
                f.write(" ".join(f"{x:.16e}" for x in row) + "\n")
        return str(filepath)

    def test_readcl_default_no_conversion(self, tmp_path):
        """Default input_convention='Cl' does no conversion."""
        data = np.array([[1.0], [2.0], [3.0]])
        filepath = self._write_cl_file(tmp_path, ["TT"], data)
        params = InputParams()
        params.lmax = 4  # l=2,3,4 -> 3 values
        result = readcl(filepath, params)
        np.testing.assert_allclose(result["TT"], np.array([1.0, 2.0, 3.0]))

    def test_readcl_dl_input_converts_to_cl(self, tmp_path):
        """input_convention='Dl' converts Dl values to Cl on read."""
        ell = np.arange(2, 5, dtype=np.float64)
        factor = ell * (ell + 1) / (2 * np.pi)
        dl_values = np.array([1.0, 2.0, 3.0])
        data = dl_values.reshape(-1, 1)
        filepath = self._write_cl_file(tmp_path, ["TT"], data)
        params = InputParams()
        params.lmax = 4
        params.input_convention = "Dl"
        result = readcl(filepath, params)
        expected_cl = dl_values / factor
        np.testing.assert_allclose(result["TT"], expected_cl)
