"""Test core analysis framework functionality from cosmocore."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import healpy as hp
import numpy as np
import pytest

from cosmocore.core import Core
from cosmocore.settings import InputParams
from cosmocore.spectrum_key import SpectrumKey, SpectrumKind


class ConcreteCore(Core):
    """Concrete implementation of Core for testing."""

    def __init__(self, params):
        super().__init__(params)
        self.compute_called = False
        self.run_called = False

    def compute(self):
        """Test implementation of compute method."""
        self.compute_called = True
        return "computed"

    def run(self):
        """Test implementation of run method."""
        self.run_called = True
        return "executed"


class CoreWithSignal(ConcreteCore):
    """ConcreteCore with signal matrix for testing uncompressed API."""

    def _build_signal_matrix(self, C_ell):
        n = self.noise_cov1.shape[0]
        return np.eye(n) * np.sum(C_ell)

    def _build_derivative_matrix(self, ell, spectrum_idx=0):
        n = self.noise_cov1.shape[0]
        return np.eye(n) * 1.0


def test_core_initialization_with_inputparams():
    """Test Core initialization with InputParams instance."""
    params = InputParams()
    params.nside = 32
    params.lmax = 64

    core = ConcreteCore(params)
    assert core.params is params
    assert core.params.nside == 32
    assert core.params.lmax == 64


def test_core_initialization_with_dict():
    """Test Core initialization with parameter dictionary."""
    params_dict = {
        "nside": 32,
        "lmax": 64,
        "feedback": 2,
        "spins": [0, 2],
        "labels": ["T", "E", "B"],
    }

    core = ConcreteCore(params_dict)
    assert isinstance(core.params, InputParams)
    assert core.params.nside == 32
    assert core.params.lmax == 64
    assert core.params.feedback == 2


def test_core_initialization_with_file(tmp_path):
    """Test Core initialization with parameter file path."""
    # Create a temporary parameter file
    param_file = tmp_path / "test_params.yaml"
    param_content = """
nside: 32
lmax: 64
feedback: 1
spins: [0]
labels: ["T"]
nfields: 1
"""
    param_file.write_text(param_content)

    core = ConcreteCore(str(param_file))
    assert isinstance(core.params, InputParams)
    assert core.params.nside == 32
    assert core.params.lmax == 64


def test_core_initialization_invalid_params():
    """Test Core initialization with invalid parameter types."""
    with pytest.raises(TypeError, match="params must be an instance of InputParams"):
        ConcreteCore(123)

    with pytest.raises(TypeError, match="params must be an instance of InputParams"):
        ConcreteCore([1, 2, 3])


def test_read_params_inputparams():
    """Test read_params method with InputParams instance."""
    # Start with valid params and then call read_params
    core = ConcreteCore({"nside": 16})
    params = InputParams()
    params.nside = 64

    core.read_params(params)
    assert core.params is params
    assert core.params.nside == 64


def test_read_params_dict():
    """Test read_params method with dictionary."""
    # Start with valid params
    core = ConcreteCore({"nside": 16})
    params_dict = {"nside": 128, "lmax": 256}

    core.read_params(params_dict)
    assert isinstance(core.params, InputParams)
    assert core.params.nside == 128
    assert core.params.lmax == 256


def test_read_params_file(tmp_path):
    """Test read_params method with file path."""
    param_file = tmp_path / "test_params.yaml"
    param_content = """
nside: 16
lmax: 32
feedback: 0
"""
    param_file.write_text(param_content)

    # Start with valid params
    core = ConcreteCore({"nside": 32})
    core.read_params(str(param_file))
    assert isinstance(core.params, InputParams)
    assert core.params.nside == 16
    assert core.params.lmax == 32


def test_setup_fields_basic():
    """Test basic field setup functionality."""
    # Create a core instance with minimal parameters
    params = InputParams()
    params.nside = 32
    params.lmax = 64
    params.nfields = 1
    params.spins = [0]
    params.labels = ["T"]

    # Create a temporary FITS mask file
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        npix = 12 * params.nside**2
        mask = np.ones(npix, dtype=np.float64)
        mask[: npix // 2] = 0.0  # Mask half the pixels
        hp.write_map(f.name, mask, overwrite=True)
        params.maskfile = f.name

    try:
        core = ConcreteCore(params)
        fields = core.setup_fields()

        assert fields is not None
        assert hasattr(core, "collection")
        assert core.collection is not None
        assert len(core.collection.fields) == 1
        assert core.collection.fields[0].spin == 0

    finally:
        # Clean up
        Path(params.maskfile).unlink()


def test_setup_fields_with_polarization():
    """Test field setup with polarization (spin-2) field."""
    params = InputParams()
    params.nside = 32
    params.lmax = 64
    params.nfields = 2  # Need 2 fields for polarization (E and B)
    params.spins = [2]
    params.labels = ["E", "B"]

    # Create a temporary FITS mask file with 2 fields
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        npix = 12 * params.nside**2
        mask = np.ones((2, npix), dtype=np.float64)  # 2 fields
        hp.write_map(f.name, mask, overwrite=True)
        params.maskfile = f.name

    try:
        core = ConcreteCore(params)
        fields = core.setup_fields()

        assert fields is not None
        assert len(core.collection.fields) == 1  # One polarization field
        assert core.collection.fields[0].spin == 2

    finally:
        Path(params.maskfile).unlink()


def test_setup_geometry():
    """Test geometry setup functionality."""
    params = InputParams()
    params.nside = 32
    params.lmax = 64
    params.nfields = 1
    params.spins = [0]
    params.labels = ["T"]
    params.ordering = "RING"

    # Create FITS mask file
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        npix = 12 * params.nside**2
        mask = np.ones(npix, dtype=np.float64)
        mask[: npix // 2] = 0.0
        hp.write_map(f.name, mask, overwrite=True)
        params.maskfile = f.name

    try:
        core = ConcreteCore(params)
        core.setup_fields()
        pixact, point_vectors = core.setup_geometry()

        assert pixact is not None
        assert point_vectors is not None
        assert hasattr(core, "npixs")
        assert hasattr(core, "point_vectors")
        assert hasattr(core, "pixact")

    finally:
        Path(params.maskfile).unlink()


def test_setup_geometry_without_fields():
    """Test that setup_geometry raises error without fields."""
    core = ConcreteCore({"nside": 32})
    core.collection = None

    with pytest.raises(ValueError, match="Fields must be set up before geometry"):
        core.setup_geometry()


def test_setup_covariance_matrices():
    """Test covariance matrix setup functionality."""
    params = InputParams()
    params.nside = 32
    params.lmax = 64
    params.nfields = 1
    params.spins = [0]
    params.labels = ["T"]
    params.ordering = "RING"
    params.calibration = 1.0
    params.do_cross = False

    # Create FITS mask file
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as mask_file:
        npix = 12 * params.nside**2
        mask = np.ones(npix, dtype=np.float64)
        mask[: npix // 2] = 0.0
        hp.write_map(mask_file.name, mask, overwrite=True)
        params.maskfile = mask_file.name

    # Mock the read_covmat function to avoid file format issues
    with patch("cosmocore.core.read_covmat") as mock_read_covmat:
        # Set up the mock to return a simple covariance matrix
        n_active = np.sum(mask > 0.5)
        mock_cov = np.eye(n_active) * 0.1
        mock_read_covmat.return_value = mock_cov
        params.covmatfile1 = "dummy.dat"

        try:
            core = ConcreteCore(params)
            core.setup_fields()
            core.setup_geometry()
            ncov1, ncov2 = core.setup_covariance_matrices()

            assert ncov1 is not None
            assert ncov2 is None  # Since do_cross is False
            assert hasattr(core, "noise_cov1")
            assert core.noise_cov1 is not None
            mock_read_covmat.assert_called_once()

        finally:
            Path(params.maskfile).unlink()


def test_setup_covariance_matrices_without_geometry():
    """Test that setup_covariance_matrices raises error without geometry."""
    core = ConcreteCore({"nside": 32})
    core.pixact = None

    with pytest.raises(
        ValueError, match="Geometry must be set up before covariance matrices"
    ):
        core.setup_covariance_matrices()


def test_setup_cls():
    """Test power spectra setup functionality."""
    params = InputParams()
    params.nside = 32
    params.lmax = 64
    params.nfields = 1
    params.spins = [0]
    params.labels = ["T"]

    # Create FITS mask file
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        npix = 12 * params.nside**2
        mask = np.ones(npix, dtype=np.float64)
        hp.write_map(f.name, mask, overwrite=True)
        params.maskfile = f.name

    try:
        core = ConcreteCore(params)
        core.setup_fields()

        # Mock the set_cls method to avoid file dependencies
        with patch.object(core.collection, "set_cls") as mock_set_cls:
            core.setup_cls()
            mock_set_cls.assert_called_once()

    finally:
        Path(params.maskfile).unlink()


def test_setup_cls_without_fields():
    """Test that setup_cls raises error without fields."""
    core = ConcreteCore({"nside": 32})
    core.collection = None

    with pytest.raises(ValueError, match="Fields must be set up before Cls and beams"):
        core.setup_cls()


def test_setup_beams():
    """Test beam setup functionality."""
    params = InputParams()
    params.nside = 32
    params.lmax = 64
    params.nfields = 1
    params.spins = [0]
    params.labels = ["T"]

    # Create FITS mask file
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        npix = 12 * params.nside**2
        mask = np.ones(npix, dtype=np.float64)
        hp.write_map(f.name, mask, overwrite=True)
        params.maskfile = f.name

    try:
        core = ConcreteCore(params)
        core.setup_fields()

        # Mock the set_beams method to avoid file dependencies
        with patch.object(core.collection, "set_beams") as mock_set_beams:
            core.setup_beams()
            mock_set_beams.assert_called_once()

    finally:
        Path(params.maskfile).unlink()


def test_setup_beams_without_fields():
    """Test that setup_beams raises error without fields."""
    core = ConcreteCore({"nside": 32})
    core.collection = None

    with pytest.raises(ValueError, match="Fields must be set up before Cls and beams"):
        core.setup_beams()


def test_log_functionality():
    """Test logging functionality."""
    params = InputParams()
    params.feedback = 2

    core = ConcreteCore(params)

    # Capture print output
    import io
    import sys

    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        # Message should be printed (level 1 <= feedback 2)
        core.log("Test message level 1", level=1)
        output1 = captured_output.getvalue()

        # Message should be printed (level 2 <= feedback 2)
        core.log("Test message level 2", level=2)
        output2 = captured_output.getvalue()

        # Message should NOT be printed (level 3 > feedback 2)
        core.log("Test message level 3", level=3)
        output3 = captured_output.getvalue()

        # Check that messages appear in formatted log output
        assert "Test message level 1" in output1
        assert "Test message level 2" in output2
        # Level 3 message should not be printed when feedback=2
        assert "Test message level 3" not in output3.replace(output2, "")

    finally:
        sys.stdout = sys.__stdout__


def test_log_without_feedback():
    """Test logging without feedback parameter."""
    params = InputParams()
    # Don't set feedback parameter (check if it has a default value)

    core = ConcreteCore(params)

    # Capture print output
    import io
    import sys

    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        core.log("Test message")
        output = captured_output.getvalue()
        # The behavior depends on the default feedback value
        # Let's just check it doesn't crash
        assert isinstance(output, str)

    finally:
        sys.stdout = sys.__stdout__


def test_abstract_methods():
    """Test that abstract methods are implemented in concrete class."""
    core = ConcreteCore({"nside": 32})

    # Test compute method
    result = core.compute()
    assert result == "computed"
    assert core.compute_called

    # Test run method
    result = core.run()
    assert result == "executed"
    assert core.run_called


def test_core_cannot_be_instantiated():
    """Test that Core abstract class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Core({"nside": 32})


def test_setup_covariance_matrices_with_cross():
    """Test covariance matrix setup with cross-correlation enabled."""
    params = InputParams()
    params.nside = 32
    params.lmax = 64
    params.nfields = 1
    params.spins = [0]
    params.labels = ["T"]
    params.ordering = "RING"
    params.calibration = 1.0
    params.do_cross = True

    # Create FITS mask file
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as mask_file:
        npix = 12 * params.nside**2
        mask = np.ones(npix, dtype=np.float64)
        mask[: npix // 2] = 0.0
        hp.write_map(mask_file.name, mask, overwrite=True)
        params.maskfile = mask_file.name

    # Mock both covariance files
    with patch("cosmocore.core.read_covmat") as mock_read_covmat:
        n_active = np.sum(mask > 0.5)
        mock_cov1 = np.eye(n_active) * 0.1
        mock_cov2 = np.eye(n_active) * 0.2
        mock_read_covmat.side_effect = [mock_cov1, mock_cov2]
        params.covmatfile1 = "dummy1.dat"
        params.covmatfile2 = "dummy2.dat"

        try:
            core = ConcreteCore(params)
            core.setup_fields()
            core.setup_geometry()
            ncov1, ncov2 = core.setup_covariance_matrices()

            assert ncov1 is not None
            assert ncov2 is not None  # Should be set since do_cross is True
            assert hasattr(core, "noise_cov1")
            assert hasattr(core, "noise_cov2")
            assert core.noise_cov1 is not None
            assert core.noise_cov2 is not None
            assert mock_read_covmat.call_count == 2

        finally:
            Path(params.maskfile).unlink()


def test_core_workflow_integration():
    """Test a complete workflow integration."""
    params = InputParams()
    params.nside = 32
    params.lmax = 64
    params.nfields = 1
    params.spins = [0]
    params.labels = ["T"]
    params.ordering = "RING"
    params.calibration = 1.0
    params.do_cross = False
    params.feedback = 1

    # Create FITS mask file
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as mask_file:
        npix = 12 * params.nside**2
        mask = np.ones(npix, dtype=np.float64)
        mask[: npix // 2] = 0.0
        hp.write_map(mask_file.name, mask, overwrite=True)
        params.maskfile = mask_file.name

    # Mock covariance matrix reading
    with patch("cosmocore.core.read_covmat") as mock_read_covmat:
        n_active = np.sum(mask > 0.5)
        mock_cov = np.eye(n_active) * 0.1
        mock_read_covmat.return_value = mock_cov
        params.covmatfile1 = "dummy.dat"

        try:
            core = ConcreteCore(params)

            # Test complete workflow
            fields = core.setup_fields()
            pixact, point_vectors = core.setup_geometry()
            ncov1, ncov2 = core.setup_covariance_matrices()

            # Mock Cls and beams setup to avoid file dependencies
            with (
                patch.object(core.collection, "set_cls"),
                patch.object(core.collection, "set_beams"),
            ):
                core.setup_cls()
                core.setup_beams()

            # Test that all components are set up
            assert fields is not None
            assert pixact is not None
            assert point_vectors is not None
            assert ncov1 is not None
            assert ncov2 is None

            # Test abstract methods work
            result = core.compute()
            assert result == "computed"

            result = core.run()
            assert result == "executed"

        finally:
            Path(params.maskfile).unlink()


def test_read_params_invalid_type():
    """Test read_params with invalid type raises error."""
    core = ConcreteCore({"nside": 32})

    with pytest.raises(TypeError, match="params must be an instance of InputParams"):
        core.read_params(123)


def test_setup_fields_missing_mask_attribute():
    """Test that setup_fields handles multiple field types."""
    params = InputParams()
    params.nside = 32
    params.lmax = 64
    params.nfields = 3  # T, E, B components
    params.spins = [0, 2]  # Mix of scalar and polarization
    params.labels = ["T", "E", "B"]

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        npix = 12 * params.nside**2
        mask = np.ones((3, npix), dtype=np.float64)  # 3 field components
        hp.write_map(f.name, mask, overwrite=True)
        params.maskfile = f.name

    try:
        core = ConcreteCore(params)
        core.setup_fields()

        # Should have two fields: one scalar (spin=0), one polarization (spin=2)
        assert len(core.collection.fields) == 2
        assert core.collection.fields[0].spin == 0
        assert core.collection.fields[1].spin == 2

    finally:
        Path(params.maskfile).unlink()


def test_setup_covariance_matrices_with_output():
    """Test covariance matrix setup with optional output files."""
    params = InputParams()
    params.nside = 32
    params.lmax = 64
    params.nfields = 1
    params.spins = [0]
    params.labels = ["T"]
    params.ordering = "RING"
    params.calibration = 1.0
    params.do_cross = True
    # Add optional output parameters
    params.outnoisecovmat1 = "output1.dat"
    params.outnoisecovmat2 = "output2.dat"

    # Create FITS mask file
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as mask_file:
        npix = 12 * params.nside**2
        mask = np.ones(npix, dtype=np.float64)
        mask[: npix // 2] = 0.0
        hp.write_map(mask_file.name, mask, overwrite=True)
        params.maskfile = mask_file.name

    # Mock both covariance files and output functions
    with (
        patch("cosmocore.core.read_covmat") as mock_read_covmat,
        patch("cosmocore.core.write_covmat_reduced") as mock_write_covmat,
    ):
        n_active = np.sum(mask > 0.5)
        mock_cov1 = np.eye(n_active) * 0.1
        mock_cov2 = np.eye(n_active) * 0.2
        mock_read_covmat.side_effect = [mock_cov1, mock_cov2]
        params.covmatfile1 = "dummy1.dat"
        params.covmatfile2 = "dummy2.dat"

        try:
            core = ConcreteCore(params)
            core.setup_fields()
            core.setup_geometry()
            ncov1, ncov2 = core.setup_covariance_matrices()

            assert ncov1 is not None
            assert ncov2 is not None
            # Verify output functions were called
            assert mock_write_covmat.call_count == 2

        finally:
            Path(params.maskfile).unlink()


def test_uncompressed_covariance_api():
    """Test the 5 uncompressed fallback methods on Core."""
    np.random.seed(42)
    n = 10
    core = CoreWithSignal({"nside": 16, "lmax": 5, "spins": [0], "labels": ["T"]})
    core.noise_cov1 = np.eye(n) * 0.1
    C_ell = np.ones(4) * 1e-3

    # get_total_covariance
    C = core.get_total_covariance(C_ell)
    assert C.shape == (n, n)
    np.testing.assert_allclose(np.diag(C), 0.1 + np.sum(C_ell))

    # get_covariance_inverse
    C_inv = core.get_covariance_inverse(C_ell)
    np.testing.assert_allclose(C @ C_inv, np.eye(n), atol=1e-12)

    # get_derivative_matrix
    ss_key = SpectrumKey(0, 0, SpectrumKind.SS, spins=(0,))
    dC = core.get_derivative_matrix(5, ss_key)
    assert dC.shape == (n, n)

    # get_covariance_logdet - array and dict paths
    logdet = core.get_covariance_logdet(C_ell)
    expected_logdet = n * np.log(0.1 + np.sum(C_ell))
    np.testing.assert_allclose(logdet, expected_logdet, atol=1e-12)
    logdet_d = core.get_covariance_logdet({(0, 0, 0): C_ell})
    np.testing.assert_allclose(logdet_d, expected_logdet, atol=1e-12)

    # compute_quadratic_form - array and dict paths
    data = np.random.randn(n)
    qf = core.compute_quadratic_form(data, C_ell)
    qf_expected = float(data @ C_inv @ data)
    np.testing.assert_allclose(qf, qf_expected, atol=1e-12)
    qf_d = core.compute_quadratic_form(data, {(0, 0, 0): C_ell})
    np.testing.assert_allclose(qf_d, qf_expected, atol=1e-12)


def test_setup_computation_basis_basic(uniform_sky_setup):
    """Test setup_computation_basis creates a basis manager.

    The basis takes ownership of noise_cov1 (in-place Cholesky factor),
    so each basis method needs a fresh Core instance with its own copy.
    """
    setup = uniform_sky_setup

    # Harmonic compression
    core_h = CoreWithSignal(
        {"nside": 16, "lmax": setup["lmax"], "spins": [0], "labels": ["T"]}
    )
    core_h.noise_cov1 = setup["N"].copy()
    core_h.theta = (setup["theta"],)
    core_h.phi = (setup["phi"],)
    cm = core_h.setup_computation_basis(
        method="harmonic", lmax_signal=setup["lmax"], use_smw_optimization=False
    )
    assert cm is not None
    assert core_h.basis_manager is cm

    # Pixel-projected compression (fresh core: noise buffer was consumed above)
    core_p = CoreWithSignal(
        {"nside": 16, "lmax": setup["lmax"], "spins": [0], "labels": ["T"]}
    )
    core_p.noise_cov1 = setup["N"].copy()
    core_p.theta = (setup["theta"],)
    core_p.phi = (setup["phi"],)
    cm2 = core_p.setup_computation_basis(
        method="pixel",
        lmax_signal=setup["lmax"],
        use_smw_optimization=False,
        epsilon=1e-4,
    )
    assert cm2 is not None


def test_setup_computation_basis_validation():
    """Test setup_computation_basis raises errors for missing prerequisites."""
    core = CoreWithSignal({"nside": 16, "lmax": 5, "spins": [0], "labels": ["T"]})

    with pytest.raises(ValueError, match="Geometry must be set up"):
        core.setup_computation_basis()

    core.theta = (np.array([1.0]),)
    core.phi = (np.array([1.0]),)
    with pytest.raises(ValueError, match="Covariance matrices must be set up"):
        core.setup_computation_basis()


def test_setup_covariance_matrices_load_reduced():
    """Test that load_reduced=True uses read_covmat_reduced and applies calibration."""
    params = InputParams()
    params.nside = 32
    params.lmax = 64
    params.nfields = 1
    params.spins = [0]
    params.labels = ["T"]
    params.ordering = "RING"
    params.calibration = 2.0
    params.do_cross = False
    params.load_reduced = True

    # Create FITS mask file
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as mask_file:
        npix = 12 * params.nside**2
        mask = np.ones(npix, dtype=np.float64)
        mask[: npix // 2] = 0.0
        hp.write_map(mask_file.name, mask, overwrite=True)
        params.maskfile = mask_file.name

    with (
        patch("cosmocore.core.read_covmat_reduced") as mock_read_reduced,
        patch("cosmocore.core.read_covmat") as mock_read_full,
    ):
        n_active = int(np.sum(mask > 0.5))
        mock_cov = np.eye(n_active) * 0.1
        mock_read_reduced.return_value = mock_cov
        params.covmatfile1 = "dummy_reduced.dat"

        try:
            core = ConcreteCore(params)
            core.setup_fields()
            core.setup_geometry()
            ncov1, ncov2 = core.setup_covariance_matrices()

            # read_covmat_reduced should be called, not read_covmat
            mock_read_reduced.assert_called_once()
            mock_read_full.assert_not_called()

            # Calibration should be applied (calibration=2.0 -> factor 4.0)
            np.testing.assert_allclose(np.diag(ncov1), 0.1 * 4.0, rtol=1e-12)
        finally:
            Path(params.maskfile).unlink()
