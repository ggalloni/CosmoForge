"""Test settings and configuration functionality from cosmocore."""

import warnings

import pytest
import yaml

from cosmocore import InputParams


def test_input_params_yaml_reading(tmp_path):
    """Test YAML parameter file reading with real InputParams attributes."""
    # Create test configuration file using actual InputParams attributes
    config_data = {
        "nside": 32,
        "lmax": 64,
        "spins": [0, 2],
        "labels": ["T", "E", "B"],
        "physical_labels": ["T", "Q", "U"],
        "maskfile": "data/mask.fits",
        "inputclfile": "data/cls.dat",
        "covmatfile1": "data/noise1.bin",
        "beam_file": "data/beam.fits",
        "fwhmarcmin": 5.0,
        "calibration": 1.2,
        "feedback": 2,
        "apply_pixwin": False,
        "ordering": "RING",
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Test parameter reading
    params = InputParams.read_parameter_file(str(config_file))

    # Verify basic parameters
    assert params.nside == 32
    assert params.lmax == 64
    assert params.spins == [0, 2]
    assert params.labels == ["T", "E", "B"]
    assert params.physical_labels == ["T", "Q", "U"]

    # Verify file paths
    assert params.maskfile == "data/mask.fits"
    assert params.inputclfile == "data/cls.dat"
    assert params.covmatfile1 == "data/noise1.bin"
    assert params.beam_file == "data/beam.fits"

    # Verify numerical parameters
    assert params.fwhmarcmin == 5.0
    assert params.calibration == 1.2
    assert params.feedback == 2
    assert params.apply_pixwin is False
    assert params.ordering == "RING"


def test_input_params_default_values():
    """Test InputParams with default initialization."""
    params = InputParams()

    # Test default values from the actual implementation
    assert params.nside == 16
    assert params.lmax == 64
    assert params.spins == [0, 2]
    assert params.labels == ["T", "E", "B"]
    assert params.physical_labels is None

    # Test file paths defaults
    assert params.inputclfile == "inputs/cls.dat"
    assert params.maskfile == "inputs/mask.fits"
    assert params.covmatfile1 == "inputs/NCVM1.bin"

    # Test numerical defaults
    assert params.feedback == 1
    assert params.calibration == 1.0
    assert params.fwhmarcmin == 440.0
    assert params.apply_pixwin is True
    assert params.ordering == "RING"


def test_input_params_update_method():
    """Test the update method directly."""
    params = InputParams()

    # Store original values
    orig_nside = params.nside
    orig_lmax = params.lmax

    # Update with new values
    updates = {"nside": 64, "lmax": 128, "feedback": 0, "calibration": 2.0}

    params.update(updates)

    # Verify updates
    assert params.nside == 64
    assert params.lmax == 128
    assert params.feedback == 0
    assert params.calibration == 2.0

    # Verify derived parameters are recomputed
    assert params.nside != orig_nside
    assert params.lmax != orig_lmax


def test_field_expansion_single_char():
    """Test expansion of concatenated single-character field labels."""
    params = InputParams()

    # Test QU expansion
    params.update({"physical_labels": ["QU"]})
    assert params.physical_labels == ["Q", "U"]

    # Test TQU expansion
    params.update({"labels": ["TQU"]})
    assert params.labels == ["T", "Q", "U"]
    assert params.nfields == 3
    assert params.nspectra == 6

    # Test mixed expansion
    params.update({"labels": ["T", "QU"], "physical_labels": ["T", "QU"]})
    assert params.labels == ["T", "Q", "U"]
    assert params.physical_labels == ["T", "Q", "U"]


def test_field_expansion_multichar():
    """Test multi-character field support and underscore separators."""
    params = InputParams()

    # Multi-character fields should be preserved
    params.update({"labels": ["T1", "T2", "E1"]})
    assert params.labels == ["T1", "T2", "E1"]

    # Test underscore separator expansion
    params.update({"labels": ["T1_T2"], "physical_labels": ["MAP1_MAP2"]})
    assert params.labels == ["T1", "T2"]
    assert params.physical_labels == ["MAP1", "MAP2"]

    # Test mixed: single-char concat, multi-char individual, underscore
    params.update(
        {
            "labels": ["T1", "QU", "E1_E2"],
            "physical_labels": ["MAP1", "QU", "FREQ1_FREQ2"],
        }
    )
    assert params.labels == ["T1", "Q", "U", "E1", "E2"]
    assert params.physical_labels == ["MAP1", "Q", "U", "FREQ1", "FREQ2"]


def test_field_expansion_compatibility():
    """Test backward compatibility with existing configurations."""
    params = InputParams()

    # Standard configuration should work unchanged
    config = {"labels": ["T", "E", "B"], "physical_labels": ["T", "Q", "U"], "nside": 16}
    params.update(config)

    assert params.labels == ["T", "E", "B"]
    assert params.physical_labels == ["T", "Q", "U"]
    assert params.nfields == 3
    assert params.nspectra == 6


# =========================================================================
# Ordering normalization tests
# =========================================================================


class TestNormalizeOrdering:
    """Tests for InputParams._normalize_ordering()."""

    def test_string_ring_case_insensitive(self):
        assert InputParams._normalize_ordering("RING") == "RING"
        assert InputParams._normalize_ordering("ring") == "RING"
        assert InputParams._normalize_ordering("Ring") == "RING"
        assert InputParams._normalize_ordering(" ring ") == "RING"

    def test_string_nested_case_insensitive(self):
        assert InputParams._normalize_ordering("NESTED") == "NESTED"
        assert InputParams._normalize_ordering("nested") == "NESTED"
        assert InputParams._normalize_ordering("Nested") == "NESTED"

    def test_string_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown ordering"):
            InputParams._normalize_ordering("GALACTIC")

    def test_legacy_int_ring(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            # 0 maps to RING
            assert InputParams._normalize_ordering(0) == "RING"
            # Any value != 1 maps to RING
            assert InputParams._normalize_ordering(2) == "RING"

    def test_legacy_int_nested(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert InputParams._normalize_ordering(1) == "NESTED"

    def test_legacy_int_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            InputParams._normalize_ordering(0)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="ordering must be str or int"):
            InputParams._normalize_ordering([1, 2])

    def test_update_normalizes_ordering(self):
        params = InputParams()
        params.update({"ordering": "nested"})
        assert params.ordering == "NESTED"


# =========================================================================
# Smoothing-type normalization tests
# =========================================================================


class TestNormalizeSmoothingType:
    """Tests for InputParams._normalize_smoothing_type()."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("none", "none"),
            ("NONE", "none"),
            ("None", "none"),
            ("gaussian", "gaussian"),
            ("GAUSSIAN", "gaussian"),
            ("Gaussian", "gaussian"),
            ("cosine_legacy", "cosine_legacy"),
            ("COSINE_LEGACY", "cosine_legacy"),
            ("cosine_npipe", "cosine_npipe"),
            ("COSINE_NPIPE", "cosine_npipe"),
            ("file", "file"),
            ("FILE", "file"),
        ],
    )
    def test_string_case_insensitive(self, input_val, expected):
        assert InputParams._normalize_smoothing_type(input_val) == expected

    def test_string_with_whitespace(self):
        assert InputParams._normalize_smoothing_type(" cosine_legacy ") == "cosine_legacy"

    def test_string_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown smoothing_type"):
            InputParams._normalize_smoothing_type("hamming")

    def test_bare_cosine_deprecated_aliases_legacy(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert InputParams._normalize_smoothing_type("cosine") == "cosine_legacy"
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    @pytest.mark.parametrize(
        "int_val,expected",
        [(0, "none"), (1, "gaussian"), (2, "cosine_legacy"), (3, "file")],
    )
    def test_legacy_int_codes(self, int_val, expected):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert InputParams._normalize_smoothing_type(int_val) == expected

    def test_legacy_int_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown smoothing_type"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                InputParams._normalize_smoothing_type(5)

    def test_legacy_int_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            InputParams._normalize_smoothing_type(2)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="smoothing_type must be str or int"):
            InputParams._normalize_smoothing_type({"type": "cosine"})

    def test_update_normalizes_smoothing_type(self):
        params = InputParams()
        params.update({"smoothing_type": "GAUSSIAN"})
        assert params.smoothing_type == "gaussian"
