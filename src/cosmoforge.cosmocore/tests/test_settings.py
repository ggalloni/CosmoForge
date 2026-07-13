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
    assert params.fwhmarcmin == 440.0
    assert params.apply_pixwin is True
    assert params.ordering == "RING"


def test_output_path_defaults_are_none():
    """Opt-in persistence (ADR-0015): every out* default is None (no implicit writes)."""
    params = InputParams()
    for attr in (
        "outinvcovmatfile1",
        "outinvcovmatfile2",
        "outnoisecovmat1",
        "outnoisecovmat2",
        "output_geometry_file",
        "outfilefisher",
        "outcovmatfile",
        "outerrfile",
    ):
        assert getattr(params, attr) is None, f"{attr} default should be None"


def test_input_params_update_method():
    """Test the update method directly."""
    params = InputParams()

    # Store original values
    orig_nside = params.nside
    orig_lmax = params.lmax

    # Update with new values
    updates = {"nside": 64, "lmax": 128, "feedback": 0}

    params.update(updates)

    # Verify updates
    assert params.nside == 64
    assert params.lmax == 128
    assert params.feedback == 0

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

    def test_update_deprecates_bare_cosine_to_legacy(self):
        params = InputParams()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            params.update({"smoothing_type": "cosine"})
            assert params.smoothing_type == "cosine_legacy"
            assert any(issubclass(rec.category, DeprecationWarning) for rec in w)


def test_calibration_removed_from_public_surface():
    """``calibration`` and ``smooth_pol`` are gone from InputParams."""
    params = InputParams()
    assert not hasattr(params, "calibration")
    assert not hasattr(params, "smooth_pol")


def test_calibration_unity_is_accepted_and_ignored():
    """A legacy ``calibration: 1.0`` key still loads, silently.

    Every config that ever shipped set it to 1.0, so those users see nothing.
    """
    params = InputParams()
    params.update({"nside": 32, "calibration": 1.0})
    assert params.nside == 32
    assert not hasattr(params, "calibration")


def test_calibration_non_unity_raises():
    """A non-unit ``calibration`` raises rather than being silently dropped.

    ``update()`` ignores unknown keys, so a plain removal would turn a real
    numerical setting into a no-op with no diagnostics — the one outcome that
    could corrupt results unnoticed. The key is therefore still recognised,
    purely so it can refuse.
    """
    params = InputParams()
    with pytest.raises(ValueError, match="calibration"):
        params.update({"calibration": 2.0})


def test_calibration_non_unity_raises_from_yaml(tmp_path):
    """The guard fires through ``read_parameter_file`` too, not just update()."""
    config_file = tmp_path / "cal.yaml"
    with open(config_file, "w") as f:
        yaml.dump({"nside": 16, "calibration": 0.98}, f)
    with pytest.raises(ValueError, match="calibration"):
        InputParams.read_parameter_file(str(config_file))


def test_calibration_unity_accepted_however_it_is_spelled():
    """A quoted ``calibration: "1.0"`` is still unity, and still accepted."""
    for spelling in (1.0, 1, "1.0", "1"):
        params = InputParams()
        params.update({"calibration": spelling})  # must not raise
        assert not hasattr(params, "calibration")


def test_calibration_bool_is_refused_not_read_as_unity():
    """``calibration: true`` must raise, not be silently read as 1.0.

    ``True == 1.0`` in Python, so a naive equality check would quietly accept a
    boolean as unity — the exact class of silent mis-parse this guard exists to
    prevent.
    """
    params = InputParams()
    with pytest.raises(ValueError, match="calibration"):
        params.update({"calibration": True})


def test_calibration_garbage_raises_the_removal_error():
    """A non-numeric value raises the removal message, not a stray TypeError."""
    params = InputParams()
    with pytest.raises(ValueError, match="no longer supported"):
        params.update({"calibration": "not-a-number"})
