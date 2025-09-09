"""
Tests for the CosmoForge CosmoLogger.
"""

import tempfile
from pathlib import Path

from cosmocore import CosmoLogger, get_logger


class TestLogger:
    """Test the CosmoLogger class."""

    def test_logger_creation(self):
        """Test that logger can be created."""
        logger = CosmoLogger("test_logger")
        assert logger.name == "test_logger"
        assert logger.logger.name == "test_logger"

    def test_get_logger_function(self):
        """Test the get_logger convenience function."""
        logger = get_logger("test_analysis", feedback_level=2)
        assert isinstance(logger, CosmoLogger)
        assert logger.name == "test_analysis"

    def test_log_levels(self):
        """Test different log levels work."""
        logger = CosmoLogger("test_levels", feedback_level=2)

        # These should not raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

    def test_level_setting(self):
        """Test changing log levels."""
        logger = CosmoLogger("test_set_level", feedback_level=1)
        logger.set_feedback_level(2)
        logger.set_feedback_level(3)

        # Should not raise exceptions
        logger.info("Test message")

    def test_file_logging(self):
        """Test logging to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            logger = CosmoLogger("test_file", log_file=log_file)

            logger.info("Test file logging")

            # Check file was created and contains content
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test file logging" in content
            assert "test_file" in content

    def test_logger_with_custom_format(self):
        """Test logger with custom format string."""
        logger = CosmoLogger("test_format", format_type="simple")

        # Should not raise exception
        logger.info("Custom format test")

    def test_feedback_level_logging(self):
        """Test feedback level based logging."""
        import io
        import sys

        # Test with feedback level 2 (verbose)
        logger = CosmoLogger("test_feedback", feedback_level=2)

        # Capture stdout
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            # Level 1 should be logged (INFO level)
            logger.log_with_feedback("Level 1 message", level=1)

            # Level 2 should be logged (DEBUG level)
            logger.log_with_feedback("Level 2 message", level=2)

            # Level 3 should NOT be logged (feedback=2, level=3)
            logger.log_with_feedback("Level 3 message", level=3)

            output = captured_output.getvalue()

            # Check that appropriate messages appear
            assert "Level 1 message" in output
            assert "Level 2 message" in output
            # Level 3 should not appear when feedback=2
            assert "Level 3 message" not in output

        finally:
            sys.stdout = old_stdout

    def test_timer_functionality(self):
        """Test the timer functionality."""
        import time

        logger = CosmoLogger("test_timer", feedback_level=3)

        # Test timer context manager
        with logger.timer("test operation") as timer:
            time.sleep(0.01)  # Sleep for 10ms

        # Timer should have recorded the operation
        assert timer.start_time is not None

    def test_scientific_logging_methods(self):
        """Test scientific context logging methods."""
        import numpy as np

        logger = CosmoLogger("test_scientific", feedback_level=2)

        # Test matrix logging
        test_matrix = np.random.rand(3, 3)
        logger.log_matrix_info("test matrix", test_matrix)

        # Test multipole range logging
        logger.log_multipole_range(2, 100)

        # Test timing log
        logger.log_timing("test operation", 1.234)

        # These should not raise exceptions

    def test_advanced_feedback_levels(self):
        """Test advanced feedback levels (3+)."""
        logger = CosmoLogger("test_advanced", feedback_level=5)

        # Test level 3 (VERY_VERBOSE)
        logger.log_with_feedback("Level 3 message", level=3)

        # Test level 4 (DEBUG_DETAILS)
        logger.log_with_feedback("Level 4 message", level=4)

        # Test level 5+ (ULTRA_VERBOSE)
        logger.log_with_feedback("Level 5 message", level=5)
        logger.log_with_feedback("Level 6 message", level=6)

    def test_timing_different_durations(self):
        """Test timing with different duration formats."""
        logger = CosmoLogger("test_timing", feedback_level=3)

        # Test microseconds
        logger.log_timing("microsecond operation", 0.0005)

        # Test milliseconds
        logger.log_timing("millisecond operation", 0.5)

        # Test seconds
        logger.log_timing("second operation", 5.0)

        # Test minutes
        logger.log_timing("minute operation", 125.0)

        # Test hours
        logger.log_timing("hour operation", 7200.0)

    def test_matrix_info_detailed(self):
        """Test matrix info with various matrix types and sizes."""
        import numpy as np

        logger = CosmoLogger("test_matrix_detail", feedback_level=5)

        # Test small matrix with detailed stats (level 5)
        small_matrix = np.random.rand(5, 5)
        logger.log_matrix_info("small matrix", small_matrix, level=5)

        # Test large matrix (no detailed stats)
        large_matrix = np.random.rand(200, 200)
        logger.log_matrix_info("large matrix", large_matrix, level=5)

        # Test complex matrix
        complex_matrix = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
        logger.log_matrix_info("complex matrix", complex_matrix, level=5)

        # Test different dtypes and memory sizes
        int_matrix = np.random.randint(0, 100, (10, 10), dtype=np.int32)
        logger.log_matrix_info("int matrix", int_matrix)

        # Test very small matrix (KB range)
        tiny_matrix = np.random.rand(2, 2)
        logger.log_matrix_info("tiny matrix", tiny_matrix)

        # Test medium matrix (MB range - between 1MB and 1024MB)
        class MediumMatrixMock:
            def __init__(self):
                self.shape = (500, 500)
                self.dtype = np.float64
                self.nbytes = 50 * 1024 * 1024  # 50MB
                self.size = 250000

        medium_mock = MediumMatrixMock()
        logger.log_matrix_info("medium MB matrix", medium_mock)

        # Test large matrix (GB range simulation)
        # Create a custom mock matrix object to simulate large size
        class LargeMatrixMock:
            def __init__(self):
                self.shape = (20000, 20000)
                self.dtype = np.float64
                self.nbytes = 2 * 1024**3  # 2GB
                self.size = 400000000

            def min(self):
                return 0.0

            def max(self):
                return 1.0

            def mean(self):
                return 0.5

        # Test that the GB formatting works
        import numpy as np

        large_mock = LargeMatrixMock()

        # Patch np.isrealobj to return True for our mock
        import unittest.mock

        with unittest.mock.patch("numpy.isrealobj", return_value=True):
            logger.log_matrix_info("large GB matrix", large_mock)

    def test_map_info_logging(self):
        """Test HEALPix map info logging."""
        logger = CosmoLogger("test_map", feedback_level=2)

        # Test with RMS
        logger.log_map_info("test_map_with_rms", nside=64, npix=49152, rms=25.5)

        # Test without RMS
        logger.log_map_info("test_map_no_rms", nside=128, npix=196608)

    def test_progress_logging(self):
        """Test progress logging."""
        logger = CosmoLogger("test_progress", feedback_level=2)

        # Test with operation name
        logger.log_progress(50, 100, "Processing data")

        # Test without operation name
        logger.log_progress(75, 100)

        # Test different percentages
        logger.log_progress(1, 1000, "Initialization")
        logger.log_progress(999, 1000, "Finalization")

    def test_timing_disabled(self):
        """Test logger with timing disabled."""
        logger = CosmoLogger("test_no_timing", feedback_level=3, enable_timing=False)

        # This should not log anything since timing is disabled
        logger.log_timing("test operation", 1.0)
