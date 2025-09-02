"""
Tests for the CosmoForge Logger.
"""

import tempfile
from pathlib import Path

from cosmocore import Logger, get_logger


class TestLogger:
    """Test the Logger class."""

    def test_logger_creation(self):
        """Test that logger can be created."""
        logger = Logger("test_logger")
        assert logger.name == "test_logger"
        assert logger.logger.name == "test_logger"

    def test_get_logger_function(self):
        """Test the get_logger convenience function."""
        logger = get_logger("test_analysis", level="DEBUG")
        assert isinstance(logger, Logger)
        assert logger.name == "test_analysis"

    def test_log_levels(self):
        """Test different log levels work."""
        logger = Logger("test_levels", level="DEBUG")

        # These should not raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

    def test_level_setting(self):
        """Test changing log levels."""
        logger = Logger("test_set_level", level="INFO")
        logger.set_level("DEBUG")
        logger.set_level(30)  # WARNING level

        # Should not raise exceptions
        logger.info("Test message")

    def test_file_logging(self):
        """Test logging to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            logger = Logger("test_file", log_file=log_file)

            logger.info("Test file logging")

            # Check file was created and contains content
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test file logging" in content
            assert "test_file" in content

    def test_logger_with_custom_format(self):
        """Test logger with custom format string."""
        custom_format = "%(levelname)s: %(message)s"
        logger = Logger("test_format", format_string=custom_format)

        # Should not raise exception
        logger.info("Custom format test")
