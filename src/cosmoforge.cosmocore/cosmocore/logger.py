"""
Enhanced logging utilities for CosmoForge with scientific formatting and feedback
integration.

This module provides a comprehensive logging system designed for cosmological analysis,
including timing utilities, scientific context logging, and seamless integration
with the existing feedback system.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np

# Feedback to logging level mapping
FEEDBACK_TO_LOG_LEVEL = {
    0: logging.CRITICAL + 10,  # Silent (above CRITICAL)
    1: logging.INFO,  # Normal output
    2: logging.DEBUG,  # Verbose output
    3: logging.DEBUG - 5,  # Very verbose (custom level)
    4: logging.DEBUG - 10,  # Debug details (custom level)
    5: logging.DEBUG - 15,  # Ultra-verbose (custom level)
}

# Define custom log levels
VERY_VERBOSE = logging.DEBUG - 5
DEBUG_DETAILS = logging.DEBUG - 10
ULTRA_VERBOSE = logging.DEBUG - 15

# Add custom levels to logging
logging.addLevelName(VERY_VERBOSE, "VERBOSE")
logging.addLevelName(DEBUG_DETAILS, "DETAILS")
logging.addLevelName(ULTRA_VERBOSE, "ULTRA")


class Timer:
    """Context manager for timing operations with automatic logging."""

    def __init__(self, logger: CosmoLogger, operation: str, level: int = 1):
        """
        Initialize timer context manager.

        Parameters
        ----------
        logger : CosmoLogger
            CosmoLogger instance to use for timing output
        operation : str
            Description of the operation being timed
        level : int
            Feedback level for timing output (default: 1)
        """
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        self.logger.log_with_feedback(f"Starting {self.operation}...", self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log duration."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.logger.log_timing(self.operation, duration, self.level)


class CosmoLogger:
    """
    Enhanced logger for CosmoForge with feedback integration and scientific context.

    This logger provides seamless integration with the existing feedback system
    while adding professional logging capabilities including file output, timing,
    and scientific context logging for cosmological analysis.
    """

    def __init__(
        self,
        name: str = "cosmoforge",
        feedback_level: int = 1,
        log_file: str | Path | None = None,
        format_type: str = "scientific",
        enable_timing: bool = True,
    ):
        """
        Initialize the enhanced cosmological logger.

        Parameters
        ----------
        name : str, default="cosmoforge"
            CosmoLogger name
        feedback_level : int, default=1
            Feedback level (0=silent, 1=normal, 2=verbose, 3+=debug)
        log_file : str or Path, optional
            Path to log file. If None, logs to console only.
        format_type : str, default="scientific"
            Format type: "scientific" or "simple"
        enable_timing : bool, default=True
            Whether to enable timing functionality
        """
        self.name = name
        self.feedback_level = feedback_level
        self.enable_timing = enable_timing

        # Convert feedback level to logging level
        self.log_level = FEEDBACK_TO_LOG_LEVEL.get(feedback_level, logging.INFO)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(min(self.log_level, logging.DEBUG - 15))

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Set format based on type
        if format_type == "scientific":
            format_string = "%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s"
        else:
            format_string = "%(levelname)s: %(message)s"

        formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

        # Console handler - create a custom handler that respects stdout redirection
        class DynamicStreamHandler(logging.StreamHandler):
            """StreamHandler that dynamically uses current sys.stdout."""

            def __init__(self):
                # Don't call super().__init__() with a stream argument
                super().__init__()

            @property
            def stream(self):
                """Always use the current sys.stdout."""
                return sys.stdout

            @stream.setter
            def stream(self, value):
                """Ignore attempts to set the stream."""
                pass

        console_handler = DynamicStreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG - 15)  # Log everything to file
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_with_feedback(self, message: str, level: int = 1):
        """
        Log message using feedback level system (backward compatibility).

        Parameters
        ----------
        message : str
            Message to log
        level : int, default=1
            Feedback level (1=normal, 2=verbose, 3+=debug)
        """
        if self.feedback_level >= level:
            # Map feedback level to appropriate logging method
            if level == 1:
                self.logger.info(message)
            elif level == 2:
                self.logger.debug(message)
            elif level == 3:
                self.logger.log(VERY_VERBOSE, message)
            elif level == 4:
                self.logger.log(DEBUG_DETAILS, message)
            elif level >= 5:
                self.logger.log(ULTRA_VERBOSE, message)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)

    def set_feedback_level(self, level: int):
        """Change the feedback level."""
        self.feedback_level = level
        self.log_level = FEEDBACK_TO_LOG_LEVEL.get(level, logging.INFO)
        for handler in self.logger.handlers:
            if (
                isinstance(handler, logging.StreamHandler)
                and handler.stream == sys.stdout
            ):
                handler.setLevel(self.log_level)

    def _format_memory_size(self, size_bytes: int) -> str:
        """
        Format memory size in human-readable format.

        Parameters
        ----------
        size_bytes : int
            Size in bytes

        Returns
        -------
        str
            Formatted memory string (e.g., "1.2 MB", "512.0 KB")
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    # Scientific context logging methods
    def log_timing(self, operation: str, duration: float, level: int = 1):
        """
        Log operation timing with scientific formatting.

        Parameters
        ----------
        operation : str
            Description of the operation
        duration : float
            Duration in seconds
        level : int, default=1
            Feedback level for timing output
        """
        if self.enable_timing:
            if duration < 1e-3:
                time_str = f"{duration * 1e6:.1f} μs"
            elif duration < 1:
                time_str = f"{duration * 1e3:.1f} ms"
            elif duration < 60:
                time_str = f"{duration:.2f} s"
            elif duration < 3600:
                minutes = int(duration // 60)
                seconds = duration % 60
                time_str = f"{minutes}m {seconds:.1f}s"
            else:
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = duration % 60
                time_str = f"{hours}h {minutes}m {seconds:.1f}s"

            self.log_with_feedback(f"{operation} completed in {time_str}", level)

    def log_matrix_info(self, name: str, matrix: np.ndarray, level: int = 1):
        """
        Log matrix information with scientific context.

        Parameters
        ----------
        name : str
            Matrix name/description
        matrix : numpy.ndarray
            Matrix to describe
        level : int, default=1
            Feedback level for matrix info
        """
        shape_str = "×".join(str(s) for s in matrix.shape)
        dtype_str = str(matrix.dtype)

        # Memory usage
        memory_str = self._format_memory_size(matrix.nbytes)

        self.log_with_feedback(
            f"{name}: shape={shape_str}, dtype={dtype_str}, memory={memory_str}", level
        )

        # Additional statistics for small matrices (debugging information)
        if self.feedback_level >= 3 and matrix.size < 10000:
            if np.isrealobj(matrix):
                self.log_with_feedback(
                    f"{name} stats: min={matrix.min():.3e}, "
                    f"max={matrix.max():.3e}, mean={matrix.mean():.3e}",
                    3,
                )

    def log_multipole_range(self, lmin: int, lmax: int, level: int = 1):
        """
        Log multipole range in cosmological notation.

        Parameters
        ----------
        lmin, lmax : int
            Minimum and maximum multipole moments
        level : int, default=1
            Feedback level for multipole info
        """
        self.log_with_feedback(f"Multipole range: ℓ ∈ [{lmin}, {lmax}]", level)

    def log_map_info(
        self,
        map_name: str,
        nside: int,
        npix: int,
        rms: float | None = None,
        level: int = 1,
    ):
        """
        Log HEALPix map information.

        Parameters
        ----------
        map_name : str
            Name/description of the map
        nside : int
            HEALPix nside parameter
        npix : int
            Number of pixels
        rms : float, optional
            RMS value of the map
        level : int, default=1
            Feedback level for map info
        """
        resolution_arcmin = hp.nside2resol(nside, arcmin=True)

        # Basic memory information only (level 1) - similar to matrix memory info
        # Calculate approximate memory usage for HEALPix map (assuming float64)
        memory_bytes = npix * 8  # 8 bytes per float64
        memory_str = self._format_memory_size(memory_bytes)

        self.log_with_feedback(f"{map_name}: HEALPix map, memory≈{memory_str}", level)

        # Detailed information (level 3 - debugging)
        if self.feedback_level >= 3:
            self.log_with_feedback(f"{map_name} nside: {nside}", 3)
            self.log_with_feedback(f"{map_name} npix: {npix:,}", 3)
            self.log_with_feedback(f"{map_name} resolution: ≈{resolution_arcmin:.1f}'", 3)

        # Statistical information (level 3 - debugging)
        if rms is not None and self.feedback_level >= 3:
            self.log_with_feedback(f"{map_name} RMS: {rms:.2f} μK", 3)

    def log_progress(self, current: int, total: int, operation: str = "", level: int = 1):
        """
        Log progress information.

        Parameters
        ----------
        current : int
            Current step/iteration
        total : int
            Total steps/iterations
        operation : str, optional
            Description of the operation
        level : int, default=1
            Feedback level for progress info
        """
        percentage = (current / total) * 100
        prefix = f"{operation} " if operation else ""
        self.log_with_feedback(
            f"{prefix}progress: {current}/{total} ({percentage:.1f}%)", level
        )

    def timer(self, operation: str, level: int = 1) -> Timer:
        """
        Create a timing context manager.

        Parameters
        ----------
        operation : str
            Description of the operation to time
        level : int, default=1
            Feedback level for timing output

        Returns
        -------
        Timer
            Context manager for timing

        Examples
        --------
        >>> logger = CosmoLogger("analysis")
        >>> with logger.timer("matrix inversion"):
        ...     result = np.linalg.inv(matrix)
        """
        return Timer(self, operation, level)


def get_logger(
    name: str = "cosmoforge",
    feedback_level: int = 1,
    log_file: str | Path | None = None,
    format_type: str = "scientific",
    enable_timing: bool = True,
) -> CosmoLogger:
    """
    Get a configured logger instance with feedback integration.

    Parameters
    ----------
    name : str, default="cosmoforge"
        CosmoLogger name
    feedback_level : int, default=1
        Feedback level (0=silent, 1=normal, 2=verbose, 3+=debug)
    log_file : str or Path, optional
        Path to log file
    format_type : str, default="scientific"
        Format type: "scientific" or "simple"
    enable_timing : bool, default=True
        Whether to enable timing functionality

    Returns
    -------
    CosmoLogger
        Configured logger instance

    Examples
    --------
    >>> logger = get_logger("fisher_analysis", feedback_level=2)
    >>> logger.info("Starting Fisher matrix computation")
    >>> with logger.timer("matrix computation"):
    ...     fisher_matrix = compute_fisher()
    >>> logger.log_matrix_info("Fisher matrix", fisher_matrix)
    """
    return CosmoLogger(
        name=name,
        feedback_level=feedback_level,
        log_file=log_file,
        format_type=format_type,
        enable_timing=enable_timing,
    )


def get_logger_from_params(params, name: str = "cosmoforge") -> CosmoLogger:
    """
    Create logger from CosmoForge parameters object.

    Parameters
    ----------
    params : InputParams
        CosmoForge parameters object
    name : str, default="cosmoforge"
        CosmoLogger name

    Returns
    -------
    CosmoLogger
        Configured logger instance
    """
    feedback_level = getattr(params, "feedback", 1)
    log_file = getattr(params, "log_file", None)
    log_format = getattr(params, "log_format", "scientific")
    enable_timing = getattr(params, "log_timing", True)

    return get_logger(
        name=name,
        feedback_level=feedback_level,
        log_file=log_file,
        format_type=log_format,
        enable_timing=enable_timing,
    )
