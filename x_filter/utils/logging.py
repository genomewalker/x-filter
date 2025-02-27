# x_filter/utils/logging.py

import logging
import os
import sys
import time
from typing import Optional, Dict, Any
import traceback
from pathlib import Path

# Configure log levels with custom level for debugging
DEBUG_VERBOSE = 5  # More detailed than DEBUG
logging.addLevelName(DEBUG_VERBOSE, "TRACE")

# Default format strings
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
DEBUG_LOG_FORMAT = (
    "%(asctime)s [%(levelname)8s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
)


class XFilterLogger(logging.Logger):
    """Custom logger with enhanced functionality for xFilter."""

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self.metrics: Dict[str, Dict[str, Any]] = {}

    def trace(self, msg, *args, **kwargs):
        """Log at TRACE level (more detailed than DEBUG)."""
        if self.isEnabledFor(DEBUG_VERBOSE):
            self._log(DEBUG_VERBOSE, msg, args, **kwargs)

    def start_timer(self, name: str) -> None:
        """Start a timer for performance tracking."""
        if name not in self.metrics:
            self.metrics[name] = {}
        self.metrics[name]["start_time"] = time.time()
        self.debug(f"Started timer: {name}")

    def end_timer(self, name: str) -> float:
        """End a timer and return elapsed time."""
        if name in self.metrics and "start_time" in self.metrics[name]:
            elapsed = time.time() - self.metrics[name]["start_time"]
            self.metrics[name]["elapsed"] = elapsed
            self.debug(f"Ended timer: {name} - {elapsed:.3f}s")
            return elapsed
        return 0.0

    def record_metric(self, name: str, value: Any) -> None:
        """Record a custom metric."""
        if name not in self.metrics:
            self.metrics[name] = {}
        self.metrics[name]["value"] = value
        self.debug(f"Recorded metric: {name} = {value}")

    def dump_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Dump all recorded metrics."""
        return self.metrics


# Register the custom logger class
logging.setLoggerClass(XFilterLogger)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    log_format: Optional[str] = None,
) -> None:
    """
    Set up logging configuration.

    Args:
        level: Log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console: Whether to log to console
        log_format: Optional custom log format
    """
    # Convert string level to numeric level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        if level.upper() == "TRACE":
            numeric_level = DEBUG_VERBOSE
        else:
            numeric_level = logging.INFO

    # Set root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Determine log format based on level
    if log_format is None:
        log_format = (
            DEBUG_LOG_FORMAT if numeric_level <= logging.DEBUG else DEFAULT_LOG_FORMAT
        )

    formatter = logging.Formatter(log_format)

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if requested
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Configure third-party loggers to reduce noise
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> XFilterLogger:
    """
    Get a logger instance.

    Args:
        name: Logger name (defaults to module name)

    Returns:
        Logger instance
    """
    if name is None:
        # Get the name of the calling module
        frame = sys._getframe(1)
        name = frame.f_globals.get("__name__", "xfilter")

    return logging.getLogger(name)


def log_exception(
    logger: XFilterLogger, exception: Exception, message: str = "An error occurred"
) -> None:
    """
    Log an exception with traceback.

    Args:
        logger: Logger instance
        exception: Exception object
        message: Custom message
    """
    logger.error(f"{message}: {str(exception)}")
    logger.debug(f"Exception details: {traceback.format_exc()}")


class LogContext:
    """Context manager for timed logging blocks."""

    def __init__(self, logger: XFilterLogger, message: str, level: str = "info"):
        self.logger = logger
        self.message = message
        self.level = level.lower()
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        log_method = getattr(self.logger, self.level)
        log_method(f"Starting: {self.message}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type is not None:
            self.logger.error(f"Failed: {self.message} - {elapsed:.3f}s")
            return False

        log_method = getattr(self.logger, self.level)
        log_method(f"Completed: {self.message} - {elapsed:.3f}s")
        return True
