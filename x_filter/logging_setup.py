import logging
import sys
from typing import Optional

def setup_logging(debug_mode: bool = False) -> None:
    """
    Configure logging for the entire application.
    Should be called before importing any other application modules.
    
    Args:
        debug_mode (bool): If True, sets logging level to DEBUG, otherwise INFO
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(levelname)s ::: %(asctime)s ::: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    # Add console handler if none exists
    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Configure application logger
    app_logger = logging.getLogger("my_logger")
    app_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    # Configure numba logger
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance. Use this instead of logging.getLogger() directly.
    
    Args:
        name (Optional[str]): Logger name. If None, returns the app logger
    
    Returns:
        logging.Logger: Configured logger instance
    """
    if name is None:
        return logging.getLogger("my_logger")
    return logging.getLogger(name)