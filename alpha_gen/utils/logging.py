"""
Centralized logging configuration for the alpha generator.
Provides consistent logging across all modules.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
import datetime

# Default log format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

# Default log directory
DEFAULT_LOG_DIR = './logs'

def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    log_dir: str = DEFAULT_LOG_DIR,
    log_format: str = DEFAULT_LOG_FORMAT,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name (default: generated based on date)
        log_dir: Directory for log files
        log_format: Log message format
        max_file_size: Maximum size in bytes for each log file
        backup_count: Number of backup log files to keep
        
    Returns:
        Root logger
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'alpha_gen_{timestamp}.log'
    
    log_path = os.path.join(log_dir, log_file)
    
    # Convert string log level to constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Initial log message
    root_logger.info(f"Logging initialized at level {log_level}")
    root_logger.info(f"Log file: {log_path}")
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """
    Get logger for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def log_exception(
    logger: logging.Logger,
    exc: Exception,
    message: str = "An exception occurred"
) -> None:
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        exc: Exception to log
        message: Additional message
    """
    logger.error(f"{message}: {type(exc).__name__}: {str(exc)}", exc_info=True)