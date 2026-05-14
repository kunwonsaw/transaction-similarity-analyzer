"""
Utility functions for the Similarity Analyzer package.

Provides logging configuration, text cleaning, and batch processing helpers
used across the analysis pipeline.
"""

import os
import re
import regex
import logging

from typing import Optional, List, Generator


# ----------------------------------------------------------------------------
# LOGGING UTILITIES
# ----------------------------------------------------------------------------
def setup_logger(
    verbosity: int = 1,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger for the SimilarityAnalyzer module.

    Creates a logger with both console and optional file output, using a
    standardized format across all handlers. Automatically clears any existing
    handlers to prevent duplicate logging.

    Args:
        verbosity: Controls logging detail (default: 1)
            - 0: ERROR only (critical issues that prevent execution)
            - 1: INFO (general progress and key events)
            - 2+: DEBUG (detailed diagnostic information)
        log_file: Path to output log file. If None, logs only to console.
            File will be created/appended with UTF-8-SIG encoding.

    Returns:
        Configured logger instance named "SimilarityAnalyzer".

    Example:
        >>> logger = setup_logger(verbosity=2, log_file="analysis.log")
        >>> logger.debug("Processing started")

    Note:
        All modules should retrieve this logger via:
            logging.getLogger("SimilarityAnalyzer")
        This ensures consistent configuration across the package.
    """
    # Map verbosity to logging levels
    if verbosity <= 0:
        level = logging.ERROR
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    # Get/create logger and clear any existing handlers to prevent duplication
    logger = logging.getLogger("SimilarityAnalyzer")
    logger.setLevel(level)
    logger.handlers.clear()

    # Define consistent format for all handlers
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (always active)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # (Optional) File handler for persistent logs
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8-sig")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Suppress joblib verbose output to prevent cluttering logs
    os.environ.setdefault("JOBLIB_VERBOSE", "0")

    return logger


# Initialize default logger; reconfigured in main() if needed
setup_logger()
LOGGER = logging.getLogger("SimilarityAnalyzer")


# ----------------------------------------------------------------------------
# TEXT CLEANING
# ----------------------------------------------------------------------------
def extract_letters_only(text: str) -> str:
    """
    Extract only Unicode letters from text, removing numbers
    and special characters.

    Uses the \\p{L} Unicode property to support Korean, Chinese,
    and other non-Latin scripts.

    Args:
        text: Input string to clean.

    Returns:
        String containing only letter characters.

    Example:
        >>> extract_letters_only("ABC-123 가나다")
        'ABC가나다'
    """
    if not isinstance(text, str):
        return ""
    return regex.sub(r'[^\p{L}]', '', text)


def clean_supplier_name(supplier_name: str) -> str:
    """
    Remove common corporate suffixes from supplier/manufacturer names.

    Strips Korean and English corporate designators to normalize names
    for comparison purposes.

    Args:
        supplier_name: Raw supplier or manufacturer name.

    Returns:
        Cleaned name with corporate suffixes removed.

    Example:
        >>> clean_supplier_name("Samsung Co. Inc.")
        'Samsung'
        >>> clean_supplier_name("삼성전자(주)")
        '삼성전자'
    """
    if not isinstance(supplier_name, str):
        return supplier_name

    suffixes = [r"\(주\)", r"주식회사", r"Co\.", r"Inc\.", r"LLC", r"Limited"]
    cleaned = supplier_name
    for suffix in suffixes:
        cleaned = re.sub(suffix, "", cleaned)

    return cleaned.strip()


# ----------------------------------------------------------------------------
# BATCH PROCESSING
# ----------------------------------------------------------------------------
def create_batches(
    generator: Generator,
    batch_size: int
) -> Generator[List, None, None]:
    """
    Group items from a generator into fixed-size batches.

    Useful for chunking large pair generators before distributing
    across parallel workers.

    Args:
        generator: Input iterable to batch.
        batch_size: Maximum number of items per batch.

    Yields:
        Lists of up to batch_size items.

    Example:
        >>> list(create_batches(range(7), batch_size=3))
        [[0, 1, 2], [3, 4, 5], [6]]
    """
    batch = []
    for item in generator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
