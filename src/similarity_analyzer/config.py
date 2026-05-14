"""
Configuration for the Similarity Analyzer.

Provides a validated dataclass that centralizes all analysis parameters,
including similarity thresholds, date filters, parallelization settings,
and text preprocessing options.
"""

import pandas as pd

from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class AnalysisConfig:
    """
    Configuration settings for similarity analysis.

    Controls the full analysis pipeline: similarity thresholds, date filtering,
    parallelization behavior, and text preprocessing. All date strings are
    converted to pandas Timestamps during initialization.

    Args:
        min_threshold: Minimum similarity score to include (0-100).
        max_threshold: Maximum similarity score to include (0-100).
        chunk_size: Number of string pairs per parallel work unit.
            Smaller chunks reduce memory but increase scheduling overhead.
        cores_count: Maximum CPU cores for parallel processing.
        words_to_delete: Domain-specific stopwords to strip from specifications
            before comparison (e.g. "urgent", "misc").
        ref_start_date: Start of the reference (baseline) date range.
        ref_end_date: End of the reference date range.
        comp_start_date: Start of the comparison (target) date range.
        comp_end_date: End of the comparison date range.
        backend: Joblib parallelization backend ('loky', 'multiprocessing',
            or 'threading').
        use_product_name: If True, group by product name instead of department.
        clean_specifications: If True, remove stopwords from spec strings.
        keep_letters_only: If True, strip all non-letter characters
            (numbers, symbols) from specs before comparison.
        save_perfect_matches: If True, collect 100% matches separately.
        product_type: Filter to 'repair', 'consumable', or 'All'.
        algorithm: Similarity algorithm — 'RapidFuzz' or 'SequenceMatcher'.

    Raises:
        ValueError: If thresholds, product_type, algorithm, or date
            ranges are invalid.

    Example:
        >>> config = AnalysisConfig(
        ...     min_threshold=80.0,
        ...     algorithm="RapidFuzz",
        ...     ref_start_date="2024-01-01",
        ...     ref_end_date="2024-12-31",
        ... )
    """

    # Similarity thresholds
    min_threshold: float = 70.0
    max_threshold: float = 100.0 - 1e-9

    # Parallelization
    chunk_size: int = 1_000
    cores_count: int = 8
    backend: str = "loky"

    # Text preprocessing
    words_to_delete: List[str] = field(default_factory=list)
    clean_specifications: bool = True
    keep_letters_only: bool = False

    # Date filters
    ref_start_date: Optional[str] = None
    ref_end_date: Optional[str] = None
    comp_start_date: Optional[str] = None
    comp_end_date: Optional[str] = None

    # Analysis options
    use_product_name: bool = False
    save_perfect_matches: bool = False
    product_type: str = "All"
    algorithm: str = "RapidFuzz"

    def __post_init__(self):
        """Validate configuration values and convert date strings."""
        self._validate_thresholds()
        self._validate_choices()
        self._validate_and_convert_dates()
        self._set_default_stopwords()

    def _validate_thresholds(self):
        """Ensure similarity thresholds are logically consistent."""
        if not (0 <= self.min_threshold <= 100):
            raise ValueError("min_threshold must be between 0 and 100")
        if not (0 <= self.max_threshold <= 100):
            raise ValueError("max_threshold must be between 0 and 100")
        if self.min_threshold > self.max_threshold:
            raise ValueError("min_threshold cannot exceed max_threshold")

    def _validate_choices(self):
        """Validate enum-like fields."""
        valid_product_types = ("repair", "consumable", "All")
        if self.product_type not in valid_product_types:
            raise ValueError(
                f"product_type must be one of {valid_product_types}, "
                f"got '{self.product_type}'"
            )

        valid_algorithms = ("RapidFuzz", "SequenceMatcher")
        if self.algorithm not in valid_algorithms:
            raise ValueError(
                f"algorithm must be one of {valid_algorithms}, "
                f"got '{self.algorithm}'"
            )

        valid_backends = ("loky", "multiprocessing", "threading")
        if self.backend not in valid_backends:
            raise ValueError(
                f"backend must be one of {valid_backends}, "
                f"got '{self.backend}'"
            )

    def _validate_and_convert_dates(self):
        """Validate date ordering and convert strings to Timestamps."""
        if self.ref_start_date and self.ref_end_date:
            if self.ref_start_date > self.ref_end_date:
                raise ValueError(
                    "Reference start date must be earlier than end date."
                )

        if self.comp_start_date and self.comp_end_date:
            if self.comp_start_date > self.comp_end_date:
                raise ValueError(
                    "Comparison start date must be earlier than end date."
                )

        self.ref_start_date = (
            pd.to_datetime(self.ref_start_date) if self.ref_start_date else None
        )
        self.ref_end_date = (
            pd.to_datetime(self.ref_end_date) if self.ref_end_date else None
        )
        self.comp_start_date = (
            pd.to_datetime(self.comp_start_date) if self.comp_start_date else None
        )
        self.comp_end_date = (
            pd.to_datetime(self.comp_end_date) if self.comp_end_date else None
        )

    def _set_default_stopwords(self):
        """Set default domain-specific stopwords if none provided."""
        if not self.words_to_delete:
            self.words_to_delete = [
                "special_equipment", "S/P", "misc", "urgent", "quote"
            ]
