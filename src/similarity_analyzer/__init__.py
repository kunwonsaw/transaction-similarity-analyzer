"""
Purchasing Transaction Similarity Analyzer
============================================
A tool for analyzing similar purchasing transactions
to help identify different types of discrepancies and fraudulent behavior.

Key Features:
- Load and process multiple Excel/CSV files
- RapidFuzz/SequenceMatcher-based similarity matching
- Joblib and Python-native multiprocessing support
- Date range filtering for temporal comparisons
- Separate extraction of perfect (100%) matches

Author: Kunwon Saw
"""

from .config import AnalysisConfig
from .analyzer import SimilarityAnalyzer
from .similarity import similarity_rapidfuzz, similarity_sequence_matcher
from .utils import setup_logger

__version__ = "0.1.0"

__all__ = [
    "AnalysisConfig",
    "SimilarityAnalyzer",
    "similarity_rapidfuzz",
    "similarity_sequence_matcher",
    "setup_logger",
]
