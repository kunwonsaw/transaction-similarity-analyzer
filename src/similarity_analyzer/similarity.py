"""
Similarity computation functions and parallel processing workers.

Provides string similarity algorithms (RapidFuzz, SequenceMatcher) and
chunked worker functions designed for use with Joblib and native
multiprocessing pipelines.
"""

import logging

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from difflib import SequenceMatcher

from .utils import extract_letters_only


LOGGER = logging.getLogger("SimilarityAnalyzer")


# ----------------------------------------------------------------------------
# SIMILARITY ALGORITHMS
# ----------------------------------------------------------------------------
def similarity_sequence_matcher(text1: str, text2: str) -> float:
    """
    Calculate similarity using Python's built-in SequenceMatcher.

    Uses the Ratcliff/Obershelp algorithm to find the longest common
    subsequences between two strings.

    Args:
        text1: First string to compare.
        text2: Second string to compare.

    Returns:
        Similarity score from 0.0 to 100.0.

    Example:
        >>> similarity_sequence_matcher("bearing 6205", "bearing 6206")
        83.33...
    """
    str1 = str(text1) if text1 else ""
    str2 = str(text2) if text2 else ""
    return SequenceMatcher(None, str1, str2).ratio() * 100.0


def similarity_rapidfuzz(text1: str, text2: str) -> float:
    """
    Calculate similarity using a multi-metric RapidFuzz average.

    Combines four complementary metrics to produce a robust score:
    - fuzz.ratio: Character-level Levenshtein similarity
    - fuzz.partial_ratio: Best partial substring match
    - fuzz.token_sort_ratio: Order-independent token comparison
    - fuzz.QRatio: Quick ratio approximation

    This averaging approach reduces sensitivity to any single metric's
    weaknesses (e.g. partial_ratio inflating short substring matches).

    Args:
        text1: First string to compare.
        text2: Second string to compare.

    Returns:
        Averaged similarity score from 0.0 to 100.0.

    Example:
        >>> similarity_rapidfuzz("motor oil 5W-30", "5W-30 motor oil")
        97.5
    """
    str1 = str(text1) if text1 else ""
    str2 = str(text2) if text2 else ""

    scores = [
        fuzz.ratio(str1, str2),
        fuzz.partial_ratio(str1, str2),
        fuzz.token_sort_ratio(str1, str2),
        fuzz.QRatio(str1, str2),
    ]
    return float(np.mean(scores))


def get_similarity_function(algorithm_name: str):
    """
    Return a similarity function by algorithm name.

    Args:
        algorithm_name: One of 'RapidFuzz' or 'SequenceMatcher'.

    Returns:
        Callable that accepts (str, str) and returns a float score.

    Raises:
        ValueError: If algorithm_name is not recognized.
    """
    function_map = {
        "SequenceMatcher": similarity_sequence_matcher,
        "RapidFuzz": similarity_rapidfuzz,
    }
    if algorithm_name not in function_map:
        raise ValueError(
            f"Unsupported algorithm: {algorithm_name}. "
            f"Choose from: {list(function_map.keys())}"
        )
    return function_map[algorithm_name]


# ----------------------------------------------------------------------------
# PARALLEL PROCESSING WORKERS
#
# These functions are designed to be picklable top-level callables for use
# with joblib.Parallel and multiprocessing.Pool. They must remain at module
# scope (not as methods or lambdas) to satisfy Python's pickle requirements.
# ----------------------------------------------------------------------------
def process_regular_similarity_chunk(
    chunk,
    min_threshold: float,
    max_threshold: float,
    algorithm: str = "RapidFuzz",
):
    """
    Score a chunk of string pairs and return those within the threshold range.

    Used by the Joblib pipeline where pairs are pre-generated from
    deduplicated composite field values.

    Args:
        chunk: List of (string1, string2) tuples to compare.
        min_threshold: Minimum similarity score to include (inclusive).
        max_threshold: Maximum similarity score to include (inclusive).
            Perfect 100.0 matches are always excluded here.
        algorithm: Similarity algorithm name.

    Returns:
        List of [string1, string2, score] for matches within range.
    """
    similarity_func = get_similarity_function(algorithm)
    results = []

    for spec1, spec2 in chunk:
        score = similarity_func(spec1, spec2)
        if min_threshold <= score <= max_threshold and score != 100:
            results.append([spec1, spec2, score])

    return results


def process_pair_chunk(args):
    """
    Score a chunk of full record pairs for the native multiprocessing pipeline.

    Unlike process_regular_similarity_chunk, this operates on complete record
    dictionaries (not just spec strings), enabling metadata extraction
    (prices, suppliers, dates) in a single pass.

    Args:
        args: Tuple of (pair_chunk, min_threshold, max_threshold,
              save_perfect, letters_only, similarity_func) where:
            - pair_chunk: List of (record_dict1, record_dict2) pairs
            - min_threshold: Minimum similarity score (inclusive)
            - max_threshold: Maximum similarity score (inclusive)
            - save_perfect: Whether to collect 100% matches separately
            - letters_only: Whether to strip non-letter characters before comparison
            - similarity_func: Pre-resolved similarity callable

    Returns:
        Tuple of (regular_results, perfect_results) where each is a list
        of record arrays matching the SimilarityAnalyzer.RESULT_COLUMNS schema.
    """
    (
        pair_chunk,
        min_threshold,
        max_threshold,
        save_perfect,
        letters_only,
        similarity_func,
    ) = args

    regular_results = []
    perfect_results = []

    for row1, row2 in pair_chunk:
        # Skip identical items
        if row1.get("order_item_number") == row2.get("order_item_number"):
            continue

        # Extract and optionally clean specifications
        spec1 = (
            extract_letters_only(row1["main_spec"])
            if letters_only
            else row1["main_spec"]
        )
        spec2 = (
            extract_letters_only(row2["main_spec"])
            if letters_only
            else row2["main_spec"]
        )

        # Calculate similarity
        score = similarity_func(spec1, spec2)

        # Classify match
        if score == 100.0 and not save_perfect:
            continue

        if score == 100.0:
            match_type = "perfect"
        elif min_threshold <= score <= max_threshold:
            match_type = "in_range"
        else:
            continue

        # Extract numeric fields with safe conversion
        unit_price1 = pd.to_numeric(row1.get("unit_price", 0), errors="coerce")
        unit_price2 = pd.to_numeric(row2.get("unit_price", 0), errors="coerce")
        quantity1 = pd.to_numeric(row1.get("order_quantity", 0), errors="coerce")
        quantity2 = pd.to_numeric(row2.get("order_quantity", 0), errors="coerce")

        total1 = (
            unit_price1 * quantity1
            if pd.notnull(unit_price1) and pd.notnull(quantity1)
            else 0
        )
        total2 = (
            unit_price2 * quantity2
            if pd.notnull(unit_price2) and pd.notnull(quantity2)
            else 0
        )

        unit_price_diff = (
            abs(unit_price1 - unit_price2)
            if pd.notnull(unit_price1) and pd.notnull(unit_price2)
            else None
        )

        record = [
            spec1, spec2, score,
            unit_price1, unit_price2,
            row1.get("supplier_name"), row2.get("supplier_name"),
            row1.get("manufacturer_name"), row2.get("manufacturer_name"),
            row1.get("order_item_number"), row2.get("order_item_number"),
            quantity1, quantity2,
            total1, total2,
            unit_price_diff,
            abs(total1 - total2),
            row1.get("final_date"), row2.get("final_date"),
        ]

        if match_type == "perfect":
            perfect_results.append(record)
        else:
            regular_results.append(record)

    return regular_results, perfect_results
