"""
Test suite for the Similarity Analyzer package.

Covers:
- Utility functions (text cleaning, batching)
- Configuration validation
- Similarity algorithms
- End-to-end pipeline with synthetic data
"""

import pytest
import pandas as pd
import numpy as np

from similarity_analyzer.utils import (
    extract_letters_only,
    clean_supplier_name,
    create_batches,
)
from similarity_analyzer.config import AnalysisConfig
from similarity_analyzer.similarity import (
    similarity_rapidfuzz,
    similarity_sequence_matcher,
    get_similarity_function,
    process_regular_similarity_chunk,
)
from similarity_analyzer.analyzer import SimilarityAnalyzer


# ----------------------------------------------------------------------------
# UTILS
# ----------------------------------------------------------------------------
class TestExtractLettersOnly:
    def test_removes_numbers_and_symbols(self):
        assert extract_letters_only("ABC-123 def!") == "ABCdef"

    def test_preserves_korean(self):
        assert extract_letters_only("삼성전자(주)123") == "삼성전자주"

    def test_empty_string(self):
        assert extract_letters_only("") == ""

    def test_non_string_input(self):
        assert extract_letters_only(None) == ""
        assert extract_letters_only(12345) == ""


class TestCleanSupplierName:
    def test_removes_korean_suffix(self):
        assert clean_supplier_name("삼성전자(주)") == "삼성전자"

    def test_removes_english_suffix(self):
        assert clean_supplier_name("Samsung Co. Inc.") == "Samsung"

    def test_removes_llc(self):
        assert clean_supplier_name("Hanwha Solutions LLC") == "Hanwha Solutions"

    def test_non_string_passthrough(self):
        assert clean_supplier_name(None) is None

    def test_no_suffix(self):
        assert clean_supplier_name("SKF") == "SKF"


class TestCreateBatches:
    def test_even_split(self):
        result = list(create_batches(range(6), batch_size=3))
        assert result == [[0, 1, 2], [3, 4, 5]]

    def test_remainder_batch(self):
        result = list(create_batches(range(7), batch_size=3))
        assert result == [[0, 1, 2], [3, 4, 5], [6]]

    def test_single_batch(self):
        result = list(create_batches(range(3), batch_size=10))
        assert result == [[0, 1, 2]]

    def test_empty_input(self):
        result = list(create_batches(iter([]), batch_size=5))
        assert result == []


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
class TestAnalysisConfig:
    def test_default_values(self):
        config = AnalysisConfig()
        assert config.min_threshold == 70.0
        assert config.algorithm == "RapidFuzz"
        assert config.backend == "loky"
        assert len(config.words_to_delete) > 0

    def test_invalid_threshold_range(self):
        with pytest.raises(ValueError, match="min_threshold cannot exceed"):
            AnalysisConfig(min_threshold=90.0, max_threshold=50.0)

    def test_invalid_threshold_bounds(self):
        with pytest.raises(ValueError):
            AnalysisConfig(min_threshold=-1.0)

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError, match="algorithm"):
            AnalysisConfig(algorithm="InvalidAlgo")

    def test_invalid_product_type(self):
        with pytest.raises(ValueError, match="product_type"):
            AnalysisConfig(product_type="unknown")

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="backend"):
            AnalysisConfig(backend="invalid_backend")

    def test_date_conversion(self):
        config = AnalysisConfig(ref_start_date="2024-01-01")
        assert isinstance(config.ref_start_date, pd.Timestamp)

    def test_invalid_date_order(self):
        with pytest.raises(ValueError, match="earlier"):
            AnalysisConfig(ref_start_date="2025-01-01", ref_end_date="2024-01-01")


# ----------------------------------------------------------------------------
# SIMILARITY FUNCTIONS
# ----------------------------------------------------------------------------
class TestSimilarityFunctions:
    def test_rapidfuzz_identical_strings(self):
        score = similarity_rapidfuzz("bearing 6205", "bearing 6205")
        assert score == 100.0

    def test_rapidfuzz_different_strings(self):
        score = similarity_rapidfuzz("bearing 6205", "motor oil 5W-30")
        assert score < 50.0

    def test_rapidfuzz_similar_strings(self):
        score = similarity_rapidfuzz("bearing 6205 sealed", "bearing 6206 sealed")
        assert 70.0 < score < 100.0

    def test_sequence_matcher_identical(self):
        score = similarity_sequence_matcher("test string", "test string")
        assert score == 100.0

    def test_sequence_matcher_empty(self):
        score = similarity_sequence_matcher("", "")
        assert score == 100.0 # SequenceMatcher does treat two empty strings as identical

    def test_get_similarity_function_valid(self):
        func = get_similarity_function("RapidFuzz")
        assert callable(func)

    def test_get_similarity_function_invalid(self):
        with pytest.raises(ValueError, match="Unsupported"):
            get_similarity_function("BadAlgorithm")


class TestProcessRegularSimilarityChunk:
    def test_returns_matches_in_range(self):
        chunk = [
            ("bearing 6205 sealed", "bearing 6206 sealed"),
            ("motor oil", "hydraulic pump"),
        ]
        results = process_regular_similarity_chunk(
            chunk, min_threshold=70.0, max_threshold=99.9
        )
        # First pair should match, second should not
        assert len(results) >= 1
        assert all(70.0 <= r[2] <= 99.9 for r in results)

    def test_excludes_perfect_matches(self):
        chunk = [("identical", "identical")]
        results = process_regular_similarity_chunk(
            chunk, min_threshold=0.0, max_threshold=100.0
        )
        assert len(results) == 0

    def test_empty_chunk(self):
        results = process_regular_similarity_chunk(
            [], min_threshold=70.0, max_threshold=99.9
        )
        assert results == []


# ----------------------------------------------------------------------------
# END-TO-END PIPELINE
# ----------------------------------------------------------------------------
class TestSimilarityAnalyzerPipeline:
    """Integration tests using synthetic data."""

    @pytest.fixture
    def sample_df(self):
        """Create a minimal synthetic DataFrame."""
        return pd.DataFrame({
            "order_item_number": [f"PO-{i:03d}" for i in range(20)],
            "main_spec": [
                "Bearing 6205-2RS sealed radial",
                "Bearing 6206-ZZ shielded radial",
                "Hydraulic pump motor 15kW 3-phase",
                "Hydraulic pump motor 18.5kW 3-phase",
                "V-Belt A68 industrial grade",
                "V-Belt A68 heavy-duty grade",
                "Filter element HF-6700 10 micron",
                "Filter element HF-6710 20 micron",
                "Gearbox oil SAE 90 20L drum",
                "Gearbox oil SAE 140 20L drum",
                "Solenoid valve 24VDC 3/4 inch",
                "Coupling flexible jaw L-100",
                "Seal kit cylinder 80x50mm",
                "Conveyor roller 600mm steel",
                "Thermocouple type K 300mm probe",
                "Bearing 6205-2RS sealed radial",
                "Motor oil 5W-30 synthetic",
                "Motor oil 10W-40 synthetic",
                "Pump seal kit 100x70mm",
                "Conveyor roller 600mm stainless",
            ],
            "department_name": ["Maintenance"] * 20,
            "supplier_name": ["Supplier A", "Supplier B"] * 10,
            "manufacturer_name": ["MFG X", "MFG Y"] * 10,
            "unit_price": [10000 + i * 500 for i in range(20)],
            "order_quantity": [i + 1 for i in range(20)],
            "total": [(10000 + i * 500) * (i + 1) for i in range(20)],
            "cost_type": ["repair"] * 10 + ["consumable"] * 10,
            "delivery_completion_date": pd.date_range("2024-01-01", periods=20, freq="15D"),
        })

    @pytest.fixture
    def sample_csv(self, sample_df, tmp_path):
        """Save sample data to a temporary CSV."""
        csv_path = tmp_path / "test_data.csv"
        sample_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return str(csv_path)

    def test_load_csv(self, sample_csv):
        config = AnalysisConfig()
        analyzer = SimilarityAnalyzer(config)
        analyzer.load_csv_data(sample_csv)
        assert analyzer.raw_data is not None
        assert len(analyzer.raw_data) == 20

    def test_load_csv_missing_file(self):
        config = AnalysisConfig()
        analyzer = SimilarityAnalyzer(config)
        with pytest.raises(FileNotFoundError):
            analyzer.load_csv_data("/nonexistent/path.csv")

    def test_preprocess_creates_expected_columns(self, sample_csv):
        config = AnalysisConfig(clean_specifications=True)
        analyzer = SimilarityAnalyzer(config)
        analyzer.load_csv_data(sample_csv)
        analyzer.preprocess()

        assert "final_date" in analyzer.processed_data.columns
        assert "department_name+main_spec" in analyzer.processed_data.columns
        assert "unique_row_id" in analyzer.processed_data.columns
        assert "supplier_name_cleaned" in analyzer.processed_data.columns

    def test_preprocess_requires_loaded_data(self):
        config = AnalysisConfig()
        analyzer = SimilarityAnalyzer(config)
        with pytest.raises(ValueError, match="No data loaded"):
            analyzer.preprocess()

    def test_full_pipeline_produces_results(self, sample_csv):
        config = AnalysisConfig(
            min_threshold=70.0,
            max_threshold=99.999,
            chunk_size=500,
            cores_count=2,
            algorithm="RapidFuzz",
            save_perfect_matches=True,
            product_type="All",
        )
        analyzer = SimilarityAnalyzer(config)
        analyzer.load_csv_data(sample_csv)
        analyzer.preprocess()
        results = analyzer.calculate_similarity(use_multiprocessing=False)

        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert "similarity" in results.columns
        assert results["similarity"].between(70.0, 100.0).all()

    def test_summary_stats(self, sample_csv):
        config = AnalysisConfig(
            min_threshold=70.0,
            cores_count=2,
            save_perfect_matches=True,
        )
        analyzer = SimilarityAnalyzer(config)
        analyzer.load_csv_data(sample_csv)
        analyzer.preprocess()
        analyzer.calculate_similarity()

        stats = analyzer.get_summary_stats()
        assert "total_matches" in stats
        assert "avg_similarity" in stats
        assert stats["total_matches"] > 0

    def test_save_results(self, sample_csv, tmp_path):
        config = AnalysisConfig(cores_count=2)
        analyzer = SimilarityAnalyzer(config)
        analyzer.load_csv_data(sample_csv)
        analyzer.preprocess()
        analyzer.calculate_similarity()

        output_dir = str(tmp_path / "output")
        analyzer.save_results(output_directory=output_dir)

        from pathlib import Path
        output_path = Path(output_dir) / "1d_main_spec_RapidFuzz"
        assert output_path.exists()
        csv_files = list(output_path.glob("main_*.csv"))
        assert len(csv_files) == 1
