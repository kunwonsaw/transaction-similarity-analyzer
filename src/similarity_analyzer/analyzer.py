"""
Core analysis engine for the Similarity Analyzer package.

Contains the SimilarityAnalyzer class (data loading, preprocessing,
parallel similarity calculation, and result export), a standalone
streaming function for memory-constrained environments, and the
CLI entry point.
"""

import os
import re
import logging
import argparse
import multiprocessing as mp

import pandas as pd
import numpy as np

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Generator
from itertools import combinations
from tqdm import tqdm
from joblib import Parallel, delayed

from .config import AnalysisConfig
from .utils import (
    setup_logger,
    extract_letters_only,
    clean_supplier_name,
    create_batches,
)
from .similarity import (
    get_similarity_function,
    process_regular_similarity_chunk,
    process_pair_chunk,
)


LOGGER = logging.getLogger("SimilarityAnalyzer")


# ----------------------------------------------------------------------------
# MAIN ANALYSIS CLASS
# ----------------------------------------------------------------------------
class SimilarityAnalyzer:
    """
    Main engine for supplier procurement data similarity analysis.

    Handles the complete analysis pipeline:
    1. Data loading from Excel/CSV
    2. Preprocessing and cleaning
    3. Similarity calculation using parallel processing
    4. Results saving and statistics generation

    The analyzer supports two parallelization strategies:
    - **Joblib** (default): Operates on deduplicated composite field values.
      Lower memory footprint, good for exploratory analysis.
    - **Native multiprocessing**: Operates on full record dictionaries with
      cross-product pairing between reference and comparison date ranges.
      Better for large-scale production runs.

    Args:
        config: AnalysisConfig instance with all analysis settings.

    Example:
        >>> config = AnalysisConfig(min_threshold=80.0, algorithm="RapidFuzz")
        >>> analyzer = SimilarityAnalyzer(config)
        >>> analyzer.load_csv_data("transactions.csv")
        >>> analyzer.preprocess()
        >>> results = analyzer.calculate_similarity()
        >>> analyzer.save_results("output/")
    """

    # Column schema for result DataFrames (shared by both pipelines)
    RESULT_COLUMNS = [
        "main_spec1", "main_spec2", "similarity",
        "unit_price1", "unit_price2",
        "supplier_name1", "supplier_name2",
        "manufacturer_name1", "manufacturer_name2",
        "order_item_number1", "order_item_number2",
        "order_quantity1", "order_quantity2",
        "total1", "total2",
        "unit_price_diff", "total_diff",
        "main_spec1_date", "main_spec2_date",
    ]

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.column_prefix = "product" if config.use_product_name else "department"

        # Data containers
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.similarity_results: Optional[pd.DataFrame] = None
        self.perfect_matches: Optional[pd.DataFrame] = None

        self._log_initialization()

    def _log_initialization(self):
        """Log configuration summary at startup."""
        LOGGER.info("✅ SimilarityAnalyzer initialized")
        LOGGER.info(
            f"   Thresholds: [{self.config.min_threshold}, "
            f"{self.config.max_threshold}], "
            f"Algorithm: {self.config.algorithm}"
        )
        LOGGER.info(
            f"   Cores: {min(self.config.cores_count, os.cpu_count())} "
            f"(available: {os.cpu_count()}), Backend: {self.config.backend}"
        )
        LOGGER.info(
            f"   Column prefix: {self.column_prefix}, "
            f"Clean specs: {self.config.clean_specifications}, "
            f"Letters only: {self.config.keep_letters_only}"
        )

        if any([
            self.config.ref_start_date, self.config.ref_end_date,
            self.config.comp_start_date, self.config.comp_end_date,
        ]):
            LOGGER.info(
                f"   Date filters: "
                f"ref=({self.config.ref_start_date}, {self.config.ref_end_date}), "
                f"comp=({self.config.comp_start_date}, {self.config.comp_end_date})"
            )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Data Loading
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_excel_data(self, file_config: Dict[str, Dict[str, Any]]) -> None:
        """
        Load data from multiple Excel files.

        Args:
            file_config: Dictionary with structure::

                {
                    'file_key': {
                        'path': 'file.xlsx',
                        'sheets': [
                            ('sheet_name', header_row, 'col_range', max_rows | None),
                            ...
                        ],
                        'tag': 'classification_tag'   # optional
                    }
                }

        Raises:
            FileNotFoundError: If any specified file does not exist.
        """
        LOGGER.info("** Loading Excel files...")

        all_frames = []

        for file_key, file_info in file_config.items():
            file_path = file_info["path"]
            if not Path(file_path).exists():
                raise FileNotFoundError(f"{file_key} not found: {file_path}")

            for sheet_info in file_info["sheets"]:
                sheet_name, header_row, col_range, max_rows = sheet_info

                read_options = {
                    "sheet_name": sheet_name,
                    "header": header_row,
                    "usecols": col_range,
                }
                if max_rows:
                    read_options["nrows"] = max_rows

                df = pd.read_excel(file_path, **read_options)

                # Add classification tag if not already present
                if "classification" not in df.columns and "tag" in file_info:
                    df.insert(0, "classification", file_info["tag"])

                all_frames.append(df)
                LOGGER.debug(f"   Loaded: {file_key}/{sheet_name} — {df.shape}")

        # Unify columns across all frames and merge
        all_columns = pd.Index(
            [col for df in all_frames for col in df.columns]
        ).unique()
        unified_frames = [
            df.reindex(columns=all_columns, fill_value="") for df in all_frames
        ]

        self.raw_data = pd.concat(unified_frames, ignore_index=True)
        LOGGER.info(f"✅ Excel loading complete: {self.raw_data.shape}")

    def load_csv_data(self, file_path: str) -> None:
        """
        Load data from a CSV file.

        Args:
            file_path: Path to a UTF-8-SIG encoded CSV file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        LOGGER.info(f"** Loading CSV: {file_path}")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"CSV not found: {file_path}")

        self.raw_data = pd.read_csv(file_path, encoding="utf-8-sig")
        LOGGER.info(f"✅ CSV loading complete: {self.raw_data.shape}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Preprocessing
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def preprocess(self) -> None:
        """
        Run the complete preprocessing pipeline.

        Steps:
            1. Sort by settlement month (if available)
            2. Classify cost types
            3. Filter by product type
            4. Clean specification strings
            5. Create letters-only specs (optional)
            6. Normalize supplier/manufacturer names
            7. Parse and unify date columns
            8. Remove duplicate order items
            9. Create composite grouping columns

        Raises:
            ValueError: If no data has been loaded yet.
        """
        LOGGER.info("** Starting data preprocessing...")

        if self.raw_data is None:
            raise ValueError(
                "No data loaded. Call load_excel_data() or load_csv_data() first."
            )

        df = self.raw_data.copy()

        # 1. Sort by settlement month if available
        if "settlement_month" in df.columns:
            df["settlement_month"] = pd.to_numeric(
                df["settlement_month"], errors="coerce"
            )
            df = df.sort_values("settlement_month")

        # 2. Create cost classification
        if "cost_type" in df.columns:
            df["classification1"] = df["cost_type"].apply(
                lambda x: "repair"
                if isinstance(x, str) and "repair" in x.lower()
                else "non_repair"
            )

        # 3. Filter by product type
        if self.config.product_type == "repair":
            df = df[df["cost_type"] == "repair"].copy()
            LOGGER.info(f"   Filtered repair items: {len(df):,}")
        elif self.config.product_type == "consumable":
            df = df[df["cost_type"] == "consumable"].copy()
            LOGGER.info(f"   Filtered consumable items: {len(df):,}")

        self.processed_data = df

        # 4–9. Column-level transformations
        if self.config.clean_specifications:
            self._clean_specifications()
        if self.config.keep_letters_only:
            self._create_letters_only_specs()
        self._clean_supplier_names()
        self._process_date_columns()
        self._remove_duplicates()
        self._create_composite_columns()

        LOGGER.info(f"✅ Preprocessing complete: {self.processed_data.shape}")

    def _clean_specifications(self):
        """Remove configured stopwords from the main_spec column."""
        if "main_spec" not in self.processed_data.columns:
            return

        pattern = (
            r"\b(?:" +
            "|".join(map(re.escape, self.config.words_to_delete)) +
            r")\b"
        )
        self.processed_data["main_spec"] = (
            self.processed_data["main_spec"]
            .astype(str)
            .str.replace(pattern, " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    def _create_letters_only_specs(self):
        """Create a letters-only variant of main_spec for comparison."""
        if "main_spec" in self.processed_data.columns:
            self.processed_data["main_spec_letters_only"] = (
                self.processed_data["main_spec"].apply(extract_letters_only)
            )

    def _clean_supplier_names(self):
        """Normalize supplier and manufacturer names by removing corporate suffixes."""
        if not {"supplier_name", "manufacturer_name"}.issubset(
            self.processed_data.columns
        ):
            return

        # Fill missing suppliers with manufacturer name
        self.processed_data["supplier_name"] = (
            self.processed_data["supplier_name"]
            .fillna(self.processed_data["manufacturer_name"])
        )

        self.processed_data["supplier_name_cleaned"] = (
            self.processed_data["supplier_name"].apply(clean_supplier_name)
        )
        self.processed_data["manufacturer_name_cleaned"] = (
            self.processed_data["manufacturer_name"].apply(clean_supplier_name)
        )

        # Reorder to place cleaned columns next to originals
        column_list = list(self.processed_data.columns)
        for original, cleaned in [
            ("supplier_name", "supplier_name_cleaned"),
            ("manufacturer_name", "manufacturer_name_cleaned"),
        ]:
            if cleaned in column_list and original in column_list:
                column_list.remove(cleaned)
                index = column_list.index(original) + 1
                column_list.insert(index, cleaned)

        self.processed_data = self.processed_data[column_list]

    def _process_date_columns(self):
        """
        Create a unified final_date column using a waterfall approach.

        Priority: delivery_completion_date > order_creation_date > pr_approval_date.
        """
        date_columns = [
            "pr_approval_date",
            "order_creation_date",
            "delivery_completion_date",
        ]

        # Convert to datetime
        for col in date_columns:
            if col in self.processed_data.columns:
                self.processed_data[col] = pd.to_datetime(
                    self.processed_data[col], errors="coerce"
                )

        # Build final_date with priority fallback
        if "final_date" not in self.processed_data.columns:
            self.processed_data["final_date"] = pd.NaT

            # Apply in reverse priority so higher-priority columns overwrite
            for col in date_columns:
                if col in self.processed_data.columns:
                    self.processed_data["final_date"] = (
                        self.processed_data["final_date"]
                        .fillna(self.processed_data[col])
                    )

        self.processed_data["final_date"] = pd.to_datetime(
            self.processed_data["final_date"], errors="coerce"
        )

    def _remove_duplicates(self):
        """Remove duplicate order items, keeping the first occurrence."""
        initial_count = len(self.processed_data)

        if "order_item_number" in self.processed_data.columns:
            self.processed_data = self.processed_data.drop_duplicates(
                subset=["order_item_number"], keep="first"
            )

        removed = initial_count - len(self.processed_data)
        LOGGER.info(
            f"   Removed {removed:,} duplicates, "
            f"remaining: {len(self.processed_data):,}"
        )

    def _create_composite_columns(self):
        """
        Create a composite grouping column and unique row ID.

        The composite column concatenates department/product name with the
        specification string, enabling grouped similarity comparisons.
        """
        spec_column = (
            "main_spec_letters_only"
            if self.config.keep_letters_only
            and "main_spec_letters_only" in self.processed_data.columns
            else "main_spec"
        )

        composite_name = f"{self.column_prefix}_name+main_spec"

        if f"{self.column_prefix}_name" in self.processed_data.columns:
            self.processed_data[composite_name] = (
                self.processed_data[f"{self.column_prefix}_name"]
                .astype(str).str.lower().str.strip()
                + " "
                + self.processed_data[spec_column]
                .astype(str).str.lower().str.strip()
            )
        else:
            self.processed_data[composite_name] = (
                self.processed_data[spec_column]
                .astype(str).str.lower().str.strip()
            )

        # Unique row ID for deduplication and traceability
        id_column = (
            "order_item_number"
            if "order_item_number" in self.processed_data.columns
            else None
        )
        id_series = (
            self.processed_data[id_column].astype(str)
            if id_column
            else self.processed_data.index.astype(str)
        )
        self.processed_data["unique_row_id"] = (
            self.processed_data[composite_name] + "_" + id_series
        )

        LOGGER.info(
            f"   Created composite column: '{composite_name}' "
            f"(based on '{spec_column}')"
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Similarity Calculation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def calculate_similarity(
        self, use_multiprocessing: bool = False
    ) -> pd.DataFrame:
        """
        Execute similarity calculation across all valid pairs.

        Args:
            use_multiprocessing: If True, use native multiprocessing.Pool
                (requires all four date filters). If False, use Joblib.

        Returns:
            DataFrame with similarity results and enriched metadata.
        """
        if use_multiprocessing:
            return self._multiprocessing_method()
        return self._joblib_method()

    # ---- Joblib pipeline ----

    def _joblib_method(self) -> pd.DataFrame:
        """
        Joblib-based similarity calculation.

        Operates on deduplicated composite field values, producing lighter
        payloads and fewer total pairs than the multiprocessing approach.
        """
        LOGGER.info("** Starting Joblib-based similarity analysis...")

        composite_name = f"{self.column_prefix}_name+main_spec"
        working_df = self.processed_data.copy()

        # Pre-filter by date ranges if configured
        has_date_filters = any([
            self.config.ref_start_date, self.config.ref_end_date,
            self.config.comp_start_date, self.config.comp_end_date,
        ])

        if has_date_filters:
            working_df, ref_df, comp_df = self._apply_date_prefilter(working_df)

        # Generate unique composite values for pairwise comparison
        unique_combinations = working_df[composite_name].unique()
        LOGGER.info(f"   Unique combinations: {len(unique_combinations):,}")

        # STEP 1: Regular matches (excluding 100%)
        LOGGER.info(
            f"** STEP 1: Regular matches "
            f"({self.config.min_threshold}~{self.config.max_threshold}, "
            f"excluding 100%)"
        )
        matches = self._calculate_regular_similarities(
            unique_combinations, composite_name
        )
        similarity_results = self._merge_similarity_results(
            matches, composite_name
        )

        # Apply cross-date-range filtering if date ranges were specified
        if has_date_filters:
            similarity_results = self._apply_date_postfilter(similarity_results)

        self.similarity_results = self._finalize_results(similarity_results)
        LOGGER.info(f"✅ STEP 1 complete: {self.similarity_results.shape}")

        # STEP 2: Perfect matches (optional)
        if self.config.save_perfect_matches:
            LOGGER.info("** STEP 2: Perfect matches (100%)")
            original_processed = self.processed_data
            self.processed_data = working_df  # Temporarily swap
            self.perfect_matches = self._find_perfect_matches()
            self.processed_data = original_processed  # Restore
            LOGGER.info(
                f"✅ STEP 2 complete: "
                f"{self.perfect_matches.shape if self.perfect_matches is not None else (0, 0)}"
            )

        LOGGER.info("✅ Similarity analysis complete")
        return self.similarity_results

    def _apply_date_prefilter(
        self, working_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Filter data to records within either the reference or comparison
        date range.

        Returns:
            Tuple of (combined_df, ref_df, comp_df).
        """
        LOGGER.info("** Pre-filtering data by date ranges...")

        ref_mask = pd.Series(True, index=working_df.index)
        comp_mask = pd.Series(True, index=working_df.index)

        if self.config.ref_start_date:
            ref_mask &= working_df["final_date"] >= self.config.ref_start_date
        if self.config.ref_end_date:
            ref_mask &= working_df["final_date"] <= self.config.ref_end_date
        if self.config.comp_start_date:
            comp_mask &= working_df["final_date"] >= self.config.comp_start_date
        if self.config.comp_end_date:
            comp_mask &= working_df["final_date"] <= self.config.comp_end_date

        combined_df = working_df[ref_mask | comp_mask].copy()

        ref_df = self.processed_data[ref_mask].copy()
        comp_df = self.processed_data[comp_mask].copy()

        LOGGER.info(
            f"   Filtered from {len(self.processed_data):,} "
            f"to {len(combined_df):,} records"
        )
        LOGGER.info(f"   Reference period: {len(ref_df):,} records")
        LOGGER.info(f"   Comparison period: {len(comp_df):,} records")

        return combined_df, ref_df, comp_df

    def _calculate_regular_similarities(
        self,
        unique_combinations: np.ndarray,
        composite_name: str,
    ) -> pd.DataFrame:
        """Calculate regular similarities using Joblib parallelization."""
        num_cores = min(self.config.cores_count, os.cpu_count())
        LOGGER.info(f"   Using {num_cores} CPU cores")

        chunk_list = list(
            self._create_chunks_fast(unique_combinations, self.config.chunk_size)
        )
        LOGGER.info(f"   Processing {len(chunk_list):,} chunks")

        results = Parallel(n_jobs=num_cores, backend=self.config.backend)(
            delayed(process_regular_similarity_chunk)(
                chunk,
                self.config.min_threshold,
                self.config.max_threshold,
                self.config.algorithm,
            )
            for chunk in tqdm(
                chunk_list, desc="Regular Matching", unit="chunk", leave=False
            )
        )

        flattened = [row for chunk in results for row in chunk]
        LOGGER.info(f"✅ Regular matches found: {len(flattened):,}")

        return pd.DataFrame(
            flattened,
            columns=[
                f"{composite_name}_1",
                f"{composite_name}_2",
                "similarity",
            ],
        )

    def _merge_similarity_results(
        self,
        matches_df: pd.DataFrame,
        composite_name: str,
    ) -> pd.DataFrame:
        """
        Enrich similarity pairs with metadata from processed data.

        Joins supplier names, prices, quantities, and dates onto each
        side of the matched pair.
        """
        LOGGER.info("** Merging similarity results with metadata...")

        metadata_columns = [
            composite_name, "main_spec", "order_item_number", "unique_row_id",
            "supplier_name", "manufacturer_name",
            "order_quantity", "unit_price", "total", "final_date",
        ]

        if f"{self.column_prefix}_name" in self.processed_data.columns:
            metadata_columns.append(f"{self.column_prefix}_name")

        available = [
            c for c in metadata_columns if c in self.processed_data.columns
        ]
        metadata = self.processed_data[available].copy()
        metadata = metadata.drop_duplicates(
            subset=[composite_name], keep="first"
        )

        # Create renamed copies for each side of the pair
        rename_1 = {
            c: f"{c}_1" for c in metadata.columns if c != composite_name
        }
        rename_2 = {
            c: f"{c}_2" for c in metadata.columns if c != composite_name
        }
        metadata_1 = metadata.rename(columns=rename_1)
        metadata_2 = metadata.rename(columns=rename_2)

        merged = (
            matches_df
            .merge(
                metadata_1,
                left_on=f"{composite_name}_1",
                right_on=composite_name,
                how="left",
            )
            .merge(
                metadata_2,
                left_on=f"{composite_name}_2",
                right_on=composite_name,
                how="left",
            )
        )

        # Drop duplicate composite key columns introduced by merge
        merged = merged.drop(
            columns=[f"{composite_name}_x", f"{composite_name}_y"],
            errors="ignore",
        )

        for col in ("final_date_1", "final_date_2"):
            if col in merged.columns:
                merged[col] = pd.to_datetime(merged[col], errors="coerce")

        LOGGER.info(f"✅ Merged results: {merged.shape}")
        return merged

    def _create_chunks_fast(
        self,
        combination_array: np.ndarray,
        chunk_size: int,
    ) -> Generator[List[Tuple], None, None]:
        """Yield fixed-size chunks of all unique pairwise combinations."""
        total_pairs = len(combination_array) * (len(combination_array) - 1) // 2
        LOGGER.info(f"   Total pairs to process: {total_pairs:,}")

        chunk = []
        for i, item1 in enumerate(combination_array):
            for item2 in combination_array[i + 1:]:
                chunk.append((item1, item2))
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk

    def _find_perfect_matches(self) -> Optional[pd.DataFrame]:
        """Find 100% similarity matches with full metadata."""
        LOGGER.info("** Searching for perfect matches...")

        spec_column = (
            "main_spec_letters_only"
            if self.config.keep_letters_only
            and "main_spec_letters_only" in self.processed_data.columns
            else "main_spec"
        )

        unique_items = (
            self.processed_data[[spec_column, "order_item_number"]]
            .dropna()
            .drop_duplicates()
        )
        unique_items = unique_items[
            unique_items[spec_column].astype(str).str.strip() != ""
        ]
        items = unique_items.to_records(index=False)

        LOGGER.info(f"   Searching {len(items):,} items")

        similarity_func = get_similarity_function(self.config.algorithm)
        results = []
        total_pairs = (len(items) * (len(items) - 1)) // 2

        for (spec1, id1), (spec2, id2) in tqdm(
            combinations(items, 2),
            total=total_pairs,
            desc="Finding Perfect Matches",
            leave=False,
        ):
            score = similarity_func(str(spec1), str(spec2))
            if score >= 100:
                results.append({
                    f"{spec_column}_1": spec1,
                    f"{spec_column}_2": spec2,
                    "order_item_number_1": id1,
                    "order_item_number_2": id2,
                    "similarity": score,
                })

        if not results:
            LOGGER.info("   No perfect matches found")
            return None

        matches_df = pd.DataFrame(results)
        LOGGER.info(f"   Perfect matches found: {len(matches_df):,}")

        # Merge metadata
        meta_cols = [
            spec_column, "order_item_number", "unique_row_id",
            f"{self.column_prefix}_name",
            "supplier_name", "manufacturer_name",
            "order_quantity", "unit_price", "total", "final_date",
        ]
        available = [
            c for c in meta_cols if c in self.processed_data.columns
        ]
        metadata = self.processed_data[available].drop_duplicates(
            "order_item_number"
        )

        metadata1 = metadata.rename(
            columns={c: f"{c}1" for c in metadata.columns}
        )
        metadata2 = metadata.rename(
            columns={c: f"{c}2" for c in metadata.columns}
        )

        merged = (
            matches_df
            .merge(
                metadata1,
                left_on="order_item_number_1",
                right_on="order_item_number1",
            )
            .merge(
                metadata2,
                left_on="order_item_number_2",
                right_on="order_item_number2",
            )
        )

        # Clean up redundant columns
        drop_cols = [
            f"{spec_column}_1", f"{spec_column}_2",
            "order_item_number_1", "order_item_number_2",
            "unique_row_id1", "unique_row_id2",
        ]
        merged.drop(
            columns=[c for c in drop_cols if c in merged.columns],
            inplace=True,
            errors="ignore",
        )

        merged["final_date1"] = pd.to_datetime(
            merged["final_date1"], errors="coerce"
        )
        merged["final_date2"] = pd.to_datetime(
            merged["final_date2"], errors="coerce"
        )

        # Apply date filters
        if any([
            self.config.ref_start_date, self.config.ref_end_date,
            self.config.comp_start_date, self.config.comp_end_date,
        ]):
            merged = merged[
                (merged["final_date1"].between(
                    self.config.ref_start_date, self.config.ref_end_date
                ))
                | (merged["final_date2"].between(
                    self.config.comp_start_date, self.config.comp_end_date
                ))
            ]
            LOGGER.info(f"   After date filter: {len(merged):,}")

        return merged

    # ---- Native multiprocessing pipeline ----

    def _multiprocessing_method(self) -> pd.DataFrame:
        """
        Native multiprocessing approach using cross-product pairing.

        Requires all four date filters (ref_start, ref_end, comp_start,
        comp_end) to define reference and comparison populations.

        Raises:
            ValueError: If any date filter is missing.
        """
        LOGGER.info("** Starting native multiprocessing analysis...")

        if not all([
            self.config.ref_start_date, self.config.ref_end_date,
            self.config.comp_start_date, self.config.comp_end_date,
        ]):
            raise ValueError(
                "Multiprocessing method requires all four date filters "
                "(ref_start_date, ref_end_date, comp_start_date, comp_end_date)"
            )

        # Split into reference and comparison populations
        ref_df = self.processed_data[
            (self.processed_data["final_date"] >= self.config.ref_start_date)
            & (self.processed_data["final_date"] <= self.config.ref_end_date)
        ].copy()

        comp_df = self.processed_data[
            (self.processed_data["final_date"] >= self.config.comp_start_date)
            & (self.processed_data["final_date"] <= self.config.comp_end_date)
        ].copy()

        ref_records = ref_df.to_dict(orient="records")
        comp_records = comp_df.to_dict(orient="records")

        LOGGER.info(
            f"   Reference: {len(ref_records):,} | "
            f"Comparison: {len(comp_records):,}"
        )

        # Build chunked cross-product pairs
        pair_generator = (
            (r1, r2) for r1 in ref_records for r2 in comp_records
        )
        batch_generator = create_batches(
            pair_generator, batch_size=self.config.chunk_size
        )

        expected_pairs = len(ref_records) * len(comp_records)
        num_chunks = (
            expected_pairs // self.config.chunk_size
            + int(expected_pairs % self.config.chunk_size != 0)
        )

        num_cores = min(self.config.cores_count, mp.cpu_count())
        LOGGER.info(
            f"   Cores: {num_cores} | "
            f"Expected pairs: {expected_pairs:,} | "
            f"Chunks: {num_chunks:,}"
        )

        similarity_func = get_similarity_function(self.config.algorithm)

        regular_results = []
        perfect_results = []

        with mp.Pool(processes=num_cores) as pool:
            for chunk_result, perfect_chunk in tqdm(
                pool.imap_unordered(
                    process_pair_chunk,
                    [
                        (
                            chunk,
                            self.config.min_threshold,
                            self.config.max_threshold,
                            self.config.save_perfect_matches,
                            self.config.keep_letters_only,
                            similarity_func,
                        )
                        for chunk in batch_generator
                    ],
                ),
                total=num_chunks,
                desc="Processing chunks",
                leave=False,
            ):
                if chunk_result:
                    regular_results.extend(chunk_result)
                if self.config.save_perfect_matches and perfect_chunk:
                    perfect_results.extend(perfect_chunk)

        # Build result DataFrames
        self.similarity_results = (
            pd.DataFrame(regular_results, columns=self.RESULT_COLUMNS)
            if regular_results
            else pd.DataFrame(columns=self.RESULT_COLUMNS)
        )

        if self.config.save_perfect_matches and perfect_results:
            self.perfect_matches = pd.DataFrame(
                perfect_results, columns=self.RESULT_COLUMNS
            )

        LOGGER.info(
            f"✅ Multiprocessing complete: "
            f"regular={len(self.similarity_results):,}, "
            f"perfect={len(self.perfect_matches) if self.config.save_perfect_matches else 0:,}"
        )

        return self.similarity_results

    # ---- Post-processing ----

    def _apply_date_postfilter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter results to cross-date-range matches only.

        Ensures that matched pairs span the reference and comparison periods
        (not ref-ref or comp-comp matches).
        """
        LOGGER.info("** Applying date filters...")

        ref_filter = pd.Series(True, index=df.index)
        comp_filter = pd.Series(True, index=df.index)

        if self.config.ref_start_date:
            ref_filter &= df["final_date_1"] >= self.config.ref_start_date
        if self.config.ref_end_date:
            ref_filter &= df["final_date_1"] <= self.config.ref_end_date
        if self.config.comp_start_date:
            comp_filter &= df["final_date_2"] >= self.config.comp_start_date
        if self.config.comp_end_date:
            comp_filter &= df["final_date_2"] <= self.config.comp_end_date

        filtered = df[ref_filter & comp_filter].copy()
        LOGGER.info(f"✅ After date filtering: {len(filtered):,}")

        return filtered

    def _finalize_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to output schema and calculate price differences."""
        composite_name = f"{self.column_prefix}_name+main_spec"

        column_map = {
            "similarity": "similarity",
            "unit_price_1": "unit_price1",
            "unit_price_2": "unit_price2",
            "supplier_name_1": "supplier_name1",
            "supplier_name_2": "supplier_name2",
            "manufacturer_name_1": "manufacturer_name1",
            "manufacturer_name_2": "manufacturer_name2",
            "final_date_1": "main_spec1_date",
            "final_date_2": "main_spec2_date",
            "main_spec_1": "main_spec1",
            "main_spec_2": "main_spec2",
            "order_item_number_1": "order_item_number1",
            "order_item_number_2": "order_item_number2",
            "order_quantity_1": "order_quantity1",
            "order_quantity_2": "order_quantity2",
            "total_1": "total1",
            "total_2": "total2",
            f"{self.column_prefix}_name_1": f"{self.column_prefix}_name1",
            f"{self.column_prefix}_name_2": f"{self.column_prefix}_name2",
        }
        df = df.rename(columns=column_map)

        # Calculate price differences
        if "unit_price1" in df.columns and "unit_price2" in df.columns:
            df["unit_price_diff"] = df.apply(
                lambda row: abs(row["unit_price2"] - row["unit_price1"])
                if pd.notnull(row["unit_price2"])
                and pd.notnull(row["unit_price1"])
                else None,
                axis=1,
            )

        if "total1" in df.columns and "total2" in df.columns:
            df["total_diff"] = df.apply(
                lambda row: abs(row["total2"] - row["total1"])
                if pd.notnull(row["total2"]) and pd.notnull(row["total1"])
                else None,
                axis=1,
            )

        # Drop internal-only columns
        drop_cols = [
            f"{composite_name}_1", f"{composite_name}_2",
            "unique_row_id_1", "unique_row_id_2",
        ]
        df = df.drop(
            columns=[c for c in drop_cols if c in df.columns]
        )

        return df

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def save_results(self, output_directory: str = "result") -> None:
        """
        Save analysis results to CSV files.

        Creates a subdirectory named by algorithm and writes:
        - main_{suffix}.csv: Regular similarity matches
        - perfect100_{suffix}.csv: Perfect matches (if enabled)

        Args:
            output_directory: Base directory for output files.
        """
        if self.similarity_results is None:
            LOGGER.error("❌ No results to save")
            return

        output_path = (
            Path(output_directory) / f"1d_main_spec_{self.config.algorithm}"
        )
        output_path.mkdir(parents=True, exist_ok=True)

        suffix_parts = [
            "letters_only" if self.config.keep_letters_only else "normal",
            "field_concat",
            "product" if self.config.use_product_name else "department",
        ]
        suffix = "_".join(suffix_parts)

        # Main results
        main_file = output_path / f"main_{suffix}.csv"
        self.similarity_results.to_csv(
            main_file, encoding="utf-8-sig", index=False
        )
        LOGGER.info(f"✅ Main results saved: {main_file}")

        # Perfect matches
        if (
            self.config.save_perfect_matches
            and self.perfect_matches is not None
            and len(self.perfect_matches) > 0
        ):
            perfect_file = output_path / f"perfect100_{suffix}.csv"
            self.perfect_matches.to_csv(
                perfect_file, encoding="utf-8-sig", index=False
            )
            LOGGER.info(f"✅ Perfect matches saved: {perfect_file}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Generate summary statistics from analysis results.

        Returns:
            Dictionary with match counts, similarity distribution,
            and price difference statistics.
        """
        if self.similarity_results is None or len(self.similarity_results) == 0:
            return {"error": "No results available"}

        stats: Dict[str, Any] = {
            "total_matches": int(len(self.similarity_results)),
            "avg_similarity": float(
                self.similarity_results["similarity"].mean()
            ),
            "similarity_range": {
                "min": float(self.similarity_results["similarity"].min()),
                "max": float(self.similarity_results["similarity"].max()),
            },
        }

        if "unit_price_diff" in self.similarity_results.columns:
            stats["unit_price_diff_stats"] = {
                "avg": float(
                    self.similarity_results["unit_price_diff"].mean()
                ),
                "median": float(
                    self.similarity_results["unit_price_diff"].median()
                ),
                "max": float(
                    self.similarity_results["unit_price_diff"].max()
                ),
            }

        if (
            self.config.save_perfect_matches
            and self.perfect_matches is not None
            and len(self.perfect_matches) > 0
            and "unit_price1" in self.perfect_matches.columns
        ):
            stats["perfect_matches"] = {
                "count": int(len(self.perfect_matches)),
                "avg_price_diff": float(
                    self.perfect_matches["unit_price1"]
                    .subtract(
                        self.perfect_matches["unit_price2"], fill_value=0
                    )
                    .abs()
                    .mean()
                ),
            }

        return stats


# ----------------------------------------------------------------------------
# STANDALONE STREAMING FUNCTION
# ----------------------------------------------------------------------------
def run_multiprocessing_streaming(
    input_csv: str,
    output_directory: str,
    cost_filter: str = "repair",
    ref_start_date: str = None,
    ref_end_date: str = None,
    comp_start_date: str = None,
    comp_end_date: str = None,
    min_threshold: float = 70.0,
    max_threshold: float = 100.0 - 1e-9,
    save_perfect: bool = True,
    letters_only: bool = False,
    chunk_size: int = 500_000,
    algorithm: str = "RapidFuzz",
):
    """
    Standalone streaming analysis for memory-constrained environments.

    Writes results directly to CSV chunk-by-chunk rather than accumulating
    in memory. Suitable for datasets too large to hold all results in RAM.

    Args:
        input_csv: Path to input CSV file.
        output_directory: Base directory for output files.
        cost_filter: Filter string for cost_type column (e.g. "repair").
        ref_start_date: Reference period start (YYYY-MM-DD).
        ref_end_date: Reference period end (YYYY-MM-DD).
        comp_start_date: Comparison period start (YYYY-MM-DD).
        comp_end_date: Comparison period end (YYYY-MM-DD).
        min_threshold: Minimum similarity score to include.
        max_threshold: Maximum similarity score to include.
        save_perfect: Whether to save 100% matches separately.
        letters_only: Whether to strip non-letter characters.
        chunk_size: Pairs per processing chunk.
        algorithm: Similarity algorithm name.
    """
    similarity_func = get_similarity_function(algorithm)

    output_path = Path(output_directory) / f"1d_main_spec_{algorithm}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and preprocess
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    df["final_date"] = pd.to_datetime(df["final_date"], errors="coerce")

    if cost_filter:
        df = df[df["cost_type"].astype(str).str.contains(cost_filter)]

    df = df.dropna(subset=["main_spec", "order_item_number"])
    df["main_spec"] = df["main_spec"].astype(str)
    df["order_item_number"] = df["order_item_number"].astype(str)

    # Split reference and comparison populations
    ref_df = df[
        (df["final_date"] >= pd.to_datetime(ref_start_date))
        & (df["final_date"] <= pd.to_datetime(ref_end_date))
    ].copy()

    comp_df = df[
        (df["final_date"] >= pd.to_datetime(comp_start_date))
        & (df["final_date"] <= pd.to_datetime(comp_end_date))
    ].copy()

    ref_records = ref_df.to_dict(orient="records")
    comp_records = comp_df.to_dict(orient="records")

    pair_generator = ((r1, r2) for r1 in ref_records for r2 in comp_records)
    batch_gen = create_batches(pair_generator, batch_size=chunk_size)

    expected_count = len(ref_records) * len(comp_records)
    num_chunks = (
        expected_count // chunk_size
        + int(expected_count % chunk_size != 0)
    )

    num_cores = max(8, mp.cpu_count() // 2)

    print(f"CPU cores: {num_cores}")
    print(
        f"Reference: {len(ref_records):,} | "
        f"Comparison: {len(comp_records):,}"
    )
    print(f"Expected pairs: {expected_count:,} | Chunks: {num_chunks:,}")
    print(f"Algorithm: {algorithm}")

    suffix = "letters_only" if letters_only else "normal"
    main_file = output_path / f"main_{suffix}.csv"
    perfect_file = output_path / f"perfect100_{suffix}.csv"

    first_main_write = True
    first_perfect_write = True

    # Stream results to disk
    with mp.Pool(processes=num_cores) as pool:
        for chunk_result, perfect_chunk in tqdm(
            pool.imap_unordered(
                process_pair_chunk,
                [
                    (
                        chunk,
                        min_threshold,
                        max_threshold,
                        save_perfect,
                        letters_only,
                        similarity_func,
                    )
                    for chunk in batch_gen
                ],
            ),
            total=num_chunks,
            desc="Processing",
            leave=False,
        ):
            if chunk_result:
                pd.DataFrame(
                    chunk_result,
                    columns=SimilarityAnalyzer.RESULT_COLUMNS,
                ).to_csv(
                    main_file,
                    mode="w" if first_main_write else "a",
                    header=first_main_write,
                    index=False,
                    encoding="utf-8-sig",
                )
                first_main_write = False

            if save_perfect and perfect_chunk:
                pd.DataFrame(
                    perfect_chunk,
                    columns=SimilarityAnalyzer.RESULT_COLUMNS,
                ).to_csv(
                    perfect_file,
                    mode="w" if first_perfect_write else "a",
                    header=first_perfect_write,
                    index=False,
                    encoding="utf-8-sig",
                )
                first_perfect_write = False

    print(f"✅ Analysis complete. Results: {output_path}")


# ----------------------------------------------------------------------------
# CLI INTERFACE
# ----------------------------------------------------------------------------
def build_cli_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Transaction Similarity Analyzer — "
            "Internal Auditing Fraud Detection Tool"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Execution mode
    parser.add_argument(
        "--mode",
        choices=["excel", "csv", "streaming"],
        default="excel",
        help="excel: multi-file Excel | csv: CSV | streaming: streaming writes",
    )

    # I/O
    parser.add_argument("--csv_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./result")

    # Excel file paths (up to 9 files)
    for i in range(1, 10):
        parser.add_argument(f"--excel_file_{i}", type=str, default="")

    # Analysis settings
    parser.add_argument("--min_threshold", type=float, default=70.0)
    parser.add_argument(
        "--max_threshold", type=float, default=100.0 - 1e-9
    )
    parser.add_argument("--chunk_size", type=int, default=1_000)
    parser.add_argument(
        "--cores_count", type=int, default=min(20, os.cpu_count() or 8)
    )
    parser.add_argument(
        "--backend", type=str, default="loky",
        choices=["loky", "multiprocessing", "threading"],
    )
    parser.add_argument("--use_product_name", action="store_true")
    parser.add_argument("--clean_specifications", action="store_true")
    parser.add_argument("--keep_letters_only", action="store_true")
    parser.add_argument("--save_perfect_matches", action="store_true")
    parser.add_argument(
        "--product_type", type=str, default="All",
        choices=["repair", "consumable", "All"],
    )
    parser.add_argument(
        "--algorithm", type=str, default="RapidFuzz",
        choices=["RapidFuzz", "SequenceMatcher"],
    )

    # Date filters
    parser.add_argument("--ref_start_date", type=str, default="2024-01-01")
    parser.add_argument("--ref_end_date", type=str, default="2025-12-31")
    parser.add_argument("--comp_start_date", type=str, default="2022-01-01")
    parser.add_argument("--comp_end_date", type=str, default="2025-12-31")

    # Execution options
    parser.add_argument("--use_multiprocessing", action="store_true")

    # Logging
    parser.add_argument(
        "--verbosity", type=int, default=1, choices=[0, 1, 2]
    )
    parser.add_argument("--log_file", type=str, default="")

    return parser


# ----------------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------------
def main():
    """
    CLI entry point.

    Usage Examples::

        # Excel mode
        python -m similarity_analyzer.analyzer --mode excel \\
            --use_product_name --clean_specifications \\
            --excel_file_1 "path1.xlsx" --excel_file_2 "path2.xlsx"

        # CSV mode with multiprocessing
        python -m similarity_analyzer.analyzer --mode csv \\
            --csv_path "data.csv" --use_multiprocessing

        # Streaming mode (for large datasets)
        python -m similarity_analyzer.analyzer --mode streaming \\
            --csv_path "data.csv" --keep_letters_only
    """
    args = build_cli_parser().parse_args()
    setup_logger(args.verbosity, args.log_file or None)

    config = AnalysisConfig(
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        chunk_size=args.chunk_size,
        cores_count=args.cores_count,
        backend=args.backend,
        ref_start_date=args.ref_start_date,
        ref_end_date=args.ref_end_date,
        comp_start_date=args.comp_start_date,
        comp_end_date=args.comp_end_date,
        use_product_name=args.use_product_name,
        clean_specifications=args.clean_specifications,
        keep_letters_only=args.keep_letters_only,
        save_perfect_matches=args.save_perfect_matches,
        product_type=args.product_type,
        algorithm=args.algorithm,
    )

    if args.mode == "excel":
        analyzer = SimilarityAnalyzer(config)

        # NOTE: Customize file_config for actual Excel structure
        file_config = {
            "file1": {
                "path": args.excel_file_1,
                "sheets": [
                    ("2022_cumulative", 1, "B:AL", 73920),
                    ("2023_cumulative", 1, "B:AS", None),
                    ("2024_raw", 1, "B:AW", None),
                ],
                "tag": "repair",
            },
            # Add remaining files as needed...
        }

        try:
            analyzer.load_excel_data(file_config)
            analyzer.preprocess()
            analyzer.calculate_similarity(
                use_multiprocessing=args.use_multiprocessing
            )
            analyzer.save_results(output_directory=args.output_dir)
            LOGGER.info(f"** Summary stats: {analyzer.get_summary_stats()}")
        except Exception:
            LOGGER.exception("❌ Excel analysis failed")

    elif args.mode == "csv":
        if not args.csv_path:
            raise ValueError("CSV mode requires --csv_path")

        analyzer = SimilarityAnalyzer(config)

        try:
            analyzer.load_csv_data(args.csv_path)
            analyzer.preprocess()
            analyzer.calculate_similarity(
                use_multiprocessing=args.use_multiprocessing
            )
            analyzer.save_results(output_directory=args.output_dir)
            LOGGER.info(f"** Summary stats: {analyzer.get_summary_stats()}")
        except Exception:
            LOGGER.exception("❌ CSV analysis failed")

    else:  # streaming
        if not args.csv_path:
            raise ValueError("Streaming mode requires --csv_path")

        try:
            run_multiprocessing_streaming(
                input_csv=args.csv_path,
                output_directory=args.output_dir,
                cost_filter="repair",
                ref_start_date=args.ref_start_date,
                ref_end_date=args.ref_end_date,
                comp_start_date=args.comp_start_date,
                comp_end_date=args.comp_end_date,
                min_threshold=args.min_threshold,
                max_threshold=args.max_threshold,
                save_perfect=args.save_perfect_matches,
                letters_only=args.keep_letters_only,
                chunk_size=args.chunk_size,
                algorithm=args.algorithm,
            )
        except Exception:
            LOGGER.exception("❌ Streaming run failed")


if __name__ == "__main__":
    main()
