# Transaction Similarity Analyzer

A high-performance Python tool for detecting similar and duplicate transactions in procurement data. Built to support internal auditing and fraud detection workflows, with a focus on **parallelized string similarity at scale**.

This project benchmarks and compares three execution strategies for pairwise string comparison — sequential, Joblib, and native multiprocessing — alongside two similarity algorithms (RapidFuzz vs. SequenceMatcher), providing practical guidance on when and how to parallelize.

---

## Why This Exists

Procurement fraud detection often requires comparing thousands of transaction descriptions against each other to find suspiciously similar entries — duplicate orders routed through different suppliers, inflated prices for near-identical items, or repackaged specifications. The core computational challenge is **pairwise string similarity**: given *n* transaction descriptions, compute similarity scores for all *n(n-1)/2* pairs.

This grows quadratically. 1,000 items produce ~500K pairs. 10,000 items produce ~50M pairs. At that scale, the choice of similarity algorithm and parallelization strategy has a dramatic impact on runtime.

---

## Key Results

### Algorithm Benchmark: RapidFuzz vs. SequenceMatcher

| Metric | SequenceMatcher | RapidFuzz |
|---|---|---|
| Avg Time (100K comparisons) | 1.502s | 0.025s |
| Throughput | 66,564/s | 3,961,644/s |
| **Speedup** | — | **59.5x** |

> RapidFuzz's C++ backend processes nearly 4 million comparisons per second — roughly 60x faster than Python's built-in SequenceMatcher.

### Parallelization Benchmark: Sequential vs. Joblib vs. Multiprocessing

All tests use RapidFuzz with `chunk_size=1,000` on a 32-core machine (13th Gen Intel i9-13900KS).

| Pairs | Method | Cores | Avg Time | Speedup | Throughput |
|---|---|---|---|---|---|
| 1,000 | Sequential | 1 | 0.007s | 1.00x | 137,559/s |
| 1,000 | Joblib | 4 | 0.052s | 0.14x | 19,405/s |
| 1,000 | Multiprocessing | 4 | 0.013s | 0.55x | 75,168/s |
| 10,000 | Sequential | 1 | 0.048s | 1.00x | 210,156/s |
| 10,000 | Multiprocessing | 2 | 0.029s | 1.65x | 347,226/s |
| 10,000 | Multiprocessing | 8 | 0.023s | 2.05x | 431,413/s |
| 50,000 | Sequential | 1 | 0.229s | 1.00x | 217,957/s |
| 50,000 | Joblib | 8 | 0.087s | 2.64x | 576,312/s |
| 50,000 | Multiprocessing | 8 | 0.048s | 4.82x | 1,050,056/s |

**Key takeaways:**
- **Below ~5K pairs, don't parallelize.** Process spawning overhead costs more than the computation itself.
- **Joblib's loky backend has a cold-start penalty** (~130ms) for pool initialization. In short benchmarks this dominates; in long-running pipelines it amortizes away.
- **Native multiprocessing scales more linearly** with core count and consistently outperforms Joblib at every size tested.
- **The crossover point** where parallelization pays off is ~10K pairs for multiprocessing and ~30K–50K pairs for Joblib.

---

## Features

- **Dual parallelization backends** — Joblib (easier API, loky backend) and native `multiprocessing.Pool` (lower overhead, finer control)
- **Two similarity algorithms** — RapidFuzz (multi-metric C++ average) and SequenceMatcher (pure Python baseline)
- **Streaming mode** — writes results to disk chunk-by-chunk for datasets too large to hold in memory
- **Date range filtering** — compare transactions across reference and comparison time periods
- **Perfect match extraction** — optionally separates 100% matches for separate analysis
- **Multilingual support** — handles Korean, Chinese, and English text via Unicode-aware cleaning

---

## Installation

```bash
git clone https://github.com/mechadept/transaction-similarity-analyzer.git
cd transaction-similarity-analyzer
pip install -e .
```

### Requirements

- Python 3.9+
- Dependencies: `pandas`, `numpy`, `rapidfuzz`, `regex`, `joblib`, `tqdm`, `openpyxl`

---

## Quick Start

```bash
python examples/quick_start.py
```

This generates 200 synthetic procurement records and runs the full analysis pipeline:

```
1. Generating synthetic procurement data...
   Created 200 records with 39 unique specs

5. Running similarity analysis (Joblib)...
   Unique combinations: 87
   Total pairs to process: 3,741

6. Results summary:
   Total matches found  : 306
   Average similarity   : 82.4%
   Similarity range     : 70.1% — 98.1%
   Perfect matches      : 1,048

✅ Quick start complete!
```

### Using the Analyzer in Your Code

```python
from similarity_analyzer import SimilarityAnalyzer, AnalysisConfig

config = AnalysisConfig(
    min_threshold=80.0,
    algorithm="RapidFuzz",
    cores_count=8,
    chunk_size=1_000,
    save_perfect_matches=True,
)

analyzer = SimilarityAnalyzer(config)
analyzer.load_csv_data("your_data.csv")
analyzer.preprocess()
results = analyzer.calculate_similarity()
analyzer.save_results("output/")

print(analyzer.get_summary_stats())
```

### CLI Usage

```bash
# CSV mode
python -m similarity_analyzer.analyzer --mode csv \
    --csv_path data.csv --algorithm RapidFuzz --cores_count 8

# Streaming mode (large datasets)
python -m similarity_analyzer.analyzer --mode streaming \
    --csv_path data.csv --chunk_size 500000

# Excel mode (multiple files)
python -m similarity_analyzer.analyzer --mode excel \
    --excel_file_1 "data_2022.xlsx" --excel_file_2 "data_2023.xlsx"
```

---

## Running Benchmarks

```bash
# Algorithm comparison (RapidFuzz vs SequenceMatcher)
python benchmarks/benchmark_algorithms.py

# Parallelization strategies (Sequential vs Joblib vs Multiprocessing)
python benchmarks/benchmark_parallelization.py \
    --sizes 1000 10000 50000 100000 \
    --cores 2 4 8 \
    --output benchmarks/results/parallelization_benchmark.txt
```

---

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

```
38 passed in 0.59s
```

---

## Project Structure

```
transaction-similarity-analyzer/
├── src/similarity_analyzer/
│   ├── __init__.py          # Package exports
│   ├── analyzer.py          # Core SimilarityAnalyzer class and CLI
│   ├── config.py            # AnalysisConfig dataclass with validation
│   ├── similarity.py        # Similarity algorithms and parallel workers
│   └── utils.py             # Text cleaning, logging, batch helpers
├── benchmarks/
│   ├── benchmark_algorithms.py
│   ├── benchmark_parallelization.py
│   └── results/
├── examples/
│   └── quick_start.py       # End-to-end demo with synthetic data
├── tests/
│   └── test_similarity.py   # 38 tests covering utils → pipeline
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## How It Works

The analyzer pipeline has four stages:

**1. Load** — Read transaction data from CSV or Excel files. Multiple Excel files with different schemas are unified into a single DataFrame.

**2. Preprocess** — Clean specification strings (remove stopwords, normalize whitespace), standardize supplier names (strip corporate suffixes like 주식회사, Co., LLC), parse dates with priority fallback, and create composite grouping columns.

**3. Analyze** — Generate all pairwise combinations of unique specifications, distribute chunks across CPU cores, and score each pair using a multi-metric RapidFuzz average (ratio + partial_ratio + token_sort_ratio + QRatio). Results are filtered by similarity threshold and optionally by date range.

**4. Export** — Save results to CSV with full metadata: both specifications, similarity score, supplier names, unit prices, quantities, totals, price differences, and dates.

---

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `min_threshold` | 70.0 | Minimum similarity score to include |
| `max_threshold` | 99.999 | Maximum similarity score to include |
| `algorithm` | RapidFuzz | Similarity algorithm (RapidFuzz or SequenceMatcher) |
| `chunk_size` | 1,000 | Pairs per parallel work unit |
| `cores_count` | 8 | Maximum CPU cores for parallel processing |
| `backend` | loky | Joblib backend (loky, multiprocessing, threading) |
| `clean_specifications` | True | Remove stopwords from specs |
| `keep_letters_only` | False | Strip all non-letter characters |
| `save_perfect_matches` | False | Collect 100% matches separately |
| `product_type` | All | Filter by cost type (repair, consumable, All) |

---

## License

MIT

---

## Author

**Kunwon Saw** — Data Scientist focused on procurement analytics and fraud detection.

- GitHub: [@mechadept](https://github.com/mechadept)
