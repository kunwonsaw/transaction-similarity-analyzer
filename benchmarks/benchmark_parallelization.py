"""
Parallelization Strategy Benchmark
====================================
Compares three execution strategies for pairwise string similarity:
  1. Sequential   — single-threaded baseline
  2. Joblib       — joblib.Parallel with loky backend
  3. Multiprocessing — native multiprocessing.Pool with imap_unordered

Holds the similarity algorithm constant (RapidFuzz) and varies:
  - Dataset size (number of pairwise comparisons)
  - Worker count (CPU cores)
  - Chunk size (pairs per work unit)

Outputs a formatted results table to stdout and optionally saves
to a text file for inclusion in the README and TDS article.

Usage:
    python benchmark_parallelization.py
    python benchmark_parallelization.py --output results/parallelization_benchmark.txt
    python benchmark_parallelization.py --sizes 1000 5000 10000 50000 --cores 1 2 4 8

Author: Kunwon Saw
"""

import os
import sys
import time
import random
import string
import argparse
import platform
import multiprocessing as mp
from datetime import datetime
from itertools import combinations
from typing import List, Tuple, Callable

from rapidfuzz import fuzz
from joblib import Parallel, delayed
import numpy as np


# ----------------------------------------------------------------------------
# SIMILARITY FUNCTION (held constant across all strategies)
# ----------------------------------------------------------------------------
def similarity_rapidfuzz(text1: str, text2: str) -> float:
    """Multi-metric RapidFuzz average (same as in the main package)."""
    scores = [
        fuzz.ratio(text1, text2),
        fuzz.partial_ratio(text1, text2),
        fuzz.token_sort_ratio(text1, text2),
        fuzz.QRatio(text1, text2),
    ]
    return float(np.mean(scores))


# ----------------------------------------------------------------------------
# SYNTHETIC DATA GENERATION
# ----------------------------------------------------------------------------
def generate_strings(count: int, length: int = 24, seed: int = 42) -> List[str]:
    """
    Generate reproducible random strings for benchmarking.

    Uses a mix of ASCII letters, digits, and spaces to roughly simulate
    real-world specification strings.

    Args:
        count: Number of strings to generate.
        length: Character length of each string.
        seed: Random seed for reproducibility.

    Returns:
        List of random strings.
    """
    rng = random.Random(seed)
    charset = string.ascii_letters + string.digits + "   "
    return [
        "".join(rng.choices(charset, k=length))
        for _ in range(count)
    ]


def generate_pairs(strings: List[str], num_pairs: int, seed: int = 42) -> List[Tuple[str, str]]:
    """
    Sample a fixed number of unique pairs from a string list.

    For small string lists where C(n,2) <= num_pairs, returns all
    combinations. Otherwise, randomly samples to hit the target count.

    Args:
        strings: Source strings to pair.
        num_pairs: Desired number of pairs.
        seed: Random seed for reproducibility.

    Returns:
        List of (string1, string2) tuples.
    """
    max_pairs = len(strings) * (len(strings) - 1) // 2

    if max_pairs <= num_pairs:
        return list(combinations(strings, 2))

    # Generate more strings if needed to reach pair count
    rng = random.Random(seed)
    all_pairs = list(combinations(strings, 2))
    rng.shuffle(all_pairs)
    return all_pairs[:num_pairs]


# ----------------------------------------------------------------------------
# EXECUTION STRATEGIES
# ----------------------------------------------------------------------------
def run_sequential(pairs: List[Tuple[str, str]]) -> int:
    """Single-threaded baseline."""
    count = 0
    for s1, s2 in pairs:
        score = similarity_rapidfuzz(s1, s2)
        if score >= 70.0:
            count += 1
    return count


def _joblib_chunk_worker(chunk: List[Tuple[str, str]]) -> int:
    """Process a chunk of pairs (Joblib target)."""
    count = 0
    for s1, s2 in chunk:
        score = similarity_rapidfuzz(s1, s2)
        if score >= 70.0:
            count += 1
    return count


def run_joblib(
    pairs: List[Tuple[str, str]],
    n_jobs: int,
    chunk_size: int,
) -> int:
    """Joblib parallel execution with loky backend."""
    chunks = [
        pairs[i : i + chunk_size]
        for i in range(0, len(pairs), chunk_size)
    ]
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_joblib_chunk_worker)(chunk) for chunk in chunks
    )
    return sum(results)


def _mp_chunk_worker(args) -> int:
    """Process a chunk of pairs (multiprocessing target)."""
    chunk = args
    count = 0
    for s1, s2 in chunk:
        score = similarity_rapidfuzz(s1, s2)
        if score >= 70.0:
            count += 1
    return count


def run_multiprocessing(
    pairs: List[Tuple[str, str]],
    num_workers: int,
    chunk_size: int,
) -> int:
    """Native multiprocessing.Pool with imap_unordered."""
    chunks = [
        pairs[i : i + chunk_size]
        for i in range(0, len(pairs), chunk_size)
    ]
    total = 0
    with mp.Pool(processes=num_workers) as pool:
        for count in pool.imap_unordered(_mp_chunk_worker, chunks):
            total += count
    return total


# ----------------------------------------------------------------------------
# BENCHMARK RUNNER
# ----------------------------------------------------------------------------
def run_benchmark(
    pair_sizes: List[int],
    core_counts: List[int],
    chunk_size: int,
    runs: int,
    string_length: int,
) -> List[dict]:
    """
    Run the full benchmark suite.

    For each pair_size:
      - Runs sequential baseline (always single-threaded)
      - Runs Joblib and multiprocessing at each core count

    Each configuration is repeated `runs` times and averaged.

    Returns:
        List of result dictionaries.
    """
    results = []

    # Pre-generate a large enough string pool
    max_pairs = max(pair_sizes)
    # Rough estimate: n strings where C(n,2) >= max_pairs
    n_strings = int((1 + (1 + 8 * max_pairs) ** 0.5) / 2) + 10
    strings = generate_strings(n_strings, length=string_length)

    for num_pairs in pair_sizes:
        print(f"\n{'='*60}")
        print(f"  Benchmark: {num_pairs:,} pairs")
        print(f"{'='*60}")

        pairs = generate_pairs(strings, num_pairs)
        actual_pairs = len(pairs)
        print(f"  Actual pairs generated: {actual_pairs:,}")

        # --- Sequential baseline ---
        print(f"\n  [Sequential] Running {runs} iterations...")
        seq_times = []
        for r in range(runs):
            t0 = time.perf_counter()
            run_sequential(pairs)
            elapsed = time.perf_counter() - t0
            seq_times.append(elapsed)
            print(f"    Run {r+1:02d}: {elapsed:.4f}s")

        seq_avg = np.mean(seq_times)
        print(f"  Sequential avg: {seq_avg:.4f}s")

        results.append({
            "pairs": actual_pairs,
            "method": "Sequential",
            "cores": 1,
            "chunk_size": "-",
            "avg_time": seq_avg,
            "speedup": 1.0,
            "throughput": actual_pairs / seq_avg,
        })

        # --- Joblib and Multiprocessing at each core count ---
        for cores in core_counts:
            for method_name, runner in [
                ("Joblib", lambda p, c: run_joblib(p, c, chunk_size)),
                ("Multiprocessing", lambda p, c: run_multiprocessing(p, c, chunk_size)),
            ]:
                print(f"\n  [{method_name}, {cores} cores] Running {runs} iterations...")
                times = []
                for r in range(runs):
                    t0 = time.perf_counter()
                    runner(pairs, cores)
                    elapsed = time.perf_counter() - t0
                    times.append(elapsed)
                    print(f"    Run {r+1:02d}: {elapsed:.4f}s")

                avg_time = np.mean(times)
                speedup = seq_avg / avg_time if avg_time > 0 else 0
                print(f"  {method_name} avg: {avg_time:.4f}s (speedup: {speedup:.2f}x)")

                results.append({
                    "pairs": actual_pairs,
                    "method": method_name,
                    "cores": cores,
                    "chunk_size": chunk_size,
                    "avg_time": avg_time,
                    "speedup": speedup,
                    "throughput": actual_pairs / avg_time if avg_time > 0 else 0,
                })

    return results


# ----------------------------------------------------------------------------
# OUTPUT FORMATTING
# ----------------------------------------------------------------------------
def format_report(
    results: List[dict],
    pair_sizes: List[int],
    core_counts: List[int],
    chunk_size: int,
    runs: int,
    string_length: int,
) -> str:
    """Format benchmark results as a readable text report."""
    lines = []
    lines.append("=" * 72)
    lines.append("  PARALLELIZATION STRATEGY BENCHMARK")
    lines.append("=" * 72)

    # Environment
    lines.append("")
    lines.append("---- Environment ----")
    lines.append(f"Timestamp            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Python Version       : {platform.python_version()}")
    lines.append(f"Platform             : {platform.platform()}")
    lines.append(f"Processor            : {platform.processor()}")
    lines.append(f"Available CPUs       : {os.cpu_count()}")

    # Config
    lines.append("")
    lines.append("---- Benchmark Config ----")
    lines.append(f"Pair Sizes           : {', '.join(f'{s:,}' for s in pair_sizes)}")
    lines.append(f"Core Counts          : {', '.join(str(c) for c in core_counts)}")
    lines.append(f"Chunk Size           : {chunk_size:,}")
    lines.append(f"String Length        : {string_length}")
    lines.append(f"Runs per config      : {runs}")
    lines.append(f"Algorithm            : RapidFuzz (multi-metric average)")

    # Results table
    lines.append("")
    lines.append("---- Results ----")
    lines.append("")

    header = f"{'Pairs':>10} | {'Method':<18} | {'Cores':>5} | {'Chunk':>7} | {'Avg Time':>10} | {'Speedup':>8} | {'Throughput':>15}"
    separator = "-" * len(header)
    lines.append(header)
    lines.append(separator)

    for r in results:
        chunk_str = f"{r['chunk_size']:,}" if isinstance(r["chunk_size"], int) else r["chunk_size"]
        lines.append(
            f"{r['pairs']:>10,} | {r['method']:<18} | {r['cores']:>5} | "
            f"{chunk_str:>7} | {r['avg_time']:>9.4f}s | "
            f"{r['speedup']:>7.2f}x | "
            f"{r['throughput']:>12,.0f}/s"
        )

    # Key takeaways section (placeholder for the author to fill in)
    lines.append("")
    lines.append("---- Observations ----")
    lines.append("(Fill in after reviewing results)")
    lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark parallelization strategies for string similarity"
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int,
        default=[1_000, 5_000, 10_000, 50_000, 100_000],
        help="Number of pairwise comparisons to benchmark",
    )
    parser.add_argument(
        "--cores", nargs="+", type=int,
        default=[2, 4, 8],
        help="Core counts to test",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=1_000,
        help="Pairs per chunk for parallel methods",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Repetitions per configuration",
    )
    parser.add_argument(
        "--string_length", type=int, default=24,
        help="Character length of generated strings",
    )
    parser.add_argument(
        "--output", type=str, default="/home/kunwonsaw/IAMAD_PJT/transaction-similarity-analyzer/benchmarks/results/parallelization_benchmark_results.txt",
        help="Path to save results text file",
    )

    args = parser.parse_args()

    print(f"Starting benchmark with {len(args.sizes)} size configs "
          f"x {len(args.cores)} core configs x {args.runs} runs each...")
    print(f"This may take a while for large pair sizes.\n")

    results = run_benchmark(
        pair_sizes=args.sizes,
        core_counts=args.cores,
        chunk_size=args.chunk_size,
        runs=args.runs,
        string_length=args.string_length,
    )

    report = format_report(
        results=results,
        pair_sizes=args.sizes,
        core_counts=args.cores,
        chunk_size=args.chunk_size,
        runs=args.runs,
        string_length=args.string_length,
    )

    print(f"\n\n{report}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
