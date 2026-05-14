import time
import random
import string
import statistics
import platform 
import sys

from difflib import SequenceMatcher
from rapidfuzz import fuzz
from datetime import datetime
from pathlib import Path

# ----------------------------
# Benchmark configuration
# ----------------------------
N_COMPARISONS = 100_000
STRING_LENGTH = 24
N_RUNS = 5  # repeat to smooth noise

# ----------------------------
# Helpers
# ----------------------------
def random_string(n=STRING_LENGTH):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))

def generate_pairs(n):
    return [(random_string(), random_string()) for _ in range(n)]

def bench_sequence_matcher(pairs):
    start = time.perf_counter()
    for a, b in pairs:
        SequenceMatcher(None, a, b).ratio()
    return time.perf_counter() - start

def bench_rapidfuzz(pairs):
    start = time.perf_counter()
    for a, b in pairs:
        fuzz.QRatio(a, b)
    return time.perf_counter() - start

def write_results(
        output_path,
        seq_times,
        rf_times,
        seq_avg,
        rf_avg
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("=== STRING SIMILARITY PERFORMANCE BENCHMARK (MICRO) ===\n\n")

        f.write("---- Environment ----\n")
        f.write(f"Timestamp            : {now}\n")
        f.write(f"Python Version       : {sys.version.split()[0]}\n")
        f.write(f"Platform             : {platform.system()} {platform.release()}\n")
        f.write(f"Machine              : {platform.machine()}\n")
        f.write(f"Processor            : {platform.processor()}\n\n")

        f.write("---- Benchmark Config ----\n")
        f.write(f"Comparisons          : {N_COMPARISONS:,}\n")
        f.write(f"String Length        : {STRING_LENGTH}\n")
        f.write(f"Runs per method      : {N_RUNS}\n\n")

        f.write("---- Per-Run Timings (seconds) ----\n")
        for i, (s, r) in enumerate(zip(seq_times, rf_times), start=1):
            f.write(f"Run {i:02d} | SequenceMatcher: {s:.6f} | Rapidfuzz: {r: .6f}\n")
        f.write("\n")

        f.write("---- Averages ----\n")
        f.write(f"SequenceMatcher Avg   : {seq_avg:.6f} sec\n")
        f.write(f"RapidFuzz Avg         : {rf_avg: .6f} sec\n")
        f.write(f"Speed-up              : {seq_avg / rf_avg:.2f}x\n\n")
        
        f.write("---- Throughput ----\n")
        f.write(f"SequenceMatcher       : {N_COMPARISONS / seq_avg:,.0f} comparisons/sec\n")
        f.write(f"RapidFuzz             : {N_COMPARISONS / rf_avg:,.0f} comparisons/sec\n")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print(f"\nRunning micro-benchmark with {N_COMPARISONS:,} comparisons")
    print(f"String length: {STRING_LENGTH}")
    print(f"Runs per method: {N_RUNS}\n")

    pairs = generate_pairs(N_COMPARISONS)

    seq_times = []
    rf_times = []

    for i in range(N_RUNS):
        t_seq = bench_sequence_matcher(pairs)
        t_rf = bench_rapidfuzz(pairs)

        seq_times.append(t_seq)
        rf_times.append(t_rf)

        print(f"Run {i+1}:")
        print(f"  SequenceMatcher: {t_seq:.4f} sec")
        print(f"  RapidFuzz      : {t_rf:.4f} sec\n")

    seq_avg = statistics.mean(seq_times)
    rf_avg = statistics.mean(rf_times)

    print("==== AVERAGE RESULTS ====")
    print(f"SequenceMatcher avg: {seq_avg:.4f} sec")
    print(f"RapidFuzz avg      : {rf_avg:.4f} sec")
    print(f"Speedup            : {seq_avg / rf_avg:.2f}x")

    print("\n==== THROUGHPUT ====")
    print(f"SequenceMatcher: {N_COMPARISONS / seq_avg:,.0f} comparisons/sec")
    print(f"RapidFuzz      : {N_COMPARISONS / rf_avg:,.0f} comparisons/sec")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results_path = results_dir / "micro_100k_results.txt"

    write_results(
        output_path=results_path,
        seq_times=seq_times,
        rf_times=rf_times,
        seq_avg=seq_avg,
        rf_avg = rf_avg,
    )

    print(f"\nResults written to: {results_path.resolve()}\n")
