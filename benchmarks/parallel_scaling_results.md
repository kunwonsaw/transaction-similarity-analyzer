## Parallel Scaling Benchmark (RapidFuzz)

Dataset: 5,000 rows (sample_procurement_data_small.csv)
Machine: i9-13900KS / 128GB RAM
Python: 3.10.12

CLI Command Used:

python3 tds_multiproc_joblib.py \
  --mode csv \
  --csv_path sample_procurement_data_small.csv \
  --cores_count 1 \
  --chunk_size 500 \ 
  --min_threshold 70 \
  --max_threshold 99 \
  --algorithm RapidFuzz \
  --verbosity 1

| Cores | Runtime (s) | Speedup |
|------|--------------|---------|
| 1    | 62           | 1.0×    |
| 8    | 10           | 6.2×    |
| 16   | 7            | 8.9×    |

Observations:
- Diminishing returns becoming noticeable when comparing 8-cores vs. 16-cores
