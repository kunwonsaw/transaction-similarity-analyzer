"""
Quick Start Example
====================
Demonstrates the SimilarityAnalyzer pipeline end-to-end using
synthetic procurement data. No proprietary data required.

Usage:
    pip install -e .
    python examples/quick_start.py
"""

import random
import pandas as pd
from pathlib import Path

from similarity_analyzer import SimilarityAnalyzer, AnalysisConfig


# ----------------------------------------------------------------------------
# SYNTHETIC DATA GENERATION
# ----------------------------------------------------------------------------
def generate_sample_data(n_records: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic-looking procurement transaction records.

    Creates a mix of exact duplicates, near-duplicates, and unrelated
    items to demonstrate all matching scenarios.
    """
    rng = random.Random(seed)

    # Base specifications with realistic part descriptions
    base_specs = [
        "Bearing 6205-2RS sealed radial",
        "Hydraulic pump motor 15kW 3-phase",
        "V-Belt A68 industrial grade",
        "Filter element HF-6700 10 micron",
        "Gearbox oil SAE 90 20L drum",
        "Coupling flexible jaw L-100",
        "Seal kit cylinder 80x50mm",
        "Conveyor roller 600mm steel",
        "Solenoid valve 24VDC 3/4 inch",
        "Thermocouple type K 300mm probe",
        "Bearing 6206-ZZ shielded radial",
        "Hydraulic pump motor 18.5kW 3-phase",
        "V-Belt B72 industrial grade",
        "Filter element HF-6710 20 micron",
        "Gearbox oil SAE 140 20L drum",
    ]

    # Variations for near-duplicate generation
    def create_variant(spec: str) -> str:
        """Apply small random modifications to a specification."""
        modifications = [
            lambda s: s.replace("sealed", "shielded"),
            lambda s: s.replace("industrial", "heavy-duty"),
            lambda s: s.replace("steel", "stainless"),
            lambda s: s.replace("20L", "25L"),
            lambda s: s.replace("10 micron", "15 micron"),
            lambda s: s + " (import)",
            lambda s: "S/P " + s,
            lambda s: s.upper(),
        ]
        mod = rng.choice(modifications)
        return mod(spec)

    suppliers = [
        "Samsung Engineering Co.",
        "Doosan Heavy Industries(주)",
        "Hyundai Robotics Inc.",
        "LS Electric 주식회사",
        "Hanwha Solutions LLC",
        "KCC Corporation",
        "TechParts Global Ltd",
    ]

    manufacturers = [
        "SKF", "NSK", "FAG", "NTN",
        "Parker Hannifin", "Bosch Rexroth",
        "Gates", "Danfoss", "Festo",
    ]

    departments = [
        "Maintenance Dept A",
        "Maintenance Dept B",
        "Production Line 1",
        "Production Line 2",
        "Utilities",
    ]

    records = []
    for i in range(n_records):
        base_spec = rng.choice(base_specs)

        # 30% chance of creating a near-duplicate variant
        spec = create_variant(base_spec) if rng.random() < 0.3 else base_spec

        supplier = rng.choice(suppliers)
        manufacturer = rng.choice(manufacturers)
        department = rng.choice(departments)

        # Generate realistic pricing with occasional large discrepancies
        base_price = rng.uniform(5_000, 500_000)
        if rng.random() < 0.1:
            # 10% chance of inflated price (simulates potential fraud signal)
            base_price *= rng.uniform(1.5, 3.0)

        quantity = rng.randint(1, 50)

        # Generate dates across 2023-2024
        year = rng.choice([2023, 2024])
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)

        records.append({
            "order_item_number": f"PO-{year}-{i+1:05d}",
            "main_spec": spec,
            "department_name": department,
            "supplier_name": supplier,
            "manufacturer_name": manufacturer,
            "unit_price": round(base_price, 2),
            "order_quantity": quantity,
            "total": round(base_price * quantity, 2),
            "cost_type": rng.choice(["repair", "consumable"]),
            "delivery_completion_date": f"{year}-{month:02d}-{day:02d}",
        })

    return pd.DataFrame(records)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Transaction Similarity Analyzer — Quick Start")
    print("=" * 60)

    # 1. Generate synthetic data
    print("\n1. Generating synthetic procurement data...")
    df = generate_sample_data(n_records=200)
    print(f"   Created {len(df)} records with {df['main_spec'].nunique()} unique specs")

    # Save to temporary CSV
    temp_csv = Path("examples/sample_data.csv")
    temp_csv.parent.mkdir(exist_ok=True)
    df.to_csv(temp_csv, index=False, encoding="utf-8-sig")
    print(f"   Saved to {temp_csv}")

    # 2. Configure and run analysis
    print("\n2. Configuring analyzer...")
    config = AnalysisConfig(
        min_threshold=70.0,
        max_threshold=99.999,
        chunk_size=500,
        cores_count=4,
        algorithm="RapidFuzz",
        clean_specifications=True,
        save_perfect_matches=True,
        product_type="All",
    )

    analyzer = SimilarityAnalyzer(config)

    print("\n3. Loading data...")
    analyzer.load_csv_data(str(temp_csv))

    print("\n4. Preprocessing...")
    analyzer.preprocess()

    print("\n5. Running similarity analysis (Joblib)...")
    results = analyzer.calculate_similarity(use_multiprocessing=False)

    # 3. Show results
    print("\n6. Results summary:")
    stats = analyzer.get_summary_stats()
    print(f"   Total matches found  : {stats['total_matches']:,}")
    print(f"   Average similarity   : {stats['avg_similarity']:.1f}%")
    print(f"   Similarity range     : {stats['similarity_range']['min']:.1f}% — "
          f"{stats['similarity_range']['max']:.1f}%")

    if "unit_price_diff_stats" in stats:
        print(f"   Avg price difference : {stats['unit_price_diff_stats']['avg']:,.0f}")
        print(f"   Max price difference : {stats['unit_price_diff_stats']['max']:,.0f}")

    if "perfect_matches" in stats:
        print(f"   Perfect matches      : {stats['perfect_matches']['count']:,}")

    # 4. Preview top matches
    if len(results) > 0:
        print("\n7. Top 5 matches by similarity:")
        top = results.nlargest(5, "similarity")
        for _, row in top.iterrows():
            spec1 = str(row.get("main_spec1", ""))[:40]
            spec2 = str(row.get("main_spec2", ""))[:40]
            score = row["similarity"]
            print(f"   {score:5.1f}% | {spec1:<40} ↔ {spec2}")

    # 5. Save results
    print("\n8. Saving results...")
    output_dir = "examples/output"
    analyzer.save_results(output_directory=output_dir)
    print(f"   Results saved to {output_dir}/")

    print("\n✅ Quick start complete!")


if __name__ == "__main__":
    main()
