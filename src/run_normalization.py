"""
Run full dataset preprocessing and normalization for all train and test sources.
Outputs *_clean.parquet and *_clean.tsv files.
"""

import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.common.normalize import preprocess_dataframe

DATASET_DIR = "dataset"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = [
    ("train", "train_source1"),
    ("train", "train_source2"),
    ("train", "train_source3"),
    ("test", "test_source1"),
    ("test", "test_source2"),
    ("test", "test_source3"),
]


def run():
    print("=== STARTING FAST PREPROCESSING & NORMALIZATION PIPELINE ===")
    t_start = time.time()

    summary_stats = []

    for split, name in FILES:
        print(f"\nProcessing {split}/{name}...")
        t0 = time.time()

        parquet_in = os.path.join(DATASET_DIR, split, f"{name}.parquet")
        df = pd.read_parquet(parquet_in)
        print(f"Loaded {len(df):,} rows in {time.time() - t0:.2f}s")

        t1 = time.time()
        df_clean = preprocess_dataframe(df)
        print(f"Normalized in {time.time() - t1:.2f}s")

        tsv_out = os.path.join(DATASET_DIR, split, f"{name}_clean.tsv")
        parquet_out = os.path.join(DATASET_DIR, split, f"{name}_clean.parquet")

        t2 = time.time()
        # Save parquet first (lightning fast)
        df_clean.to_parquet(parquet_out, engine="pyarrow")
        print(f"Saved {name}_clean.parquet in {time.time() - t2:.2f}s")

        t3 = time.time()
        # Save clean TSV
        df_clean.to_csv(tsv_out, sep="\t", index=False)
        print(f"Saved {name}_clean.tsv in {time.time() - t3:.2f}s")

        stat = {
            "name": name,
            "total_rows": len(df_clean),
            "empty_names": int((df_clean["name_norm"] == "").sum()),
            "empty_addresses": int((df_clean["address_norm"] == "").sum()),
            "missing_postal_codes": int((~df_clean["has_postal_code"]).sum()),
            "has_postal_code_cnt": int(df_clean["has_postal_code"].sum()),
            "has_postal_code_pct": float(df_clean["has_postal_code"].mean() * 100),
            "avg_name_len": float(df_clean["name_len"].mean()),
            "avg_address_len": float(df_clean["address_len"].mean()),
        }
        summary_stats.append(stat)

    print("\n" + "=" * 60)
    print("PREPROCESSING & NORMALIZATION SUMMARY TABLE")
    print("=" * 60)
    summary_df = pd.DataFrame(summary_stats)
    print(summary_df.to_string(index=False))

    summary_df.to_csv(os.path.join(OUTPUT_DIR, "normalization_summary.csv"), index=False)
    print(f"\nAll clean datasets saved! Total pipeline runtime: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    run()
