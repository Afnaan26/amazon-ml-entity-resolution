import pandas as pd
import os
import time

dataset_dir = "dataset"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")

files = [
    (train_dir, "train_source1"),
    (train_dir, "train_source2"),
    (train_dir, "train_source3"),
    (train_dir, "train_ground_truth"),
    (test_dir, "test_source1"),
    (test_dir, "test_source2"),
    (test_dir, "test_source3"),
]

for folder, name in files:
    tsv_path = os.path.join(folder, f"{name}.tsv")
    parquet_path = os.path.join(folder, f"{name}.parquet")
    if os.path.exists(tsv_path) and not os.path.exists(parquet_path):
        t0 = time.time()
        print(f"Converting {name}.tsv -> parquet...")
        df = pd.read_csv(tsv_path, sep="\t")
        df.to_parquet(parquet_path, engine="pyarrow")
        print(f"Done {name} in {time.time()-t0:.2f}s, shape: {df.shape}")
    else:
        print(f"Skipping or already exists: {parquet_path}")

print("All parquet conversions ready!")
