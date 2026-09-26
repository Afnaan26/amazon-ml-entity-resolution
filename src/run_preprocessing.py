import pandas as pd
from preprocessing import preprocess_dataframe


# ============================================================
# FILE PATHS
# ============================================================

S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Source 1...")
s1 = pd.read_csv(S1_PATH, sep="\t")

print("Loading Source 2...")
s2 = pd.read_csv(S2_PATH, sep="\t")

print("Loading Source 3...")
s3 = pd.read_csv(S3_PATH, sep="\t")


print("\nLoaded successfully!")

print("S1 shape:", s1.shape)
print("S2 shape:", s2.shape)
print("S3 shape:", s3.shape)


# ============================================================
# PREPROCESS
# ============================================================

print("\nPreprocessing Source 1...")
s1_processed = preprocess_dataframe(s1)

print("Preprocessing Source 2...")
s2_processed = preprocess_dataframe(s2)

print("Preprocessing Source 3...")
s3_processed = preprocess_dataframe(s3)


# ============================================================
# DISPLAY SAMPLE
# ============================================================

print("\n========== SOURCE 1 SAMPLE ==========")

print(
    s1_processed[
        [
            "entity_id",
            "business_name",
            "name_norm",
            "business_address",
            "address_norm",
            "country_norm"
        ]
    ].head(10).to_string(index=False)
)


print("\n========== SOURCE 2 SAMPLE ==========")

print(
    s2_processed[
        [
            "entity_id",
            "business_name",
            "name_norm",
            "business_address",
            "address_norm",
            "country_norm"
        ]
    ].head(10).to_string(index=False)
)


print("\n========== SOURCE 3 SAMPLE ==========")

print(
    s3_processed[
        [
            "entity_id",
            "business_name",
            "name_norm",
            "business_address",
            "address_norm",
            "country_norm"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

for name, df in [
    ("S1", s1_processed),
    ("S2", s2_processed),
    ("S3", s3_processed)
]:

    print(f"\n{name}")

    print(
        df[
            [
                "name_norm",
                "address_norm",
                "country_norm"
            ]
        ].isna().sum()
    )


# ============================================================
# NORMALIZED EMPTY VALUES
# ============================================================

print("\n========== EMPTY NORMALIZED VALUES ==========")

for name, df in [
    ("S1", s1_processed),
    ("S2", s2_processed),
    ("S3", s3_processed)
]:

    print(f"\n{name}")

    print(
        "Empty names:",
        (df["name_norm"] == "").sum()
    )

    print(
        "Empty addresses:",
        (df["address_norm"] == "").sum()
    )

    print(
        "Empty countries:",
        (df["country_norm"] == "").sum()
    )