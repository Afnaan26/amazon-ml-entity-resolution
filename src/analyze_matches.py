import pandas as pd
from preprocessing import preprocess_dataframe


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("Data loaded.")


# ============================================================
# PREPROCESS
# ============================================================

print("Preprocessing...")

s1 = preprocess_dataframe(s1)
s2 = preprocess_dataframe(s2)
s3 = preprocess_dataframe(s3)


# ============================================================
# CREATE LOOKUP DICTIONARIES
# ============================================================

s1_idx = s1.set_index("entity_id")
s2_idx = s2.set_index("entity_id")
s3_idx = s3.set_index("entity_id")


# ============================================================
# ANALYZE GROUND TRUTH
# ============================================================

total_matches = 0

name_exact = 0
address_exact = 0

name_only = 0
address_only = 0
both_exact = 0
neither_exact = 0

missing_match_records = 0


for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    matched_ids = str(row["matched_entity_ids"]).split(",")

    source1 = s1_idx.loc[s1_id]

    for match_id in matched_ids:

        match_id = match_id.strip()

        if match_id.startswith("S2-"):
            if match_id not in s2_idx.index:
                continue

            match = s2_idx.loc[match_id]

        elif match_id.startswith("S3-"):
            if match_id not in s3_idx.index:
                continue

            match = s3_idx.loc[match_id]

        else:
            continue

        total_matches += 1

        name_same = (
            source1["name_norm"] != ""
            and match["name_norm"] != ""
            and source1["name_norm"] == match["name_norm"]
        )

        address_same = (
            source1["address_norm"] != ""
            and match["address_norm"] != ""
            and source1["address_norm"] == match["address_norm"]
        )

        if name_same:
            name_exact += 1

        if address_same:
            address_exact += 1

        if name_same and address_same:
            both_exact += 1

        elif name_same:
            name_only += 1

        elif address_same:
            address_only += 1

        else:
            neither_exact += 1


# ============================================================
# RESULTS
# ============================================================

print("\n======================================")
print("GROUND TRUTH MATCH ANALYSIS")
print("======================================")

print("Total true matches:", total_matches)

print("\nExact normalized name:")
print(
    name_exact,
    f"({name_exact / total_matches:.2%})"
)

print("\nExact normalized address:")
print(
    address_exact,
    f"({address_exact / total_matches:.2%})"
)

print("\nBoth name and address exact:")
print(
    both_exact,
    f"({both_exact / total_matches:.2%})"
)

print("\nName only exact:")
print(
    name_only,
    f"({name_only / total_matches:.2%})"
)

print("\nAddress only exact:")
print(
    address_only,
    f"({address_only / total_matches:.2%})"
)

print("\nNeither exact:")
print(
    neither_exact,
    f"({neither_exact / total_matches:.2%})"
)