import pandas as pd
import numpy as np
import re
import os
import sys

# Ensure UTF-8 output encoding for standard output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

os.makedirs("docs", exist_ok=True)
out_file = open("docs/eda_findings.txt", "w", encoding="utf-8")

def log(msg=""):
    print(msg)
    out_file.write(str(msg) + "\n")
    out_file.flush()

S1_PATH = "dataset/train/train_source1.parquet" if os.path.exists("dataset/train/train_source1.parquet") else "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.parquet" if os.path.exists("dataset/train/train_source2.parquet") else "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.parquet" if os.path.exists("dataset/train/train_source3.parquet") else "dataset/train/train_source3.tsv"
GT_PATH = "dataset/train/train_ground_truth.parquet" if os.path.exists("dataset/train/train_ground_truth.parquet") else "dataset/train/train_ground_truth.tsv"

TEST_S1_PATH = "dataset/test/test_source1.parquet" if os.path.exists("dataset/test/test_source1.parquet") else "dataset/test/test_source1.tsv"
TEST_S2_PATH = "dataset/test/test_source2.parquet" if os.path.exists("dataset/test/test_source2.parquet") else "dataset/test/test_source2.tsv"
TEST_S3_PATH = "dataset/test/test_source3.parquet" if os.path.exists("dataset/test/test_source3.parquet") else "dataset/test/test_source3.tsv"

def read_df(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t")

log("=== LOADING TRAIN DATASETS ===")
s1 = read_df(S1_PATH)
s2 = read_df(S2_PATH)
s3 = read_df(S3_PATH)
gt = read_df(GT_PATH)

log(f"Train S1 rows: {len(s1)}, cols: {list(s1.columns)}")
log(f"Train S2 rows: {len(s2)}, cols: {list(s2.columns)}")
log(f"Train S3 rows: {len(s3)}, cols: {list(s3.columns)}")
log(f"Train GT rows: {len(gt)}, cols: {list(gt.columns)}")

log("\n=== LOADING TEST DATASETS ===")
test_s1 = read_df(TEST_S1_PATH)
test_s2 = read_df(TEST_S2_PATH)
test_s3 = read_df(TEST_S3_PATH)
log(f"Test S1 rows: {len(test_s1)}")
log(f"Test S2 rows: {len(test_s2)}")
log(f"Test S3 rows: {len(test_s3)}")

log("\n=== NULL RATES IN TRAIN ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    log(f"\n{name} Null Counts & Rates:")
    for col in df.columns:
        null_cnt = df[col].isna().sum()
        null_rate = null_cnt / len(df) * 100
        log(f"  {col}: {null_cnt} ({null_rate:.2f}%)")

log("\n=== DUPLICATE ID CHECK ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    dup_ids = df["entity_id"].duplicated().sum()
    log(f"  {name} Duplicate entity_ids: {dup_ids}")

log("\n=== GROUND TRUTH MATCH COUNT DISTRIBUTION ===")
gt['matched_list'] = gt['matched_entity_ids'].apply(lambda x: [i.strip() for i in str(x).split(',') if i.strip()])
gt['match_count'] = gt['matched_list'].apply(len)
match_dist = gt['match_count'].value_counts().sort_index()
log("Matches per S1 entity distribution:")
log(match_dist.to_string())

# Singletons (entities with 0 true matches in GT)
all_s1_ids = set(s1['entity_id'])
gt_s1_ids = set(gt['source1_entity_id'])
missing_in_gt = len(all_s1_ids - gt_s1_ids)
log(f"\nS1 entities total: {len(all_s1_ids)}, S1 entities in GT: {len(gt_s1_ids)}, S1 entities missing from GT: {missing_in_gt}")

# Matches breakdown S2 vs S3
s2_matches_per_s1 = gt['matched_list'].apply(lambda lst: sum(1 for item in lst if item.startswith("S2-")))
s3_matches_per_s1 = gt['matched_list'].apply(lambda lst: sum(1 for item in lst if item.startswith("S3-")))

log("\nS2 matches per S1 entity distribution:")
log(s2_matches_per_s1.value_counts().sort_index().to_string())
log("\nS3 matches per S1 entity distribution:")
log(s3_matches_per_s1.value_counts().sort_index().to_string())

log("\n=== COUNTRY DISTRIBUTION IN TRAIN ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    log(f"\n{name} Country Top 10:")
    log(df['country'].value_counts(dropna=False).head(10).to_string())

log("\n=== COUNTRY DISTRIBUTION IN TEST ===")
for name, df in [("Test S1", test_s1), ("Test S2", test_s2), ("Test S3", test_s3)]:
    log(f"\n{name} Country Top 10:")
    log(df['country'].value_counts(dropna=False).head(10).to_string())

log("\n=== NOISE SAMPLING (60 GROUND TRUTH MATCH PAIRS) ===")
# Efficient sampling: gather needed IDs for the first 100 rows of gt to avoid converting 12.5M rows into dicts
needed_s1 = set()
needed_matches = set()
for _, row in gt.head(100).iterrows():
    needed_s1.add(row['source1_entity_id'])
    for m in row['matched_list']:
        needed_matches.add(m)

s1_map = s1[s1['entity_id'].isin(needed_s1)].set_index("entity_id").to_dict(orient="index")
s2_map = s2[s2['entity_id'].isin(needed_matches)].set_index("entity_id").to_dict(orient="index")
s3_map = s3[s3['entity_id'].isin(needed_matches)].set_index("entity_id").to_dict(orient="index")

sample_file = open("docs/sample_pairs.txt", "w", encoding="utf-8")
count = 0
for _, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    rec1 = s1_map.get(s1_id)
    if not rec1: continue
    
    for match_id in row['matched_list']:
        if match_id.startswith("S2-"):
            rec2 = s2_map.get(match_id)
        elif match_id.startswith("S3-"):
            rec2 = s3_map.get(match_id)
        else:
            rec2 = None
        if rec2:
            count += 1
            sample_file.write(f"Pair {count}: S1 ({s1_id}) <-> Match ({match_id})\n")
            sample_file.write(f"  S1 Name:    {rec1.get('business_name')}\n")
            sample_file.write(f"  Match Name: {rec2.get('business_name')}\n")
            sample_file.write(f"  S1 Addr:    {rec1.get('business_address')}\n")
            sample_file.write(f"  Match Addr: {rec2.get('business_address')}\n")
            sample_file.write(f"  S1 Country: {rec1.get('country')} | Match Country: {rec2.get('country')}\n")
            sample_file.write("-" * 60 + "\n")
            if count >= 60:
                break
    if count >= 60:
        break

sample_file.close()
log(f"Wrote {count} sample pairs to docs/sample_pairs.txt")

log("\n=== DEEP EDA: MATCH RATE RELATIONSHIPS (TASK 8) ===")
log("Definition of Metrics:")
log("1. S1 Entity Metrics: Mean matches per entity, % of S1 entities with >= 1 S2 match, % with >= 1 S3 match.")
log("2. Candidate Source Metrics: % of candidate records (S2, S3) that appear in Ground Truth matches.")

# S1 feature extraction for deep EDA
s1_feat = pd.DataFrame({
    'entity_id': s1['entity_id'],
    'country': s1['country'].fillna('UNKNOWN').str.upper().str.strip(),
    'name_len': s1['business_name'].fillna('').str.len(),
    'has_name': s1['business_name'].notna() & (s1['business_name'].str.strip() != ''),
    'has_address': s1['business_address'].notna() & (s1['business_address'].str.strip() != ''),
    'has_postal_code': s1['business_address'].fillna('').str.contains(r'\b\d{5,6}\b', regex=True)
})
gt_stat = pd.DataFrame({
    'entity_id': gt['source1_entity_id'],
    'total_matches': gt['match_count'],
    'has_s2': s2_matches_per_s1 > 0,
    'has_s3': s3_matches_per_s1 > 0
})
s1_merged = s1_feat.merge(gt_stat, on='entity_id', how='left')

log("\n--- A. S1 Country vs Match Rate ---")
s1_country_eda = s1_merged.groupby('country').agg(
    total_entities=('entity_id', 'count'),
    mean_matches=('total_matches', 'mean'),
    pct_with_s2=('has_s2', lambda x: x.mean() * 100),
    pct_with_s3=('has_s3', lambda x: x.mean() * 100)
)
log(s1_country_eda.to_string())

log("\n--- B. S1 Name Length Buckets vs Match Rate ---")
bins = [-1, 0, 10, 20, 30, 50, 10000]
labels = ['0', '1-10', '11-20', '21-30', '31-50', '51+']
s1_merged['name_bucket'] = pd.cut(s1_merged['name_len'], bins=bins, labels=labels)
s1_name_eda = s1_merged.groupby('name_bucket', observed=False).agg(
    total_entities=('entity_id', 'count'),
    mean_matches=('total_matches', 'mean'),
    pct_with_s2=('has_s2', lambda x: x.mean() * 100),
    pct_with_s3=('has_s3', lambda x: x.mean() * 100)
)
log(s1_name_eda.to_string())

log("\n--- C. S1 Address Presence vs Match Rate ---")
s1_addr_eda = s1_merged.groupby('has_address').agg(
    total_entities=('entity_id', 'count'),
    mean_matches=('total_matches', 'mean'),
    pct_with_s2=('has_s2', lambda x: x.mean() * 100),
    pct_with_s3=('has_s3', lambda x: x.mean() * 100)
)
log(s1_addr_eda.to_string())

log("\n--- D. S1 Postal Code Presence vs Match Rate ---")
s1_post_eda = s1_merged.groupby('has_postal_code').agg(
    total_entities=('entity_id', 'count'),
    mean_matches=('total_matches', 'mean'),
    pct_with_s2=('has_s2', lambda x: x.mean() * 100),
    pct_with_s3=('has_s3', lambda x: x.mean() * 100)
)
log(s1_post_eda.to_string())

log("\n--- E. S1 Name Presence vs Match Rate ---")
s1_name_pres_eda = s1_merged.groupby('has_name').agg(
    total_entities=('entity_id', 'count'),
    mean_matches=('total_matches', 'mean'),
    pct_with_s2=('has_s2', lambda x: x.mean() * 100),
    pct_with_s3=('has_s3', lambda x: x.mean() * 100)
)
log(s1_name_pres_eda.to_string())

# Candidate Source (S2 and S3) match rate breakdown
log("\n--- Candidate Sources (S2 / S3) Match Rate Analysis ---")
s2_matched_set = set()
s3_matched_set = set()
for raw in gt['matched_entity_ids']:
    for mid in str(raw).split(','):
        mid = mid.strip()
        if mid.startswith('S2-'): s2_matched_set.add(mid)
        elif mid.startswith('S3-'): s3_matched_set.add(mid)

for s_name, df_cand, matched_set in [("Source 2", s2, s2_matched_set), ("Source 3", s3, s3_matched_set)]:
    log(f"\n{s_name} Candidate Ground Truth Match Rates:")
    is_matched = df_cand['entity_id'].isin(matched_set)
    has_addr = df_cand['business_address'].notna() & (df_cand['business_address'].str.strip() != '')
    has_post = df_cand['business_address'].fillna('').str.contains(r'\b\d{5,6}\b', regex=True)
    has_nm = df_cand['business_name'].notna() & (df_cand['business_name'].str.strip() != '')
    name_len = df_cand['business_name'].fillna('').str.len()
    name_bkt = pd.cut(name_len, bins=bins, labels=labels)
    
    total_cand = len(df_cand)
    total_matched = is_matched.sum()
    log(f"  Overall Match Rate: {total_matched:,} / {total_cand:,} ({total_matched / total_cand * 100:.2f}%)")
    
    # Address presence
    match_by_addr = is_matched.groupby(has_addr).agg(['count', 'mean'])
    log(f"  Match Rate by Address Presence:\n{match_by_addr.to_string()}")
    
    # Postal code presence
    match_by_post = is_matched.groupby(has_post).agg(['count', 'mean'])
    log(f"  Match Rate by Postal Code Presence:\n{match_by_post.to_string()}")
    
    # Name length bucket
    match_by_name = is_matched.groupby(name_bkt, observed=False).agg(['count', 'mean'])
    log(f"  Match Rate by Name Length Bucket:\n{match_by_name.to_string()}")

out_file.close()
print("EDA analysis completed successfully! Output saved to docs/eda_findings.txt and docs/sample_pairs.txt")

