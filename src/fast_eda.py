import pandas as pd
import numpy as np
import re
import json

print("Step 1: Loading Ground Truth...")
gt = pd.read_parquet("dataset/train/train_ground_truth.parquet")
print(f"Ground Truth shape: {gt.shape}")

print("\nStep 2: Analyzing Ground Truth Match Counts...")
gt['matched_list'] = gt['matched_entity_ids'].apply(lambda x: [i.strip() for i in str(x).split(',') if i.strip()])
gt['match_count'] = gt['matched_list'].apply(len)
match_dist = gt['match_count'].value_counts().sort_index().to_dict()
print(f"Matches per S1 entity distribution: {match_dist}")

s2_cnts = gt['matched_list'].apply(lambda lst: sum(1 for item in lst if item.startswith("S2-"))).value_counts().sort_index().to_dict()
s3_cnts = gt['matched_list'].apply(lambda lst: sum(1 for item in lst if item.startswith("S3-"))).value_counts().sort_index().to_dict()
print(f"S2 match count breakdown: {s2_cnts}")
print(f"S3 match count breakdown: {s3_cnts}")

print("\nStep 3: Loading Source 1...")
s1 = pd.read_parquet("dataset/train/train_source1.parquet")
print(f"S1 shape: {s1.shape}")
print(f"S1 nulls: {s1.isna().sum().to_dict()}")
print(f"S1 duplicate entity_ids: {s1['entity_id'].duplicated().sum()}")
print(f"S1 country value counts:\n{s1['country'].value_counts(dropna=False)}")

print("\nStep 4: Checking S1 Entities vs GT...")
s1_all = set(s1['entity_id'])
gt_s1 = set(gt['source1_entity_id'])
print(f"Total S1 entities: {len(s1_all)}")
print(f"S1 entities present in Ground Truth: {len(gt_s1)}")
print(f"S1 entities with 0 matches in Ground Truth (singletons): {len(s1_all - gt_s1)}")

print("\nStep 5: Loading S2...")
s2 = pd.read_parquet("dataset/train/train_source2.parquet")
print(f"S2 shape: {s2.shape}")
print(f"S2 nulls: {s2.isna().sum().to_dict()}")
print(f"S2 duplicate entity_ids: {s2['entity_id'].duplicated().sum()}")
print(f"S2 country top 10:\n{s2['country'].value_counts(dropna=False).head(10)}")

print("\nStep 6: Loading S3...")
s3 = pd.read_parquet("dataset/train/train_source3.parquet")
print(f"S3 shape: {s3.shape}")
print(f"S3 nulls: {s3.isna().sum().to_dict()}")
print(f"S3 duplicate entity_ids: {s3['entity_id'].duplicated().sum()}")
print(f"S3 country top 10:\n{s3['country'].value_counts(dropna=False).head(10)}")

print("\nStep 7: Sampling 60 True Matched Pairs for Noise Cataloguing...")
s1_dict = s1.set_index("entity_id").to_dict(orient="index")
s2_dict = s2.set_index("entity_id").to_dict(orient="index")
s3_dict = s3.set_index("entity_id").to_dict(orient="index")

sample_pairs = []
count = 0
for _, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    r1 = s1_dict.get(s1_id)
    if not r1: continue
    
    for match_id in row['matched_list']:
        if match_id.startswith("S2-"):
            r2 = s2_dict.get(match_id)
        elif match_id.startswith("S3-"):
            r2 = s3_dict.get(match_id)
        else:
            r2 = None
        if r2:
            sample_pairs.append({
                "pair_num": count + 1,
                "s1_id": s1_id,
                "match_id": match_id,
                "s1_name": str(r1.get("business_name", "")),
                "match_name": str(r2.get("business_name", "")),
                "s1_addr": str(r1.get("business_address", "")),
                "match_addr": str(r2.get("business_address", "")),
                "s1_country": str(r1.get("country", "")),
                "match_country": str(r2.get("country", ""))
            })
            count += 1
            if count >= 60:
                break
    if count >= 60:
        break

print(f"\nSampled {len(sample_pairs)} matched pairs.")
with open("sample_pairs.json", "w", encoding="utf-8") as f:
    json.dump(sample_pairs, f, indent=2)

print("Fast EDA Done!")
